"""Project-owned dense OLMo2-compatible causal language model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lm_from_zero.models.config import (
    DenseFlopEstimate,
    DenseParameterBreakdown,
    Olmo2Config,
)
from lm_from_zero.models.interfaces import CausalLMOutput, DenseKVCache, LayerKVCache

DenseModelVariant = Literal[
    "baseline",
    "learned_absolute_positions",
    "layer_norm",
    "gelu",
    "mha",
    "without_qk_norm",
    "tied_embeddings",
]

_DENSE_MODEL_VARIANTS: frozenset[str] = frozenset(
    {
        "baseline",
        "learned_absolute_positions",
        "layer_norm",
        "gelu",
        "mha",
        "without_qk_norm",
        "tied_embeddings",
    }
)


class RMSNorm(nn.Module):
    """RMS normalization with fp32 variance accumulation."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        variance = values.square().mean(dim=-1, keepdim=True)
        normalized = values * torch.rsqrt(variance + self.eps)
        return (self.weight * normalized).to(input_dtype)


def _validate_dense_model_variant(variant: str) -> None:
    if variant not in _DENSE_MODEL_VARIANTS:
        choices = ", ".join(sorted(_DENSE_MODEL_VARIANTS))
        raise ValueError(
            f"unsupported dense model variant {variant!r}; choices: {choices}"
        )


def dense_variant_parameter_breakdown(
    config: Olmo2Config,
    variant: DenseModelVariant = "baseline",
) -> DenseParameterBreakdown:
    """Return exact realized parameter accounting for one dense variant."""

    _validate_dense_model_variant(variant)
    base = config.parameter_breakdown()
    attention_delta = 0
    norm_delta = 0
    if variant == "mha":
        attention_delta = config.num_hidden_layers * (
            2 * config.hidden_size * (config.hidden_size - config.key_value_size)
        )
        norm_delta = config.num_hidden_layers * (
            config.hidden_size - config.key_value_size
        )
    elif variant == "without_qk_norm":
        norm_delta = -config.num_hidden_layers * (
            config.hidden_size + config.key_value_size
        )
    embedding_delta = (
        config.max_position_embeddings * config.hidden_size
        if variant == "learned_absolute_positions"
        else 0
    )
    output_delta = -base.output_head if variant == "tied_embeddings" else 0
    attention = base.attention_projections + attention_delta
    norms = base.normalization_scales + norm_delta
    embeddings = base.token_embeddings + embedding_delta
    output = base.output_head + output_delta
    total = embeddings + attention + base.mlp_projections + norms + output
    return base.model_copy(
        update={
            "token_embeddings": embeddings,
            "attention_projections": attention,
            "normalization_scales": norms,
            "output_head": output,
            "total": total,
        }
    )


def dense_variant_forward_flops(
    config: Olmo2Config,
    sequence_length: int,
    variant: DenseModelVariant = "baseline",
) -> DenseFlopEstimate:
    """Return the analytic forward FLOPs for one dense variant."""

    _validate_dense_model_variant(variant)
    base = config.forward_flops(sequence_length)
    attention_delta = 0
    if variant == "mha":
        attention_delta = (
            2
            * config.num_hidden_layers
            * config.hidden_size
            * (config.hidden_size - config.key_value_size)
        )
    projection = base.projection_flops_per_token + attention_delta
    return base.model_copy(
        update={
            "projection_flops_per_token": projection,
            "total_flops_per_token": projection + base.attention_flops_per_token,
        }
    )


def _make_norm(
    norm_type: type[nn.Module],
    hidden_size: int,
    eps: float,
    *,
    disabled: bool = False,
) -> nn.Module:
    if disabled:
        return nn.Identity()
    if norm_type is nn.LayerNorm:
        # Keep the variant bias-free, like the canonical OLMo2 projections.
        return nn.LayerNorm(hidden_size, eps=eps, elementwise_affine=True, bias=False)
    return RMSNorm(hidden_size, eps)


def rotate_half(values: Tensor) -> Tensor:
    """Rotate the two RoPE half-vectors."""

    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    """Base RoPE frequencies matching the OLMo2 export convention."""

    inv_freq: Tensor

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        frequency_ids = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (frequency_ids / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [batch, sequence]")
        frequencies = torch.einsum(
            "bd,bs->bsd",
            self.inv_freq.float().expand(position_ids.shape[0], -1),
            position_ids.float(),
        )
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos(), embeddings.sin()


def apply_rotary_embedding(
    query: Tensor,
    key: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply precomputed RoPE values to query and grouped key heads."""

    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    query_dtype = query.dtype
    key_dtype = key.dtype
    rotated_query = query * cosine + rotate_half(query) * sine
    rotated_key = key * cosine + rotate_half(key) * sine
    return rotated_query.to(query_dtype), rotated_key.to(key_dtype)


class Olmo2Attention(nn.Module):
    """Bias-free grouped-query attention with flat QK normalization."""

    def __init__(
        self,
        config: Olmo2Config,
        variant: DenseModelVariant = "baseline",
    ) -> None:
        super().__init__()
        _validate_dense_model_variant(variant)
        self.config = config
        self.variant = variant
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = (
            config.num_attention_heads
            if variant == "mha"
            else config.num_key_value_heads
        )
        self.key_value_size = self.num_key_value_heads * self.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.key_value_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.key_value_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        norm_type = nn.LayerNorm if variant == "layer_norm" else RMSNorm
        self.q_norm: RMSNorm = cast(
            RMSNorm,
            _make_norm(
                norm_type,
                config.hidden_size,
                config.rms_norm_eps,
                disabled=variant == "without_qk_norm",
            ),
        )
        self.k_norm: RMSNorm = cast(
            RMSNorm,
            _make_norm(
                norm_type,
                self.key_value_size,
                config.rms_norm_eps,
                disabled=variant == "without_qk_norm",
            ),
        )

    def _validate_past(
        self,
        past: LayerKVCache | None,
        batch_size: int,
    ) -> int:
        if past is None:
            return 0
        expected_prefix = (
            batch_size,
            self.num_key_value_heads,
        )
        if past.key.ndim != 4 or past.value.ndim != 4:
            raise ValueError("cached keys and values must be rank four")
        if past.key.shape != past.value.shape:
            raise ValueError("cached key and value shapes must match")
        if past.key.shape[:2] != expected_prefix:
            raise ValueError("cached batch or key/value head count is incompatible")
        if past.key.shape[-1] != self.head_dim:
            raise ValueError("cached head dimension is incompatible")
        return past.key.shape[2]

    def _attention_mask(
        self,
        attention_mask: Tensor | None,
        *,
        batch_size: int,
        query_length: int,
        key_length: int,
        past_length: int,
        device: torch.device,
    ) -> Tensor | None:
        if attention_mask is None and past_length == 0:
            return None
        query_positions = torch.arange(
            past_length,
            past_length + query_length,
            device=device,
        )
        key_positions = torch.arange(key_length, device=device)
        allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        allowed = allowed.view(1, 1, query_length, key_length)
        if attention_mask is None:
            return allowed
        if attention_mask.shape != (batch_size, key_length):
            raise ValueError(
                "attention_mask must cover the batch and complete key history"
            )
        return allowed & attention_mask.to(device=device, dtype=torch.bool).view(
            batch_size, 1, 1, key_length
        )

    def forward(
        self,
        hidden_states: Tensor,
        cosine: Tensor | None,
        sine: Tensor | None,
        *,
        attention_mask: Tensor | None = None,
        past: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerKVCache | None]:
        batch_size, query_length, _ = hidden_states.shape
        past_length = self._validate_past(past, batch_size)

        query = self.q_norm(self.q_proj(hidden_states))
        key = self.k_norm(self.k_proj(hidden_states))
        value = self.v_proj(hidden_states)
        query = query.view(
            batch_size,
            query_length,
            self.num_attention_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key.view(
            batch_size,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.view(
            batch_size,
            query_length,
            self.num_key_value_heads,
            self.head_dim,
        ).transpose(1, 2)
        if (cosine is None) != (sine is None):
            raise ValueError("cosine and sine must be provided together")
        if cosine is not None and sine is not None:
            query, key = apply_rotary_embedding(query, key, cosine, sine)

        if past is not None:
            key = torch.cat((past.key, key), dim=2)
            value = torch.cat((past.value, value), dim=2)
        key_length = key.shape[2]
        mask = self._attention_mask(
            attention_mask,
            batch_size=batch_size,
            query_length=query_length,
            key_length=key_length,
            past_length=past_length,
            device=hidden_states.device,
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=mask is None,
            enable_gqa=self.num_attention_heads != self.num_key_value_heads,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, query_length, self.config.hidden_size)
        )
        output = self.o_proj(attended)
        new_cache = LayerKVCache(key=key, value=value) if use_cache else None
        return output, new_cache


class SwiGLU(nn.Module):
    """Bias-free OLMo2 feed-forward branch."""

    def __init__(self, config: Olmo2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        gated: Tensor = F.silu(self.gate_proj(hidden_states))
        up: Tensor = self.up_proj(hidden_states)
        output: Tensor = self.down_proj(gated * up)
        return output


class GELUMLP(nn.Module):
    """Bias-free two-projection GELU feed-forward branch.

    The intermediate width is selected by the caller so the ablation can keep
    the feed-forward parameter budget close to the canonical three-projection
    SwiGLU branch.
    """

    def __init__(self, config: Olmo2Config) -> None:
        super().__init__()
        intermediate_size = (3 * config.intermediate_size) // 2
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: Tensor) -> Tensor:
        activated = F.gelu(self.up_proj(hidden_states))
        output: Tensor = self.down_proj(activated)
        return output


class Olmo2DecoderLayer(nn.Module):
    """Attention and MLP branches with OLMo2 post-branch normalization."""

    def __init__(
        self,
        config: Olmo2Config,
        variant: DenseModelVariant = "baseline",
    ) -> None:
        super().__init__()
        _validate_dense_model_variant(variant)
        self.variant = variant
        self.self_attn = Olmo2Attention(config, variant)
        # Keep the public branch contract typed as the canonical SwiGLU.  The
        # GELU ablation exposes the same up/down projection names while
        # intentionally omitting the gate projection at runtime.
        self.mlp: SwiGLU = (
            GELUMLP(config) if variant == "gelu" else SwiGLU(config)  # type: ignore[assignment]
        )
        norm_type = nn.LayerNorm if variant == "layer_norm" else RMSNorm
        self.post_attention_layernorm = _make_norm(
            norm_type, config.hidden_size, config.rms_norm_eps
        )
        self.post_feedforward_layernorm = _make_norm(
            norm_type, config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: Tensor,
        cosine: Tensor | None,
        sine: Tensor | None,
        *,
        attention_mask: Tensor | None = None,
        past: LayerKVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, LayerKVCache | None]:
        residual = hidden_states
        attention_output, new_cache = self.self_attn(
            hidden_states,
            cosine,
            sine,
            attention_mask=attention_mask,
            past=past,
            use_cache=use_cache,
        )
        hidden_states = residual + self.post_attention_layernorm(attention_output)

        residual = hidden_states
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + self.post_feedforward_layernorm(mlp_output)
        return hidden_states, new_cache


def _causal_loss(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.shape[:2] != labels.shape:
        raise ValueError("labels must match input batch and sequence dimensions")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long token IDs")
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if shift_labels.numel() == 0:
        return shift_logits.sum() * 0.0
    losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    )
    valid = shift_labels.view(-1) != -100
    return (losses * valid).sum() / valid.sum().clamp_min(1)


def _linear_causal_loss(
    hidden_states: Tensor,
    linear_weight: Tensor,
    labels: Tensor,
) -> Tensor:
    """Compute shifted causal CE without materializing vocabulary logits."""

    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("labels must match input batch and sequence dimensions")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long token IDs")
    shifted_hidden = hidden_states[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    if shifted_labels.numel() == 0:
        return shifted_hidden.sum() * 0.0
    losses = F.linear_cross_entropy(
        shifted_hidden.view(-1, shifted_hidden.shape[-1]),
        linear_weight,
        shifted_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    )
    valid = shifted_labels.view(-1) != -100
    return (losses * valid).sum() / valid.sum().clamp_min(1)


class Olmo2ForCausalLM(nn.Module):
    """Untied project-owned OLMo2-compatible causal language model."""

    def __init__(
        self,
        config: Olmo2Config,
        *,
        variant: DenseModelVariant = "baseline",
    ) -> None:
        super().__init__()
        _validate_dense_model_variant(variant)
        self.config = config
        self.variant = variant
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            Olmo2DecoderLayer(config, variant) for _ in range(config.num_hidden_layers)
        )
        self.norm = _make_norm(
            nn.LayerNorm if variant == "layer_norm" else RMSNorm,
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.position_embeddings: nn.Embedding | None = None
        if variant == "learned_absolute_positions":
            self.position_embeddings = nn.Embedding(
                config.max_position_embeddings, config.hidden_size
            )
            self.rotary_embedding: RotaryEmbedding | None = None
        else:
            self.rotary_embedding = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize_module)
        if variant == "tied_embeddings":
            self.lm_head.weight = self.embed_tokens.weight
        with torch.no_grad():
            self.embed_tokens.weight[config.pad_token_id].zero_()

    def _initialize_module(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _validate_inputs(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None,
        cache: DenseKVCache | None,
    ) -> tuple[int, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long token IDs")
        if input_ids.shape[1] == 0:
            raise ValueError("input sequence cannot be empty")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contain a token outside the vocabulary")
        if cache is not None and len(cache) != len(self.layers):
            raise ValueError("cache layer count does not match the model")

        past_length = 0 if cache is None else cache_sequence_length(cache)
        total_length = past_length + input_ids.shape[1]
        if total_length > self.config.max_position_embeddings:
            raise ValueError("input and cache exceed the configured context length")
        if position_ids is None:
            positions = torch.arange(
                past_length,
                total_length,
                device=input_ids.device,
                dtype=torch.long,
            ).expand(input_ids.shape[0], -1)
        else:
            if position_ids.shape != input_ids.shape:
                raise ValueError("position_ids must match input_ids")
            if position_ids.dtype != torch.long:
                raise ValueError("position_ids must use torch.long indices")
            if torch.any(position_ids < 0) or torch.any(
                position_ids >= self.config.max_position_embeddings
            ):
                raise ValueError("position_ids are outside the configured context")
            positions = position_ids
        return past_length, positions

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        cache: DenseKVCache | None = None,
        use_cache: bool = False,
        loss_backend: str = "full",
        loss_only: bool = False,
        validate_inputs: bool = True,
    ) -> CausalLMOutput:
        """Run causal logits, optional shifted loss, and optional KV caching."""

        if loss_backend not in {"full", "linear"}:
            raise ValueError("unsupported causal loss backend")
        if loss_only and (labels is None or loss_backend != "linear"):
            raise ValueError("loss-only forward requires labels and linear loss")
        if validate_inputs:
            past_length, positions = self._validate_inputs(
                input_ids, position_ids, cache
            )
        else:
            past_length = 0 if cache is None else cache_sequence_length(cache)
            positions = (
                torch.arange(
                    past_length,
                    past_length + input_ids.shape[1],
                    device=input_ids.device,
                    dtype=torch.long,
                ).expand(input_ids.shape[0], -1)
                if position_ids is None
                else position_ids
            )
        total_length = past_length + input_ids.shape[1]
        if attention_mask is not None and attention_mask.shape != (
            input_ids.shape[0],
            total_length,
        ):
            raise ValueError(
                "attention_mask must cover the batch and complete token history"
            )

        hidden_states = self.embed_tokens(input_ids)
        if self.rotary_embedding is None:
            assert self.position_embeddings is not None
            hidden_states = hidden_states + self.position_embeddings(positions)
            cosine = sine = None
        else:
            cosine, sine = self.rotary_embedding(positions)
        new_cache: list[LayerKVCache] = []
        for layer_index, layer in enumerate(self.layers):
            layer_past = None if cache is None else cache[layer_index]
            hidden_states, layer_cache = layer(
                hidden_states,
                cosine,
                sine,
                attention_mask=attention_mask,
                past=layer_past,
                use_cache=use_cache,
            )
            if layer_cache is not None:
                new_cache.append(layer_cache)
        hidden_states = self.norm(hidden_states)
        loss: Tensor | None
        if loss_only:
            assert labels is not None
            loss = _linear_causal_loss(hidden_states, self.lm_head.weight, labels)
            logits = hidden_states.new_empty((0,))
        else:
            logits = self.lm_head(hidden_states)
            loss = None if labels is None else _causal_loss(logits, labels)
        output_cache: DenseKVCache | None = tuple(new_cache) if use_cache else None
        return CausalLMOutput(logits=logits, loss=loss, cache=output_cache)

    def trainable_parameter_count(self) -> int:
        """Return the authoritative realized trainable parameter count."""

        return sum(parameter.numel() for parameter in self.parameters())


def cache_sequence_length(cache: Sequence[LayerKVCache]) -> int:
    """Validate a dense cache and return its common sequence length."""

    if not cache:
        return 0
    if any(layer.key.ndim != 4 or layer.value.ndim != 4 for layer in cache):
        raise ValueError("cached keys and values must be rank four")
    if any(layer.key.shape != layer.value.shape for layer in cache):
        raise ValueError("cached key and value shapes must match")
    lengths = {layer.key.shape[2] for layer in cache}
    if len(lengths) != 1:
        raise ValueError("dense cache layers have different sequence lengths")
    return next(iter(lengths))
