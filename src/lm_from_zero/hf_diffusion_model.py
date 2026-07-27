"""Self-contained Transformers model copied into masked diffusion exports."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput

from .hf_diffusion_config import LLaDAConfig


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


def _rotate_half(values: Tensor) -> Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    """Base rotary frequencies for the bidirectional attention blocks."""

    inv_freq: Tensor

    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        frequency_ids = torch.arange(0, head_dim, 2, dtype=torch.float32)
        inv_freq = 1.0 / (theta ** (frequency_ids / head_dim))
        # Persist this derived buffer because Transformers may instantiate remote
        # models on the meta device and cannot reconstruct non-persistent values
        # when materializing the module during ``from_pretrained``.
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    @torch.no_grad()
    def forward(self, position_ids: Tensor) -> tuple[Tensor, Tensor]:
        frequencies = torch.einsum(
            "bd,bs->bsd",
            self.inv_freq.float().expand(position_ids.shape[0], -1),
            position_ids.float(),
        )
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        return embeddings.cos(), embeddings.sin()


def _apply_rotary(
    query: Tensor,
    key: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> tuple[Tensor, Tensor]:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    query_dtype = query.dtype
    key_dtype = key.dtype
    query = query * cosine + _rotate_half(query) * sine
    key = key * cosine + _rotate_half(key) * sine
    return query.to(query_dtype), key.to(key_dtype)


class DiffusionAttention(nn.Module):
    """Bias-free bidirectional multi-head attention."""

    def __init__(self, config: LLaDAConfig) -> None:
        super().__init__()
        self.config = config
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        cosine: Tensor,
        sine: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size, sequence_length, _ = hidden_states.shape
        projected_shape = (
            batch_size,
            sequence_length,
            self.config.num_attention_heads,
            self.config.head_dim,
        )
        query = self.q_proj(hidden_states).view(projected_shape).transpose(1, 2)
        key = self.k_proj(hidden_states).view(projected_shape).transpose(1, 2)
        value = self.v_proj(hidden_states).view(projected_shape).transpose(1, 2)
        query, key = _apply_rotary(query, key, cosine, sine)
        mask = (
            None
            if attention_mask is None
            else attention_mask.to(
                device=hidden_states.device,
                dtype=torch.bool,
            ).view(batch_size, 1, 1, sequence_length)
        )
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=mask,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        flattened = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.config.hidden_size)
        )
        output: Tensor = self.o_proj(flattened)
        return output


class DiffusionMLP(nn.Module):
    """Bias-free SwiGLU feed-forward branch."""

    def __init__(self, config: LLaDAConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_proj = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        output: Tensor = self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        return output


class DiffusionBlock(nn.Module):
    """Pre-RMSNorm attention and SwiGLU residual branches."""

    def __init__(self, config: LLaDAConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = DiffusionAttention(config)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            config.rms_norm_eps,
        )
        self.mlp = DiffusionMLP(config)

    def forward(
        self,
        hidden_states: Tensor,
        cosine: Tensor,
        sine: Tensor,
        *,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden_states = hidden_states + self.self_attn(
            self.input_layernorm(hidden_states),
            cosine,
            sine,
            attention_mask=attention_mask,
        )
        output: Tensor = hidden_states + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return output


def _masked_diffusion_loss(
    logits: Tensor,
    labels: Tensor,
    eligible_mask: Tensor,
    time: Tensor,
) -> Tensor:
    eligible = eligible_mask.to(device=logits.device, dtype=torch.bool)
    supervised = labels != -100
    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    weighted = token_losses * supervised / time.to(logits.device).unsqueeze(1)
    eligible_counts = eligible.sum(dim=1)
    nonempty = eligible.any(dim=1)
    if not torch.any(nonempty):
        return logits.sum() * 0.0
    per_example = weighted.sum(dim=1) / eligible_counts.clamp_min(1)
    return per_example[nonempty].mean()


class LLaDAForMaskedDiffusion(PreTrainedModel):  # type: ignore[no-untyped-call]
    """Transformers-compatible project-owned masked diffusion denoiser."""

    config_class = LLaDAConfig
    base_model_prefix = ""
    main_input_name = "input_ids"
    _supports_sdpa = True

    def __init__(self, config: LLaDAConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )
        self.layers = nn.ModuleList(
            DiffusionBlock(config) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_embedding = RotaryEmbedding(config.head_dim, config.rope_theta)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()  # type: ignore[no-untyped-call]
        with torch.no_grad():
            self.embed_tokens.weight[config.pad_token_id].zero_()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear | nn.Embedding):
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        eligible_mask: Tensor | None = None,
        time: Tensor | None = None,
        return_dict: bool | None = None,
        **_: Any,
    ) -> MaskedLMOutput | tuple[Tensor, ...]:
        if position_ids is None:
            position_ids = torch.arange(
                input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            ).expand(input_ids.shape[0], -1)
        hidden_states = self.embed_tokens(input_ids)
        cosine, sine = self.rotary_embedding(position_ids)
        resolved_mask = (
            None
            if attention_mask is None
            else attention_mask.to(device=input_ids.device, dtype=torch.bool)
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cosine,
                sine,
                attention_mask=resolved_mask,
            )
        logits = self.lm_head(self.norm(hidden_states))
        arguments = (labels, eligible_mask, time)
        if all(argument is None for argument in arguments):
            loss = None
        elif any(argument is None for argument in arguments):
            raise ValueError(
                "labels, eligible_mask, and time are all required for diffusion loss"
            )
        else:
            assert labels is not None
            assert eligible_mask is not None
            assert time is not None
            loss = _masked_diffusion_loss(logits, labels, eligible_mask, time)
        if return_dict is False:
            return (logits,) if loss is None else (loss, logits)
        return MaskedLMOutput(loss=cast(Any, loss), logits=logits)
