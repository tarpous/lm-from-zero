"""Project-owned LLaDA-style masked discrete-diffusion language model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lm_from_zero.models.config import MaskedDiffusionConfig
from lm_from_zero.models.interfaces import MaskedDiffusionOutput
from lm_from_zero.models.olmo2 import (
    RMSNorm,
    RotaryEmbedding,
    apply_rotary_embedding,
)


@dataclass(frozen=True, slots=True)
class DiffusionCorruptionBatch:
    """One deterministic Monte Carlo sample of the diffusion objective."""

    input_ids: Tensor
    labels: Tensor
    eligible_mask: Tensor
    corrupted_mask: Tensor
    time: Tensor


def base_pretraining_eligible_mask(
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    pad_token_id: int = 0,
    bos_token_id: int = 1,
) -> Tensor:
    """Protect padding and BOS while supervising ordinary content and EOS."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must use torch.long token IDs")
    if attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    return (
        attention_mask.to(device=input_ids.device, dtype=torch.bool)
        & (input_ids != pad_token_id)
        & (input_ids != bos_token_id)
    )


def corrupt_for_diffusion(
    input_ids: Tensor,
    eligible_mask: Tensor,
    *,
    mask_token_id: int,
    epsilon: float = 1e-3,
    generator: torch.Generator | None = None,
) -> DiffusionCorruptionBatch:
    """Sample continuous time and independently mask eligible target tokens."""

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if input_ids.dtype != torch.long:
        raise ValueError("input_ids must use torch.long token IDs")
    if eligible_mask.shape != input_ids.shape:
        raise ValueError("eligible_mask must match input_ids")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon must be between zero and one")
    if mask_token_id < 0:
        raise ValueError("mask_token_id must be non-negative")

    eligible = eligible_mask.to(device=input_ids.device, dtype=torch.bool)
    batch_size = input_ids.shape[0]
    time = epsilon + (1.0 - epsilon) * torch.rand(
        batch_size,
        device=input_ids.device,
        dtype=torch.float32,
        generator=generator,
    )
    uniforms = torch.rand(
        input_ids.shape,
        device=input_ids.device,
        dtype=torch.float32,
        generator=generator,
    )
    corrupted = eligible & (uniforms < time.unsqueeze(1))
    nonempty = eligible.any(dim=1)
    missing = nonempty & ~corrupted.any(dim=1)
    fallback = uniforms.masked_fill(~eligible, torch.inf).argmin(dim=1)
    forced = F.one_hot(fallback, num_classes=input_ids.shape[1]).to(dtype=torch.bool)
    corrupted = corrupted | (forced & missing.unsqueeze(1))

    corrupted_input = input_ids.clone()
    corrupted_input[corrupted] = mask_token_id
    labels = torch.full_like(input_ids, -100)
    labels[corrupted] = input_ids[corrupted]
    return DiffusionCorruptionBatch(
        input_ids=corrupted_input,
        labels=labels,
        eligible_mask=eligible,
        corrupted_mask=corrupted,
        time=time,
    )


def masked_diffusion_loss(
    logits: Tensor,
    labels: Tensor,
    eligible_mask: Tensor,
    time: Tensor,
    *,
    validate: bool = True,
) -> Tensor:
    """Return the per-example eligible-normalized, ``1/t``-weighted loss."""

    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocabulary]")
    if labels.shape != logits.shape[:2] or eligible_mask.shape != labels.shape:
        raise ValueError("labels and eligible_mask must match logits")
    if labels.dtype != torch.long:
        raise ValueError("labels must use torch.long token IDs")
    if time.shape != (logits.shape[0],):
        raise ValueError("time must contain one value per example")
    if validate and (
        not bool(torch.isfinite(time).all())
        or bool(torch.any((time <= 0) | (time > 1)))
    ):
        raise ValueError("time values must be finite and in (0, 1]")

    eligible = eligible_mask.to(device=logits.device, dtype=torch.bool)
    supervised = labels != -100
    nonempty = eligible.any(dim=1)
    if validate:
        if torch.any(supervised & ~eligible):
            raise ValueError("supervised labels must be a subset of eligible positions")
        if torch.any(nonempty & ~supervised.any(dim=1)):
            raise ValueError(
                "every non-empty example must supervise at least one token"
            )
        if torch.any(supervised & ((labels < 0) | (labels >= logits.shape[-1]))):
            raise ValueError("labels contain a token outside the vocabulary")

    token_losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]).float(),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    weighted = token_losses * supervised / time.to(logits.device).unsqueeze(1)
    eligible_counts = eligible.sum(dim=1)
    per_example = weighted.sum(dim=1) / eligible_counts.clamp_min(1)
    return (per_example * nonempty).sum() / nonempty.sum().clamp_min(1)


def _linear_masked_diffusion_loss(
    hidden_states: Tensor,
    linear_weight: Tensor,
    labels: Tensor,
    eligible_mask: Tensor,
    time: Tensor,
) -> Tensor:
    """Compute the diffusion objective without full vocabulary logits."""

    token_losses = F.linear_cross_entropy(
        hidden_states.reshape(-1, hidden_states.shape[-1]),
        linear_weight,
        labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(labels)
    eligible = eligible_mask.to(device=hidden_states.device, dtype=torch.bool)
    supervised = labels != -100
    weighted = token_losses * supervised / time.to(hidden_states.device).unsqueeze(1)
    counts = eligible.sum(dim=1)
    nonempty = counts > 0
    per_example = weighted.sum(dim=1) / counts.clamp_min(1)
    return (per_example * nonempty).sum() / nonempty.sum().clamp_min(1)


class DiffusionAttention(nn.Module):
    """Bias-free multi-head bidirectional self-attention."""

    def __init__(self, config: MaskedDiffusionConfig) -> None:
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
        query, key = apply_rotary_embedding(query, key, cosine, sine)
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

    def __init__(self, config: MaskedDiffusionConfig) -> None:
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
        output: Tensor = self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
        return output


class DiffusionBlock(nn.Module):
    """Pre-RMSNorm bidirectional attention and SwiGLU residual branches."""

    def __init__(self, config: MaskedDiffusionConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = DiffusionAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
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


class MaskedDiffusionForMaskedLM(nn.Module):
    """Untied bidirectional denoiser with no time input or causal cache."""

    def __init__(self, config: MaskedDiffusionConfig) -> None:
        super().__init__()
        self.config = config
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
        self.apply(self._initialize_module)
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

    def _validate_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None,
        position_ids: Tensor | None,
    ) -> tuple[Tensor | None, Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.dtype != torch.long:
            raise ValueError("input_ids must use torch.long token IDs")
        if input_ids.shape[1] == 0:
            raise ValueError("input sequence cannot be empty")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("input exceeds the configured context length")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contain a token outside the vocabulary")
        resolved_mask = None
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids")
            resolved_mask = attention_mask.to(
                device=input_ids.device,
                dtype=torch.bool,
            )
            if torch.any(~resolved_mask.any(dim=1)):
                raise ValueError("each example must contain at least one visible token")
        if position_ids is None:
            positions = torch.arange(
                input_ids.shape[1],
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
        return resolved_mask, positions

    def forward(
        self,
        input_ids: Tensor,
        *,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor | None = None,
        eligible_mask: Tensor | None = None,
        time: Tensor | None = None,
        loss_backend: str = "full",
        loss_only: bool = False,
        validate_inputs: bool = True,
    ) -> MaskedDiffusionOutput:
        """Run bidirectional logits and the optional masked objective."""

        if loss_backend not in {"full", "linear"}:
            raise ValueError("unsupported diffusion loss backend")
        if loss_only and (labels is None or loss_backend != "linear"):
            raise ValueError("loss-only forward requires labels and linear loss")
        if validate_inputs:
            resolved_mask, positions = self._validate_inputs(
                input_ids,
                attention_mask,
                position_ids,
            )
        else:
            resolved_mask = (
                None
                if attention_mask is None
                else attention_mask.to(device=input_ids.device, dtype=torch.bool)
            )
            positions = (
                torch.arange(
                    input_ids.shape[1],
                    device=input_ids.device,
                    dtype=torch.long,
                ).expand(input_ids.shape[0], -1)
                if position_ids is None
                else position_ids
            )
        hidden_states = self.embed_tokens(input_ids)
        cosine, sine = self.rotary_embedding(positions)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                cosine,
                sine,
                attention_mask=resolved_mask,
            )
        hidden_states = self.norm(hidden_states)
        logits = (
            hidden_states.new_empty((0,)) if loss_only else self.lm_head(hidden_states)
        )
        objective_arguments = (labels, eligible_mask, time)
        if all(argument is None for argument in objective_arguments):
            loss = None
        elif any(argument is None for argument in objective_arguments):
            raise ValueError(
                "labels, eligible_mask, and time are all required for diffusion loss"
            )
        else:
            assert labels is not None
            assert eligible_mask is not None
            assert time is not None
            loss = (
                _linear_masked_diffusion_loss(
                    hidden_states,
                    self.lm_head.weight,
                    labels,
                    eligible_mask,
                    time,
                )
                if loss_only
                else masked_diffusion_loss(
                    logits,
                    labels,
                    eligible_mask,
                    time,
                    validate=validate_inputs,
                )
            )
        return MaskedDiffusionOutput(logits=logits, loss=loss)

    def trainable_parameter_count(self) -> int:
        """Return the authoritative realized trainable parameter count."""

        return sum(parameter.numel() for parameter in self.parameters())
