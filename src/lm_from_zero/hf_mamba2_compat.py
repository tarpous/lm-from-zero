"""Hugging Face Mamba-2 compatibility model with official grouped RMSNorm."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import Mamba2Config
from transformers import Mamba2ForCausalLM as TransformersMamba2ForCausalLM


class GroupedGatedRMSNorm(nn.Module):
    """Apply SiLU gating followed by RMS normalization within fixed groups."""

    def __init__(self, hidden_size: int, group_size: int, eps: float) -> None:
        super().__init__()
        if hidden_size <= 0 or group_size <= 0 or hidden_size % group_size != 0:
            raise ValueError("group size must evenly divide the hidden size")
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.group_size = group_size
        self.eps = eps

    def forward(self, hidden_states: Tensor, gate: Tensor | None = None) -> Tensor:
        input_dtype = hidden_states.dtype
        values = hidden_states.float()
        if gate is not None:
            values = values * F.silu(gate.float())
        grouped = values.view(*values.shape[:-1], -1, self.group_size)
        normalized = grouped * torch.rsqrt(
            grouped.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized.flatten(start_dim=-2) * self.weight.float()).to(input_dtype)


class GroupedMamba2ForCausalLM(TransformersMamba2ForCausalLM):
    """Transformers Mamba-2 with the official grouped gated RMSNorm restored."""

    def __init__(self, config: Mamba2Config) -> None:
        super().__init__(config)  # type: ignore[no-untyped-call]
        hidden_size = int(self.config.hidden_size * self.config.expand)
        group_size = int(getattr(self.config, "rms_norm_group_size", hidden_size))
        for layer in self.backbone.layers:
            mixer = cast(Any, layer.mixer)
            existing = cast(Any, mixer.norm)
            replacement = GroupedGatedRMSNorm(
                hidden_size,
                group_size,
                float(existing.variance_epsilon),
            )
            replacement.to(device=existing.weight.device, dtype=existing.weight.dtype)
            with torch.no_grad():
                replacement.weight.copy_(existing.weight)
            mixer.norm = replacement
