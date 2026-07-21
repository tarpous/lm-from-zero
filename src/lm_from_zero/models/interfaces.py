"""Shared typed model outputs and architecture-specific cache state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from torch import Tensor

from lm_from_zero.models.config import Olmo2Config


@dataclass(frozen=True, slots=True)
class LayerKVCache:
    """One dense layer's grouped key/value history."""

    key: Tensor
    value: Tensor


DenseKVCache = tuple[LayerKVCache, ...]


@dataclass(frozen=True, slots=True)
class CausalLMOutput:
    """Common causal-language-model forward result."""

    logits: Tensor
    loss: Tensor | None = None
    cache: DenseKVCache | None = None


class CausalLanguageModel(Protocol):
    """Interface consumed by causal training and generation code."""

    config: Olmo2Config

    def forward(
        self,
        input_ids: Tensor,
        *,
        labels: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        cache: DenseKVCache | None = None,
        use_cache: bool = False,
    ) -> CausalLMOutput:
        """Run a causal forward pass."""
