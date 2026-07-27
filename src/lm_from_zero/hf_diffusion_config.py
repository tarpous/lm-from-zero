"""Self-contained Transformers configuration for masked diffusion exports."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class LLaDAConfig(PretrainedConfig):  # type: ignore[no-untyped-call]
    """Configuration for the exported project-owned bidirectional denoiser."""

    model_type = "lm_from_zero_llada"

    def __init__(
        self,
        *,
        vocab_size: int = 16_000,
        hidden_size: int = 384,
        intermediate_size: int = 1_152,
        num_hidden_layers: int = 4,
        num_attention_heads: int = 6,
        max_position_embeddings: int = 1_024,
        rope_theta: float = 10_000.0,
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        attention_dropout: float = 0.0,
        corruption_epsilon: float = 1e-3,
        mask_token_id: int = 7,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_word_embeddings: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.corruption_epsilon = corruption_epsilon
        self.mask_token_id = mask_token_id

    @property
    def head_dim(self) -> int:
        """Return the per-head projection width."""

        return self.hidden_size // self.num_attention_heads
