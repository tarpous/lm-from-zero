"""Validated model configuration and analytic size estimates."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lm_from_zero.tokenizer.bpe import SPECIAL_TOKEN_IDS

SHA256_PATTERN = r"^[0-9a-f]{64}$"


def fixed_special_tokens_hash() -> str:
    """Hash the fixed special-token mapping used by every model family."""

    encoded = json.dumps(
        SPECIAL_TOKEN_IDS,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


class DenseParameterBreakdown(BaseModel):
    """Exact trainable-parameter accounting for the dense architecture."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    token_embeddings: Annotated[int, Field(ge=0)]
    attention_projections: Annotated[int, Field(ge=0)]
    mlp_projections: Annotated[int, Field(ge=0)]
    normalization_scales: Annotated[int, Field(ge=0)]
    output_head: Annotated[int, Field(ge=0)]
    total: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        components = (
            self.token_embeddings
            + self.attention_projections
            + self.mlp_projections
            + self.normalization_scales
            + self.output_head
        )
        if components != self.total:
            raise ValueError("parameter components do not equal the total")
        return self


class DenseFlopEstimate(BaseModel):
    """Analytic forward-pass FLOPs under a multiply-add-equals-two policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sequence_length: Annotated[int, Field(gt=0)]
    projection_flops_per_token: Annotated[int, Field(gt=0)]
    attention_flops_per_token: Annotated[int, Field(gt=0)]
    total_flops_per_token: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        if (
            self.projection_flops_per_token + self.attention_flops_per_token
            != self.total_flops_per_token
        ):
            raise ValueError("FLOP components do not equal the total")
        return self


class Olmo2Config(BaseModel):
    """Resolved configuration contract for the project-owned dense model."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-model-config"] = "lm-from-zero-model-config"
    format_version: Literal[1] = 1
    model_name: Annotated[str, Field(min_length=1)] = "zero-20m-tinystories"
    architecture: Literal["olmo2"] = "olmo2"
    objective: Literal["causal_lm"] = "causal_lm"
    export_architecture: Literal["Olmo2ForCausalLM"] = "Olmo2ForCausalLM"
    normalization: Literal["olmo2_branch_post_norm"] = "olmo2_branch_post_norm"
    attention_backend: Literal["sdpa"] = "sdpa"
    dtype_policy: Literal["bf16_mixed"] = "bf16_mixed"
    cache_policy: Literal["dynamic_kv"] = "dynamic_kv"
    tokenizer_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    special_tokens_hash: Annotated[str, Field(pattern=SHA256_PATTERN)] = Field(
        default_factory=fixed_special_tokens_hash
    )
    vocab_size: Annotated[int, Field(gt=8, le=65536)] = 16_000
    num_hidden_layers: Annotated[int, Field(gt=0)] = 5
    hidden_size: Annotated[int, Field(gt=0)] = 384
    num_attention_heads: Annotated[int, Field(gt=0)] = 6
    num_key_value_heads: Annotated[int, Field(gt=0)] = 2
    intermediate_size: Annotated[int, Field(gt=0)] = 1_024
    max_position_embeddings: Annotated[int, Field(gt=0)] = 1_024
    rope_theta: Annotated[float, Field(gt=0)] = 10_000.0
    rms_norm_eps: Annotated[float, Field(gt=0)] = 1e-5
    initializer_range: Annotated[float, Field(gt=0)] = 0.02
    attention_dropout: Annotated[float, Field(ge=0, lt=1)] = 0.0
    pad_token_id: Literal[0] = 0
    bos_token_id: Literal[1] = 1
    eos_token_id: Literal[2] = 2
    tie_word_embeddings: Literal[False] = False
    attention_bias: Literal[False] = False
    use_cache: bool = True

    @model_validator(mode="after")
    def validate_dimensions_and_tokens(self) -> Self:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                "num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.head_dim % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.special_tokens_hash != fixed_special_tokens_hash():
            raise ValueError("special-token hash does not match the fixed mapping")
        if self.eos_token_id >= self.vocab_size:
            raise ValueError("special-token IDs must fit in the vocabulary")
        return self

    @property
    def head_dim(self) -> int:
        """Return the per-head projection width."""

        return self.hidden_size // self.num_attention_heads

    @property
    def key_value_size(self) -> int:
        """Return the flattened width of grouped keys and values."""

        return self.num_key_value_heads * self.head_dim

    def canonical_json(self) -> str:
        """Return a deterministic serialized configuration."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def config_hash(self) -> str:
        """Return the canonical configuration SHA-256."""

        return sha256(self.canonical_json().encode()).hexdigest()

    def parameter_breakdown(self) -> DenseParameterBreakdown:
        """Compute exact trainable parameters without allocating a model."""

        hidden = self.hidden_size
        key_value = self.key_value_size
        layers = self.num_hidden_layers
        embeddings = self.vocab_size * hidden
        attention = layers * (
            hidden * hidden + 2 * hidden * key_value + hidden * hidden
        )
        mlp = layers * 3 * hidden * self.intermediate_size
        norms = layers * (hidden + key_value + 2 * hidden) + hidden
        output = self.vocab_size * hidden
        total = embeddings + attention + mlp + norms + output
        return DenseParameterBreakdown(
            token_embeddings=embeddings,
            attention_projections=attention,
            mlp_projections=mlp,
            normalization_scales=norms,
            output_head=output,
            total=total,
        )

    def forward_flops(self, sequence_length: int) -> DenseFlopEstimate:
        """Estimate a full-context forward pass per token."""

        if not 0 < sequence_length <= self.max_position_embeddings:
            raise ValueError("sequence_length is outside the configured context")
        parameters = self.parameter_breakdown()
        projection_parameters = (
            parameters.attention_projections
            + parameters.mlp_projections
            + parameters.output_head
        )
        projection_flops = 2 * projection_parameters
        attention_flops = (
            4 * self.num_hidden_layers * sequence_length * self.hidden_size
        )
        return DenseFlopEstimate(
            sequence_length=sequence_length,
            projection_flops_per_token=projection_flops,
            attention_flops_per_token=attention_flops,
            total_flops_per_token=projection_flops + attention_flops,
        )
