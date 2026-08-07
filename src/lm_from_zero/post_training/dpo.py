"""Project-owned preference records, rendering, and the DPO objective."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Annotated, Final, Literal, Self

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from lm_from_zero.post_training.chat import (
    BOS_SPECIAL_TOKEN,
    DEFAULT_CHAT_TEMPLATE,
    END_SPECIAL_TOKEN,
    EOS_SPECIAL_TOKEN,
    ROLE_SPECIAL_TOKENS,
    ChatMessage,
    ChatRole,
    ChatTemplate,
    Conversation,
)
from lm_from_zero.tokenizer.bpe import SPECIAL_TOKEN_IDS, ByteBPE

DPO_FORMAT: Final = "lm-from-zero-dpo-config"
PREFERENCE_PAIR_FORMAT: Final = "lm-from-zero-preference-pair"
REFERENCE_CACHE_FORMAT: Final = "lm-from-zero-reference-logprob-cache"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class DPOFormatError(ValueError):
    """Raised when a preference example or DPO tensor violates its contract."""


class PreferencePair(BaseModel):
    """Canonical pair with one shared prompt and two assistant responses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-preference-pair"] = PREFERENCE_PAIR_FORMAT
    format_version: Literal[1] = 1
    prompt: Conversation
    chosen: ChatMessage
    rejected: ChatMessage

    @model_validator(mode="after")
    def validate_pair(self) -> Self:
        if self.prompt.messages[-1].role != "user":
            raise ValueError("preference prompts must end with a user turn")
        if self.chosen.role != "assistant" or self.rejected.role != "assistant":
            raise ValueError("chosen and rejected responses must be assistant turns")
        if self.chosen.content == self.rejected.content:
            raise ValueError("chosen and rejected responses must differ")
        return self


class DPOConfig(BaseModel):
    """Versioned DPO policy from the project plan."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-dpo-config"] = DPO_FORMAT
    format_version: Literal[1] = 1
    beta: Annotated[float, Field(gt=0)] = 0.1
    learning_rate: Annotated[float, Field(gt=0)] = 5e-7
    max_length: Annotated[int, Field(gt=1)] = 1_024
    epochs: Literal[1] = 1
    pair_count: Annotated[int, Field(gt=0)] = 50_000
    prompt_loss_masked: Literal[True] = True
    frozen_reference: Literal[True] = True


class ReferenceLogProbCacheIdentity(BaseModel):
    """Immutable identity shared by DPO and APO reference-log-prob caches."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-reference-logprob-cache"] = REFERENCE_CACHE_FORMAT
    format_version: Literal[1] = 1
    model_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    tokenizer_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    checkpoint_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    template_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    max_length: Annotated[int, Field(gt=1)]

    def canonical_json(self) -> str:
        """Return the stable cache identity encoding."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def cache_key(self) -> str:
        """Return the cache key; objective names intentionally do not enter it."""

        return sha256(self.canonical_json().encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class PreferenceSequence:
    """Tokenized prompt/response sequence with an assistant-response mask."""

    input_ids: tuple[int, ...]
    response_mask: tuple[bool, ...]
    template_hash: str
    prompt_prefix_length: int
    truncated: bool
    response_token_count: int

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.response_mask):
            raise DPOFormatError("input IDs and response mask must have equal lengths")
        if not self.input_ids:
            raise DPOFormatError("a preference sequence cannot be empty")
        if not 0 < self.prompt_prefix_length < len(self.input_ids):
            raise DPOFormatError("preference prompt prefix must leave response tokens")
        if any(self.response_mask[: self.prompt_prefix_length]):
            raise DPOFormatError("prompt tokens must be masked")
        actual_count = sum(self.response_mask)
        if actual_count != self.response_token_count or actual_count <= 0:
            raise DPOFormatError("a preference sequence must contain response targets")


@dataclass(frozen=True, slots=True)
class PreferencePairExample:
    """Two rendered sequences that share an identical prompt prefix."""

    chosen: PreferenceSequence
    rejected: PreferenceSequence
    template_hash: str

    def __post_init__(self) -> None:
        if self.chosen.template_hash != self.template_hash:
            raise DPOFormatError("chosen template hash does not match pair")
        if self.rejected.template_hash != self.template_hash:
            raise DPOFormatError("rejected template hash does not match pair")
        if self.chosen.prompt_prefix_length != self.rejected.prompt_prefix_length:
            raise DPOFormatError("chosen and rejected prompts have different lengths")
        prefix = self.chosen.prompt_prefix_length
        if self.chosen.input_ids[:prefix] != self.rejected.input_ids[:prefix]:
            raise DPOFormatError("chosen and rejected prompts are not identical")


@dataclass(frozen=True, slots=True)
class DPOObjectiveOutput:
    """DPO loss and diagnostics for one batch."""

    loss: Tensor
    logits: Tensor
    chosen_rewards: Tensor
    rejected_rewards: Tensor
    reward_margins: Tensor


@dataclass(frozen=True, slots=True)
class _TokenBlock:
    role: ChatRole
    token_ids: list[int]


def _message_blocks(
    messages: Sequence[ChatMessage],
    tokenizer: ByteBPE,
) -> list[_TokenBlock]:
    return [
        _TokenBlock(
            role=message.role,
            token_ids=[
                SPECIAL_TOKEN_IDS[ROLE_SPECIAL_TOKENS[message.role]],
                *tokenizer.encode(message.content),
                SPECIAL_TOKEN_IDS[END_SPECIAL_TOKEN],
            ],
        )
        for message in messages
    ]


def _drop_oldest_turn_pair(blocks: list[_TokenBlock]) -> bool:
    body_start = 1 if blocks and blocks[0].role == "system" else 0
    if len(blocks) - body_start < 4:
        return False
    if blocks[body_start].role != "user" or blocks[body_start + 1].role != "assistant":
        return False
    del blocks[body_start : body_start + 2]
    return True


def _drop_oldest_content_token(blocks: Sequence[_TokenBlock]) -> bool:
    for block in blocks:
        if len(block.token_ids) > 2:
            block.token_ids.pop(1)
            return True
    return False


def _truncate_response_content(
    response_tokens: list[int],
    *,
    max_content_tokens: int,
) -> tuple[list[int], bool]:
    if max_content_tokens < 0:
        raise DPOFormatError("max_length is too short for preference control markers")
    if len(response_tokens) <= max_content_tokens:
        return response_tokens, False
    if max_content_tokens == 0:
        return [], True
    return response_tokens[-max_content_tokens:], True


def _assemble_preference_sequence(
    prompt_blocks: Sequence[_TokenBlock],
    response_content: Sequence[int],
    *,
    template_hash: str,
    truncated: bool,
) -> PreferenceSequence:
    prompt_ids = [token for block in prompt_blocks for token in block.token_ids]
    assistant_role = SPECIAL_TOKEN_IDS[ROLE_SPECIAL_TOKENS["assistant"]]
    end_token = SPECIAL_TOKEN_IDS[END_SPECIAL_TOKEN]
    eos_token = SPECIAL_TOKEN_IDS[EOS_SPECIAL_TOKEN]
    input_ids = (
        SPECIAL_TOKEN_IDS[BOS_SPECIAL_TOKEN],
        *prompt_ids,
        assistant_role,
        *response_content,
        end_token,
        eos_token,
    )
    prompt_prefix_length = 1 + len(prompt_ids) + 1
    response_mask = (False,) * prompt_prefix_length + (True,) * (
        len(response_content) + 2
    )
    return PreferenceSequence(
        input_ids=input_ids,
        response_mask=response_mask,
        template_hash=template_hash,
        prompt_prefix_length=prompt_prefix_length,
        truncated=truncated,
        response_token_count=len(response_content) + 2,
    )


def render_preference_pair(
    pair: PreferencePair,
    tokenizer: ByteBPE,
    *,
    template: ChatTemplate = DEFAULT_CHAT_TEMPLATE,
    max_length: int = 1_024,
    truncation: Literal["error", "left"] = "error",
) -> PreferencePairExample:
    """Render chosen/rejected responses against one shared masked prompt."""

    if max_length <= 1:
        raise DPOFormatError("max_length must be greater than one")
    if truncation not in {"error", "left"}:
        raise DPOFormatError("unsupported preference truncation policy")

    prompt_blocks = _message_blocks(pair.prompt.messages, tokenizer)
    chosen_content = tokenizer.encode(pair.chosen.content)
    rejected_content = tokenizer.encode(pair.rejected.content)
    min_prompt_blocks = 2 if pair.prompt.messages[0].role != "system" else 4
    max_response_block_length = max_length - 2 - min_prompt_blocks
    chosen_content, chosen_truncated = _truncate_response_content(
        chosen_content,
        max_content_tokens=max_response_block_length - 2,
    )
    rejected_content, rejected_truncated = _truncate_response_content(
        rejected_content,
        max_content_tokens=max_response_block_length - 2,
    )
    chosen_block_length = len(chosen_content) + 2
    rejected_block_length = len(rejected_content) + 2
    prompt_budget = max_length - 2 - max(chosen_block_length, rejected_block_length)
    prompt_truncated = False

    if truncation == "error":
        full_prompt_length = sum(len(block.token_ids) for block in prompt_blocks)
        if full_prompt_length > prompt_budget or chosen_truncated or rejected_truncated:
            raise DPOFormatError("preference pair exceeds the DPO context length")
    else:
        while sum(len(block.token_ids) for block in prompt_blocks) > prompt_budget:
            changed = _drop_oldest_turn_pair(prompt_blocks)
            if not changed:
                changed = _drop_oldest_content_token(prompt_blocks)
            if not changed:
                raise DPOFormatError(
                    "preference context is too short for its control markers"
                )
            prompt_truncated = True

    chosen = _assemble_preference_sequence(
        prompt_blocks,
        chosen_content,
        template_hash=template.template_hash,
        truncated=chosen_truncated or prompt_truncated,
    )
    rejected = _assemble_preference_sequence(
        prompt_blocks,
        rejected_content,
        template_hash=template.template_hash,
        truncated=rejected_truncated or prompt_truncated,
    )
    return PreferencePairExample(
        chosen=chosen,
        rejected=rejected,
        template_hash=template.template_hash,
    )


def masked_sequence_logprob(
    logits: Tensor,
    input_ids: Tensor,
    response_mask: Tensor,
) -> Tensor:
    """Sum causal log-probabilities only over response target tokens."""

    if logits.ndim != 3:
        raise DPOFormatError("logits must have shape [batch, sequence, vocabulary]")
    if input_ids.ndim != 2 or logits.shape[:2] != input_ids.shape:
        raise DPOFormatError("input IDs must match logits batch and sequence")
    if response_mask.shape != input_ids.shape:
        raise DPOFormatError("response mask must match input IDs")
    if input_ids.dtype != torch.long:
        raise DPOFormatError("input IDs must use torch.long token IDs")
    if response_mask.dtype != torch.bool:
        raise DPOFormatError("response mask must use torch.bool")
    shifted_mask = response_mask[:, 1:]
    if not bool(shifted_mask.any(dim=1).all()):
        raise DPOFormatError("every sequence must contain response target tokens")
    log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
    token_log_probs = log_probs.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (token_log_probs * shifted_mask).sum(dim=1)


def dpo_objective(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor,
    reference_rejected_logps: Tensor,
    *,
    beta: float = 0.1,
) -> DPOObjectiveOutput:
    """Compute the standard DPO logistic objective and diagnostics."""

    if beta <= 0:
        raise DPOFormatError("DPO beta must be positive")
    tensors = (
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
    )
    if any(tensor.ndim != 1 for tensor in tensors):
        raise DPOFormatError("DPO log-probability tensors must be one-dimensional")
    if any(tensor.shape != tensors[0].shape for tensor in tensors[1:]):
        raise DPOFormatError("DPO log-probability tensors must have equal shapes")
    if tensors[0].numel() == 0:
        raise DPOFormatError("DPO requires a non-empty preference batch")
    policy_delta = policy_chosen_logps - policy_rejected_logps
    reference_delta = reference_chosen_logps - reference_rejected_logps
    reward_margins = policy_delta - reference_delta
    logits = beta * reward_margins
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)
    return DPOObjectiveOutput(
        loss=-F.logsigmoid(logits).mean(),
        logits=logits,
        chosen_rewards=chosen_rewards,
        rejected_rewards=rejected_rewards,
        reward_margins=reward_margins,
    )


def dpo_loss(
    policy_chosen_logps: Tensor,
    policy_rejected_logps: Tensor,
    reference_chosen_logps: Tensor,
    reference_rejected_logps: Tensor,
    *,
    beta: float = 0.1,
) -> Tensor:
    """Return only the scalar DPO loss for optimizer integration."""

    return dpo_objective(
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta=beta,
    ).loss
