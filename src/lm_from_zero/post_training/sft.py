"""Assistant-only supervised fine-tuning examples and loss."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Literal

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
    ChatRole,
    ChatTemplate,
    Conversation,
)
from lm_from_zero.tokenizer.bpe import SPECIAL_TOKEN_IDS, ByteBPE

IGNORE_INDEX = -100


class SFTFormatError(ValueError):
    """Raised when a conversation cannot produce a valid SFT example."""


class SFTConfig(BaseModel):
    """Versioned first-stage causal SFT policy from the project plan."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-sft-config"] = "lm-from-zero-sft-config"
    format_version: Literal[1] = 1
    max_length: Annotated[int, Field(gt=1)] = 1_024
    learning_rate: Annotated[float, Field(gt=0)] = 2e-5
    epochs: Literal[1] = 1
    example_count: Annotated[int, Field(gt=0)] = 100_000
    packing: Literal["length_bucketed", "padding_free"] = "length_bucketed"
    assistant_only_loss: Literal[True] = True

    @model_validator(mode="after")
    def validate_packing_policy(self) -> SFTConfig:
        if self.packing == "padding_free" and self.max_length < 2:
            raise ValueError("padding-free packing requires a usable context")
        return self


@dataclass(frozen=True, slots=True)
class _TokenBlock:
    role: ChatRole
    token_ids: list[int]
    labels: list[int]


@dataclass(frozen=True, slots=True)
class SupervisedChatExample:
    """Tokenized conversation with prompt labels masked by ``IGNORE_INDEX``."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    template_hash: str
    truncated: bool
    assistant_token_count: int

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.labels):
            raise SFTFormatError("input IDs and labels must have the same length")
        if not self.input_ids:
            raise SFTFormatError("an SFT example cannot be empty")
        actual_count = sum(label != IGNORE_INDEX for label in self.labels)
        if actual_count != self.assistant_token_count or actual_count <= 0:
            raise SFTFormatError("an SFT example must contain assistant targets")


@dataclass(frozen=True, slots=True)
class SFTBatch:
    """Right-padded tensors consumed by a causal model."""

    input_ids: Tensor
    labels: Tensor
    attention_mask: Tensor
    lengths: tuple[int, ...]
    assistant_token_counts: tuple[int, ...]


def _message_blocks(
    conversation: Conversation,
    tokenizer: ByteBPE,
) -> list[_TokenBlock]:
    blocks: list[_TokenBlock] = []
    for message in conversation.messages:
        content_ids = tokenizer.encode(message.content)
        token_ids = [
            SPECIAL_TOKEN_IDS[ROLE_SPECIAL_TOKENS[message.role]],
            *content_ids,
            SPECIAL_TOKEN_IDS[END_SPECIAL_TOKEN],
        ]
        labels = [IGNORE_INDEX] * len(token_ids)
        if message.role == "assistant":
            labels[1:] = token_ids[1:]
        blocks.append(
            _TokenBlock(role=message.role, token_ids=token_ids, labels=labels)
        )
    return blocks


def _assembled_length(blocks: Sequence[_TokenBlock]) -> int:
    return 2 + sum(len(block.token_ids) for block in blocks)


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
            block.labels.pop(1)
            return True
    return False


def _assemble(
    blocks: Sequence[_TokenBlock],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    input_ids = [SPECIAL_TOKEN_IDS[BOS_SPECIAL_TOKEN]]
    labels = [IGNORE_INDEX]
    for block in blocks:
        input_ids.extend(block.token_ids)
        labels.extend(block.labels)
    input_ids.append(SPECIAL_TOKEN_IDS[EOS_SPECIAL_TOKEN])
    labels.append(
        SPECIAL_TOKEN_IDS[EOS_SPECIAL_TOKEN]
        if blocks and blocks[-1].role == "assistant"
        else IGNORE_INDEX
    )
    return tuple(input_ids), tuple(labels)


def render_supervised_chat(
    conversation: Conversation,
    tokenizer: ByteBPE,
    *,
    template: ChatTemplate = DEFAULT_CHAT_TEMPLATE,
    max_length: int = 1_024,
    truncation: Literal["error", "left"] = "error",
) -> SupervisedChatExample:
    """Tokenize a final-assistant conversation and mask non-assistant labels.

    ``left`` truncation removes complete oldest user/assistant pairs first and
    then removes oldest content tokens while preserving all control markers.
    The final assistant turn is never removed.
    """

    if max_length <= 1:
        raise SFTFormatError("max_length must be greater than one")
    if truncation not in {"error", "left"}:
        raise SFTFormatError("unsupported SFT truncation policy")
    if not conversation.messages or conversation.messages[-1].role != "assistant":
        raise SFTFormatError("SFT conversations must end with an assistant turn")
    blocks = _message_blocks(conversation, tokenizer)
    if not any(block.role == "assistant" for block in blocks):
        raise SFTFormatError("SFT conversations require an assistant turn")

    was_truncated = False
    while _assembled_length(blocks) > max_length:
        if truncation == "error":
            raise SFTFormatError("conversation exceeds the SFT context length")
        changed = _drop_oldest_turn_pair(blocks)
        if not changed:
            changed = _drop_oldest_content_token(blocks)
        if not changed:
            raise SFTFormatError("SFT context is too short for its control markers")
        was_truncated = True

    input_ids, labels = _assemble(blocks)
    assistant_token_count = sum(label != IGNORE_INDEX for label in labels)
    return SupervisedChatExample(
        input_ids=input_ids,
        labels=labels,
        template_hash=template.template_hash,
        truncated=was_truncated,
        assistant_token_count=assistant_token_count,
    )


def collate_supervised_chat(
    examples: Sequence[SupervisedChatExample],
    *,
    pad_token_id: int = SPECIAL_TOKEN_IDS["<|pad|>"],
    pad_to_length: int | None = None,
) -> SFTBatch:
    """Right-pad examples and preserve assistant-target counts."""

    if not examples:
        raise SFTFormatError("cannot collate an empty SFT batch")
    if pad_token_id < 0:
        raise SFTFormatError("pad_token_id cannot be negative")
    lengths = tuple(len(example.input_ids) for example in examples)
    target_length = max(lengths) if pad_to_length is None else pad_to_length
    if target_length < max(lengths):
        raise SFTFormatError("pad_to_length is shorter than an example")
    input_ids = torch.full(
        (len(examples), target_length), pad_token_id, dtype=torch.long
    )
    labels = torch.full((len(examples), target_length), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros((len(examples), target_length), dtype=torch.bool)
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(example.labels, dtype=torch.long)
        attention_mask[row, :length] = True
    return SFTBatch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        lengths=lengths,
        assistant_token_counts=tuple(
            example.assistant_token_count for example in examples
        ),
    )


def assistant_only_causal_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute shifted CE over assistant labels and no prompt/padding labels."""

    if logits.ndim != 3:
        raise SFTFormatError("logits must have shape [batch, sequence, vocabulary]")
    if labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise SFTFormatError("labels must match the logits batch and sequence")
    if labels.dtype != torch.long:
        raise SFTFormatError("labels must use torch.long token IDs")
    shift_logits = logits[:, :-1].contiguous().float()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels != IGNORE_INDEX
    if not bool(valid.any()):
        raise SFTFormatError("SFT labels contain no assistant targets")
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )
