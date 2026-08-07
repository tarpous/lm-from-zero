"""Versioned role-delimited conversations for project-owned post-training."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from lm_from_zero.tokenizer.bpe import SPECIAL_TOKENS

ChatRole = Literal["system", "user", "assistant"]
CHAT_TEMPLATE_FORMAT: Final = "lm-from-zero-chat-template"
CHAT_TEMPLATE_VERSION: Final = 1
ROLE_SPECIAL_TOKENS: Final[dict[ChatRole, str]] = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
END_SPECIAL_TOKEN: Final = "<|end|>"
BOS_SPECIAL_TOKEN: Final = "<|bos|>"
EOS_SPECIAL_TOKEN: Final = "<|eos|>"


class ChatMessage(BaseModel):
    """One validated role/content pair."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: ChatRole
    content: str

    @field_validator("content")
    @classmethod
    def reject_control_tokens(cls, content: str) -> str:
        if any(token in content for token in SPECIAL_TOKENS):
            raise ValueError("message content cannot contain reserved control tokens")
        return content


class Conversation(BaseModel):
    """A system-optional conversation with alternating user/assistant turns."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    messages: tuple[ChatMessage, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_turn_order(self) -> Self:
        messages = self.messages
        first_body_index = 0
        if messages[0].role == "system":
            first_body_index = 1
        elif messages[0].role != "user":
            raise ValueError("a conversation must begin with system or user")

        expected: ChatRole = "user"
        for message in messages[first_body_index:]:
            if message.role != expected:
                raise ValueError(
                    f"conversation expected a {expected} turn, got {message.role}"
                )
            expected = "assistant" if expected == "user" else "user"
        return self


class ChatTemplate(BaseModel):
    """Stable serialization of the role-delimited chat format."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-chat-template"] = CHAT_TEMPLATE_FORMAT
    format_version: Literal[1] = CHAT_TEMPLATE_VERSION
    name: Literal["role-delimited"] = "role-delimited"
    bos_token: Literal["<|bos|>"] = BOS_SPECIAL_TOKEN
    eos_token: Literal["<|eos|>"] = EOS_SPECIAL_TOKEN
    end_token: Literal["<|end|>"] = END_SPECIAL_TOKEN
    role_tokens: dict[ChatRole, str] = Field(
        default_factory=lambda: dict(ROLE_SPECIAL_TOKENS)
    )

    @model_validator(mode="after")
    def validate_fixed_tokens(self) -> Self:
        if tuple(self.role_tokens) != ("system", "user", "assistant"):
            raise ValueError("role token mapping must use the fixed role order")
        if self.role_tokens != ROLE_SPECIAL_TOKENS:
            raise ValueError("role token mapping does not match fixed special tokens")
        return self

    def render(self, conversation: Conversation) -> str:
        """Render one conversation with no ambiguous whitespace separators."""

        pieces: list[str] = [self.bos_token]
        for message in conversation.messages:
            pieces.extend(
                (self.role_tokens[message.role], message.content, self.end_token)
            )
        pieces.append(self.eos_token)
        return "".join(pieces)

    def canonical_json(self) -> str:
        """Return the stable template manifest used by downstream hashes."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def template_hash(self) -> str:
        """Return the SHA-256 hash of the canonical template manifest."""

        return sha256(self.canonical_json().encode()).hexdigest()


DEFAULT_CHAT_TEMPLATE = ChatTemplate()
