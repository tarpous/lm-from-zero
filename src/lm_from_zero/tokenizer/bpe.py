"""Deterministic, pure-Python byte-pair encoding.

The implementation intentionally favors explicitness and reproducibility over
training speed. Every possible byte has an initial token, so ordinary input
never needs an unknown token.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from itertools import pairwise
from pathlib import Path
from typing import Final, Self

from lm_from_zero.tokenizer.pretokenizer import (
    PRETOKENIZER_MODES,
    PretokenizerMode,
    pretokenize,
)

SPECIAL_TOKENS: Final[tuple[str, ...]] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|mask|>",
)
SPECIAL_TOKEN_IDS: Final[dict[str, int]] = {
    token: token_id for token_id, token in enumerate(SPECIAL_TOKENS)
}
BYTE_TOKEN_OFFSET: Final[int] = len(SPECIAL_TOKENS)
INITIAL_VOCAB_SIZE: Final[int] = BYTE_TOKEN_OFFSET + 256
MAX_UINT16_VOCAB_SIZE: Final[int] = 1 << 16
FORMAT_VERSION: Final[int] = 1
BYTE_ENCODING: Final[str] = "gpt2-byte-level-unicode-order"
ENCODING_CACHE_CAPACITY: Final[int] = 65_536

TokenPair = tuple[int, int]


def _byte_level_symbols() -> tuple[str, ...]:
    """Return the reversible GPT-2/ByteLevel symbol for each raw byte."""

    visible_bytes = list(range(ord("!"), ord("~") + 1))
    visible_bytes.extend(range(ord("¡"), ord("¬") + 1))
    visible_bytes.extend(range(ord("®"), ord("ÿ") + 1))
    byte_values = list(visible_bytes)
    code_points = list(visible_bytes)
    extra_code_point = 256
    for byte_value in range(256):
        if byte_value not in visible_bytes:
            byte_values.append(byte_value)
            code_points.append(extra_code_point)
            extra_code_point += 1

    symbols = [""] * 256
    for byte_value, code_point in zip(byte_values, code_points, strict=True):
        symbols[byte_value] = chr(code_point)
    return tuple(symbols)


BYTE_LEVEL_SYMBOLS: Final[tuple[str, ...]] = _byte_level_symbols()
BYTE_VALUES_BY_TOKEN_ID: Final[tuple[int, ...]] = tuple(
    sorted(range(256), key=BYTE_LEVEL_SYMBOLS.__getitem__)
)
BYTE_TO_TOKEN_ID: Final[dict[int, int]] = {
    byte_value: BYTE_TOKEN_OFFSET + index
    for index, byte_value in enumerate(BYTE_VALUES_BY_TOKEN_ID)
}


def _merge_pair(sequence: Sequence[int], pair: TokenPair, new_id: int) -> list[int]:
    """Replace non-overlapping occurrences of ``pair`` from left to right."""

    merged: list[int] = []
    index = 0
    while index < len(sequence):
        if (
            index + 1 < len(sequence)
            and sequence[index] == pair[0]
            and sequence[index + 1] == pair[1]
        ):
            merged.append(new_id)
            index += 2
        else:
            merged.append(sequence[index])
            index += 1
    return merged


@dataclass(frozen=True, slots=True)
class ByteBPE:
    """A trained byte-level BPE tokenizer.

    Merge token IDs are implicit: the first merge receives ID 264, the second
    265, and so on. Special strings are parsed only when explicitly allowed by
    the caller; otherwise they remain ordinary UTF-8 bytes.
    """

    merges: tuple[TokenPair, ...] = ()
    pretokenizer: PretokenizerMode = "none"
    _vocabulary: tuple[bytes | None, ...] = field(init=False, repr=False)
    _merge_ranks: dict[TokenPair, int] = field(init=False, repr=False, compare=False)
    _encoding_cache: dict[bytes, tuple[int, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.pretokenizer not in PRETOKENIZER_MODES:
            raise ValueError(f"unsupported pretokenizer: {self.pretokenizer}")
        vocabulary: list[bytes | None] = [None] * len(SPECIAL_TOKENS)
        vocabulary.extend(bytes((value,)) for value in BYTE_VALUES_BY_TOKEN_ID)

        for merge_index, pair in enumerate(self.merges):
            expected_max_id = INITIAL_VOCAB_SIZE + merge_index
            left, right = pair
            if left < BYTE_TOKEN_OFFSET or right < BYTE_TOKEN_OFFSET:
                raise ValueError("BPE merges cannot contain special-token IDs")
            if left >= expected_max_id or right >= expected_max_id:
                raise ValueError(
                    "each merge may reference only byte tokens or earlier merges"
                )
            left_bytes = vocabulary[left]
            right_bytes = vocabulary[right]
            if left_bytes is None or right_bytes is None:
                raise ValueError("BPE merges cannot contain special tokens")
            vocabulary.append(left_bytes + right_bytes)

        if len(vocabulary) > MAX_UINT16_VOCAB_SIZE:
            raise ValueError("tokenizer vocabulary does not fit in uint16")
        object.__setattr__(self, "_vocabulary", tuple(vocabulary))
        object.__setattr__(
            self,
            "_merge_ranks",
            {pair: rank for rank, pair in enumerate(self.merges)},
        )
        object.__setattr__(self, "_encoding_cache", {})

    @classmethod
    def train(
        cls,
        documents: Iterable[bytes | str],
        *,
        target_vocab_size: int,
        min_frequency: int = 2,
        pretokenizer: PretokenizerMode = "none",
    ) -> Self:
        """Train with frequency-first, lexicographic pair tie-breaking.

        Documents remain separate while counting pairs, so merges never cross a
        document boundary. Training can stop below the requested vocabulary size
        when no pair reaches ``min_frequency``.
        """

        from lm_from_zero.tokenizer.trainer import train_bpe

        result = train_bpe(
            documents,
            target_vocab_size=target_vocab_size,
            min_frequency=min_frequency,
            pretokenizer=pretokenizer,
        )
        return cls(
            merges=result.tokenizer.merges,
            pretokenizer=result.tokenizer.pretokenizer,
        )

    @property
    def vocab_size(self) -> int:
        """Return the realized vocabulary size, including special tokens."""

        return len(self._vocabulary)

    def encode_bytes(self, data: bytes) -> list[int]:
        """Encode arbitrary bytes without special-token parsing."""

        token_ids = [BYTE_TO_TOKEN_ID[value] for value in data]
        while len(token_ids) > 1:
            ranked_pairs = (
                (self._merge_ranks[pair], pair)
                for pair in pairwise(token_ids)
                if pair in self._merge_ranks
            )
            selected = min(ranked_pairs, default=None)
            if selected is None:
                break
            merge_rank, pair = selected
            token_ids = _merge_pair(
                token_ids,
                pair,
                INITIAL_VOCAB_SIZE + merge_rank,
            )
        return token_ids

    def _encode_text(self, text: str) -> list[int]:
        encoded: list[int] = []
        for chunk in pretokenize(text, self.pretokenizer):
            cached = self._encoding_cache.get(chunk)
            if cached is None:
                cached = tuple(self.encode_bytes(chunk))
                if len(self._encoding_cache) < ENCODING_CACHE_CAPACITY:
                    self._encoding_cache.setdefault(chunk, cached)
            encoded.extend(cached)
        return encoded

    def encode(
        self,
        text: str,
        *,
        allowed_special: Collection[str] = (),
    ) -> list[int]:
        """Encode UTF-8 text, parsing only explicitly allowed special tokens."""

        allowed = frozenset(allowed_special)
        unknown = allowed.difference(SPECIAL_TOKEN_IDS)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown special token(s): {names}")
        if not allowed:
            return self._encode_text(text)

        ordered_special = sorted(allowed, key=lambda token: (-len(token), token))
        encoded: list[int] = []
        cursor = 0
        while cursor < len(text):
            matches = [(text.find(token, cursor), token) for token in ordered_special]
            matches = [match for match in matches if match[0] >= 0]
            if not matches:
                encoded.extend(self._encode_text(text[cursor:]))
                break
            position, token = min(matches, key=lambda match: (match[0], match[1]))
            encoded.extend(self._encode_text(text[cursor:position]))
            encoded.append(SPECIAL_TOKEN_IDS[token])
            cursor = position + len(token)
        return encoded

    def token_bytes(self, token_id: int) -> bytes:
        """Return ordinary token bytes, rejecting special and invalid IDs."""

        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"token ID {token_id} is outside the vocabulary")
        value = self._vocabulary[token_id]
        if value is None:
            raise ValueError(f"token ID {token_id} is a special token")
        return value

    def decode_bytes(
        self,
        token_ids: Iterable[int],
        *,
        render_special: bool = False,
    ) -> bytes:
        """Decode IDs to bytes, optionally rendering control-token spellings."""

        decoded = bytearray()
        for token_id in token_ids:
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(f"token ID {token_id} is outside the vocabulary")
            value = self._vocabulary[token_id]
            if value is None:
                if not render_special:
                    raise ValueError(
                        "special token encountered; set render_special=True "
                        "to render it"
                    )
                decoded.extend(SPECIAL_TOKENS[token_id].encode("utf-8"))
            else:
                decoded.extend(value)
        return bytes(decoded)

    def decode(
        self,
        token_ids: Iterable[int],
        *,
        render_special: bool = False,
        errors: str = "strict",
    ) -> str:
        """Decode IDs as UTF-8 text."""

        return self.decode_bytes(token_ids, render_special=render_special).decode(
            "utf-8", errors=errors
        )

    def to_dict(self) -> dict[str, object]:
        """Return the stable, versioned serialization payload."""

        return {
            "format": "lm-from-zero-byte-bpe",
            "format_version": FORMAT_VERSION,
            "byte_encoding": BYTE_ENCODING,
            "pretokenizer": self.pretokenizer,
            "special_tokens": {
                token: token_id for token_id, token in enumerate(SPECIAL_TOKENS)
            },
            "byte_token_offset": BYTE_TOKEN_OFFSET,
            "merges": [list(pair) for pair in self.merges],
        }

    def canonical_json(self) -> str:
        """Return canonical JSON suitable for hashing and manifests."""

        return json.dumps(
            self.to_dict(), ensure_ascii=True, sort_keys=True, separators=(",", ":")
        )

    @property
    def model_hash(self) -> str:
        """Return the SHA-256 hash of the canonical tokenizer model."""

        return sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        """Atomically write the tokenizer model as canonical JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(self.canonical_json() + "\n", encoding="utf-8")
        temporary.replace(destination)

    @classmethod
    def from_dict(cls, payload: object) -> Self:
        """Validate and load a serialized tokenizer payload."""

        if not isinstance(payload, dict):
            raise ValueError("tokenizer payload must be an object")
        expected_special = {
            token: token_id for token_id, token in enumerate(SPECIAL_TOKENS)
        }
        if payload.get("format") != "lm-from-zero-byte-bpe":
            raise ValueError("unrecognized tokenizer format")
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported tokenizer format version")
        if payload.get("byte_encoding") != BYTE_ENCODING:
            raise ValueError("byte-token encoding does not match the model format")
        raw_pretokenizer = payload.get("pretokenizer")
        if raw_pretokenizer == "none":
            pretokenizer: PretokenizerMode = "none"
        elif raw_pretokenizer == "gpt2":
            pretokenizer = "gpt2"
        else:
            raise ValueError("unsupported tokenizer pretokenizer")
        if payload.get("special_tokens") != expected_special:
            raise ValueError("special-token mapping does not match fixed IDs")
        if payload.get("byte_token_offset") != BYTE_TOKEN_OFFSET:
            raise ValueError("byte-token offset does not match fixed IDs")

        raw_merges = payload.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError("merges must be a list")
        merges: list[TokenPair] = []
        for raw_pair in raw_merges:
            if (
                not isinstance(raw_pair, list)
                or len(raw_pair) != 2
                or any(type(token_id) is not int for token_id in raw_pair)
            ):
                raise ValueError("every merge must contain exactly two integer IDs")
            merges.append((raw_pair[0], raw_pair[1]))
        return cls(merges=tuple(merges), pretokenizer=pretokenizer)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load and validate a tokenizer model from JSON."""

        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load tokenizer model: {error}") from error
        return cls.from_dict(payload)
