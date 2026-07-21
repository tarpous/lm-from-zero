"""Project-owned GPT-2/ByteLevel pre-tokenization."""

from __future__ import annotations

from typing import Literal
from unicodedata import category

PretokenizerMode = Literal["none", "gpt2"]
PRETOKENIZER_MODES = frozenset(("none", "gpt2"))
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")


def _character_kind(character: str) -> str:
    unicode_category = category(character)
    if unicode_category.startswith("L"):
        return "letter"
    if unicode_category.startswith("N"):
        return "number"
    if character.isspace():
        return "space"
    return "other"


def _gpt2_chunks(text: str) -> tuple[str, ...]:
    chunks: list[str] = []
    index = 0
    while index < len(text):
        contraction = next(
            (item for item in _CONTRACTIONS if text.startswith(item, index)),
            None,
        )
        if contraction is not None:
            chunks.append(contraction)
            index += len(contraction)
            continue

        start = index
        if text[index] == " " and index + 1 < len(text):
            next_kind = _character_kind(text[index + 1])
            if next_kind != "space":
                index += 1
                kind = next_kind
            else:
                kind = "space"
        else:
            kind = _character_kind(text[index])

        if kind == "space":
            end = index + 1
            while end < len(text) and _character_kind(text[end]) == "space":
                end += 1
            if end < len(text) and end - start > 1:
                end -= 1
            chunks.append(text[start:end])
            index = end
            continue

        index += 1
        while index < len(text) and _character_kind(text[index]) == kind:
            index += 1
        chunks.append(text[start:index])
    return tuple(chunks)


def pretokenize(text: str, mode: PretokenizerMode) -> tuple[bytes, ...]:
    """Split text into UTF-8 byte chunks using the selected deterministic mode."""

    if not text:
        return ()
    if mode == "none":
        return (text.encode("utf-8"),)
    if mode == "gpt2":
        return tuple(chunk.encode("utf-8") for chunk in _gpt2_chunks(text))
    raise ValueError(f"unsupported pretokenizer mode: {mode}")
