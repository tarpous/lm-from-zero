"""Deterministic document splitting and checked uint16 token shards."""

from __future__ import annotations

import json
import os
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lm_from_zero.progress import ProgressReporter
from lm_from_zero.tokenizer.bpe import SPECIAL_TOKEN_IDS, ByteBPE

Split = Literal["train", "validation", "test"]
SHA256_PATTERN = r"^[0-9a-f]{64}$"
UINT16_LIMIT = 1 << 16


class DataValidationError(ValueError):
    """Raised when data provenance or a shard artifact is invalid."""


class SplitPolicy(BaseModel):
    """Integer-bucket split policy with deterministic hash assignment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    seed: int = 1337
    bucket_count: Annotated[int, Field(gt=0)] = 10_000
    validation_buckets: Annotated[int, Field(ge=0)] = 100
    test_buckets: Annotated[int, Field(ge=0)] = 100

    @model_validator(mode="after")
    def validate_bucket_allocation(self) -> Self:
        if self.validation_buckets + self.test_buckets >= self.bucket_count:
            raise ValueError("validation and test buckets must leave a train split")
        return self


@dataclass(frozen=True, slots=True)
class HashedDocument:
    """An immutable document with stable provenance and split assignment."""

    source_index: int
    data: bytes
    content_hash: str
    split: Split


class ShardCursor(BaseModel):
    """The next split-local document to process after a completed shard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    next_document_index: Annotated[int, Field(ge=0)]


class ShardManifest(BaseModel):
    """Complete integrity and provenance metadata for one token shard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-token-shard"] = "lm-from-zero-token-shard"
    format_version: Literal[1] = 1
    status: Literal["complete"] = "complete"
    data_file: str
    split: Split
    dtype: Literal["uint16-le"] = "uint16-le"
    token_count: Annotated[int, Field(gt=0)]
    byte_count: Annotated[int, Field(gt=0)]
    sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    tokenizer_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    source_name: Annotated[str, Field(min_length=1)]
    source_revision: Annotated[str, Field(min_length=1)]
    source_document_hashes: tuple[Annotated[str, Field(pattern=SHA256_PATTERN)], ...]
    split_seed: int
    eos_token_id: Literal[2] = 2
    cursor: ShardCursor

    @model_validator(mode="after")
    def validate_internal_consistency(self) -> Self:
        if self.byte_count != self.token_count * 2:
            raise ValueError("byte_count must equal token_count * 2")
        if Path(self.data_file).name != self.data_file:
            raise ValueError("data_file must be a filename without directories")
        if not self.source_document_hashes:
            raise ValueError("source_document_hashes cannot be empty")
        if len(set(self.source_document_hashes)) != len(self.source_document_hashes):
            raise ValueError("source_document_hashes contains duplicates")
        return self


class RankCursor(BaseModel):
    """Serializable stride cursor for deterministic rank-aware iteration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rank: Annotated[int, Field(ge=0)]
    world_size: Annotated[int, Field(gt=0)]
    next_local_index: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def validate_rank(self) -> Self:
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        return self

    def take(self, count: int) -> tuple[tuple[int, ...], RankCursor]:
        """Return the next global indices and an advanced immutable cursor."""

        if count < 0:
            raise ValueError("count cannot be negative")
        indices = tuple(
            self.rank + local_index * self.world_size
            for local_index in range(
                self.next_local_index, self.next_local_index + count
            )
        )
        return indices, self.model_copy(
            update={"next_local_index": self.next_local_index + count}
        )


def document_hash(document: bytes | str) -> str:
    """Hash exact document bytes; strings use canonical UTF-8 encoding."""

    data = document if isinstance(document, bytes) else document.encode("utf-8")
    return sha256(data).hexdigest()


def assign_split(content_hash: str, policy: SplitPolicy) -> Split:
    """Assign a content hash to a split using the policy seed."""

    if len(content_hash) != 64:
        raise DataValidationError("content hash must contain 64 hexadecimal digits")
    try:
        bytes.fromhex(content_hash)
    except ValueError as error:
        raise DataValidationError("content hash is not valid hexadecimal") from error
    if content_hash != content_hash.lower():
        raise DataValidationError("content hash must use lowercase hexadecimal")

    seeded = sha256(f"{policy.seed}:{content_hash}".encode()).digest()
    bucket = int.from_bytes(seeded[:8], "big") % policy.bucket_count
    if bucket < policy.validation_buckets:
        return "validation"
    if bucket < policy.validation_buckets + policy.test_buckets:
        return "test"
    return "train"


def split_documents(
    documents: Iterable[bytes | str], policy: SplitPolicy
) -> dict[Split, list[HashedDocument]]:
    """Hash, deduplicate, and split documents without order-dependent choices."""

    result: dict[Split, list[HashedDocument]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen: set[str] = set()
    for source_index, document in enumerate(documents):
        data = document if isinstance(document, bytes) else document.encode("utf-8")
        content_hash = document_hash(data)
        if content_hash in seen:
            raise DataValidationError(
                f"duplicate document content at source index {source_index}"
            )
        seen.add(content_hash)
        split = assign_split(content_hash, policy)
        result[split].append(
            HashedDocument(
                source_index=source_index,
                data=data,
                content_hash=content_hash,
                split=split,
            )
        )
    return result


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        raise DataValidationError(f"incomplete temporary file exists: {temporary.name}")
    encoded = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _write_shard(
    output_directory: Path,
    *,
    split: Split,
    shard_index: int,
    tokens: Sequence[int],
    document_hashes: Sequence[str],
    tokenizer_hash: str,
    source_name: str,
    source_revision: str,
    split_seed: int,
    next_document_index: int,
) -> Path:
    stem = f"{split}-{shard_index:05d}"
    data_path = output_directory / f"{stem}.bin"
    manifest_path = output_directory / f"{stem}.json"
    partial_path = output_directory / f".{stem}.bin.partial"
    for path in (data_path, manifest_path, partial_path):
        if path.exists():
            raise DataValidationError(
                f"refusing to overwrite existing file: {path.name}"
            )

    token_array = np.asarray(tokens, dtype="<u2")
    with partial_path.open("xb") as handle:
        token_array.tofile(handle)
        handle.flush()
        os.fsync(handle.fileno())
    partial_path.replace(data_path)

    manifest = ShardManifest(
        data_file=data_path.name,
        split=split,
        token_count=len(tokens),
        byte_count=data_path.stat().st_size,
        sha256=_file_sha256(data_path),
        tokenizer_hash=tokenizer_hash,
        source_name=source_name,
        source_revision=source_revision,
        source_document_hashes=tuple(document_hashes),
        split_seed=split_seed,
        cursor=ShardCursor(next_document_index=next_document_index),
    )
    _atomic_write_json(manifest_path, manifest.model_dump(mode="json"))
    return manifest_path


def write_token_shards(
    documents: Sequence[HashedDocument],
    tokenizer: ByteBPE,
    output_directory: str | Path,
    *,
    split: Split,
    max_tokens_per_shard: int,
    source_name: str,
    source_revision: str,
    split_seed: int,
) -> tuple[Path, ...]:
    """Pack whole documents with EOS boundaries into atomic uint16 shards."""

    if max_tokens_per_shard <= 0:
        raise ValueError("max_tokens_per_shard must be positive")
    if tokenizer.vocab_size > UINT16_LIMIT:
        raise DataValidationError("tokenizer vocabulary does not fit in uint16")
    if any(document.split != split for document in documents):
        raise DataValidationError("document split does not match requested shard split")
    if len({document.content_hash for document in documents}) != len(documents):
        raise DataValidationError("documents contain duplicate content hashes")

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    manifests: list[Path] = []
    tokens = array("H")
    hashes: list[str] = []
    eos_token_id = SPECIAL_TOKEN_IDS["<|eos|>"]
    progress = ProgressReporter(f"shard building ({split})")
    progress.phase(
        "tokenizing documents",
        total=len(documents) if documents else None,
    )

    for document_index, document in enumerate(documents):
        document_tokens = [*tokenizer.encode_bytes(document.data), eos_token_id]
        if len(document_tokens) > max_tokens_per_shard:
            raise DataValidationError(
                f"document {document_index} exceeds max_tokens_per_shard"
            )
        if tokens and len(tokens) + len(document_tokens) > max_tokens_per_shard:
            manifests.append(
                _write_shard(
                    destination,
                    split=split,
                    shard_index=len(manifests),
                    tokens=tokens,
                    document_hashes=hashes,
                    tokenizer_hash=tokenizer.model_hash,
                    source_name=source_name,
                    source_revision=source_revision,
                    split_seed=split_seed,
                    next_document_index=document_index,
                )
            )
            tokens = array("H")
            hashes = []
        tokens.extend(document_tokens)
        hashes.append(document.content_hash)
        progress.update(document_index + 1, fields={"shards": len(manifests)})

    if tokens:
        manifests.append(
            _write_shard(
                destination,
                split=split,
                shard_index=len(manifests),
                tokens=tokens,
                document_hashes=hashes,
                tokenizer_hash=tokenizer.model_hash,
                source_name=source_name,
                source_revision=source_revision,
                split_seed=split_seed,
                next_document_index=len(documents),
            )
        )
    progress.finish("complete")
    return tuple(manifests)


def load_manifest(path: str | Path) -> ShardManifest:
    """Load a complete manifest and normalize parse errors."""

    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return ShardManifest.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise DataValidationError(f"invalid shard manifest: {error}") from error


def validate_shard(
    manifest_path: str | Path,
    *,
    expected_tokenizer_hash: str | None = None,
    expected_vocab_size: int | None = None,
) -> npt.NDArray[np.uint16]:
    """Validate integrity and return a read-only memory map of shard tokens."""

    path = Path(manifest_path)
    manifest = load_manifest(path)
    if (
        expected_tokenizer_hash is not None
        and manifest.tokenizer_hash != expected_tokenizer_hash
    ):
        raise DataValidationError("tokenizer hash does not match shard manifest")
    data_path = path.parent / manifest.data_file
    if not data_path.is_file():
        raise DataValidationError("shard data file is missing")
    if data_path.stat().st_size != manifest.byte_count:
        raise DataValidationError("shard byte length does not match manifest")
    if _file_sha256(data_path) != manifest.sha256:
        raise DataValidationError("shard checksum does not match manifest")

    tokens = np.memmap(
        data_path,
        dtype="<u2",
        mode="r",
        shape=(manifest.token_count,),
    )
    if expected_vocab_size is not None:
        if not 0 < expected_vocab_size <= UINT16_LIMIT:
            raise ValueError("expected_vocab_size must be in [1, 65536]")
        if np.any(tokens >= expected_vocab_size):
            raise DataValidationError("shard contains a token outside the vocabulary")
    return tokens


def validate_shard_directory(
    directory: str | Path,
    *,
    expected_tokenizer_hash: str | None = None,
    expected_vocab_size: int | None = None,
) -> tuple[npt.NDArray[np.uint16], ...]:
    """Reject incomplete/orphaned artifacts and validate every manifest."""

    path = Path(directory)
    partials = sorted(path.glob("*.partial")) + sorted(path.glob(".*.partial"))
    if partials:
        raise DataValidationError(f"incomplete shard artifact: {partials[0].name}")
    manifest_paths = sorted(path.glob("*.json"))
    manifests = [load_manifest(item) for item in manifest_paths]
    referenced_files = {manifest.data_file for manifest in manifests}
    binary_files = {item.name for item in path.glob("*.bin")}
    if binary_files != referenced_files:
        raise DataValidationError("orphaned or missing shard data file detected")
    tokenizer_hashes = {manifest.tokenizer_hash for manifest in manifests}
    if len(tokenizer_hashes) > 1:
        raise DataValidationError("shard directory contains mixed tokenizer hashes")
    seen_documents: set[str] = set()
    for manifest in manifests:
        duplicates = seen_documents.intersection(manifest.source_document_hashes)
        if duplicates:
            raise DataValidationError("document appears in more than one shard")
        seen_documents.update(manifest.source_document_hashes)
    return tuple(
        validate_shard(
            manifest_path,
            expected_tokenizer_hash=expected_tokenizer_hash,
            expected_vocab_size=expected_vocab_size,
        )
        for manifest_path in manifest_paths
    )
