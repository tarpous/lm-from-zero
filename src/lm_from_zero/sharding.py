"""Atomic construction and validation of complete token-shard sets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lm_from_zero.data import (
    DataValidationError,
    Split,
    SplitPolicy,
    load_manifest,
    split_documents,
    validate_shard_directory,
    write_token_shards,
)
from lm_from_zero.sampling import load_sample
from lm_from_zero.tokenizer.bpe import INITIAL_VOCAB_SIZE, ByteBPE
from lm_from_zero.tokenizer.pipeline import load_training_manifest

SHA256_PATTERN = r"^[0-9a-f]{64}$"
SPLIT_ORDER: tuple[Split, ...] = ("train", "validation", "test")
DEFAULT_SPLIT_POLICY = SplitPolicy()


class ShardSplitStats(BaseModel):
    """Measured contents of one logical split."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: Split
    document_count: Annotated[int, Field(ge=0)]
    token_count: Annotated[int, Field(ge=0)]
    manifest_files: tuple[str, ...]

    @model_validator(mode="after")
    def validate_files(self) -> Self:
        if any(Path(name).name != name for name in self.manifest_files):
            raise ValueError("shard manifest entries must be filenames")
        if len(set(self.manifest_files)) != len(self.manifest_files):
            raise ValueError("shard manifest entries must be unique")
        if (self.document_count == 0) != (len(self.manifest_files) == 0):
            raise ValueError("empty splits must not contain shard manifests")
        if (self.token_count == 0) != (len(self.manifest_files) == 0):
            raise ValueError("empty splits must not contain tokens")
        return self


class ShardBuildManifest(BaseModel):
    """Dataset, tokenizer, split, and shard-set integrity contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-shard-build"] = "lm-from-zero-shard-build"
    format_version: Literal[1] = 1
    status: Literal["complete"] = "complete"
    shard_directory: Literal["shards"] = "shards"
    source_dataset_id: Annotated[str, Field(min_length=1)]
    source_revision: Annotated[str, Field(min_length=1)]
    source_license: Annotated[str, Field(min_length=1)]
    source_sample_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    source_content_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    source_document_count: Annotated[int, Field(gt=0)]
    source_text_bytes: Annotated[int, Field(gt=0)]
    tokenizer_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    tokenizer_vocab_size: Annotated[int, Field(ge=INITIAL_VOCAB_SIZE, le=65536)]
    split_policy: SplitPolicy
    max_tokens_per_shard: Annotated[int, Field(gt=0, le=100_000_000)]
    splits: tuple[ShardSplitStats, ShardSplitStats, ShardSplitStats]
    total_token_count: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_totals(self) -> Self:
        if tuple(item.split for item in self.splits) != SPLIT_ORDER:
            raise ValueError("split statistics must use train/validation/test order")
        if (
            sum(item.document_count for item in self.splits)
            != self.source_document_count
        ):
            raise ValueError("split document counts do not match the source sample")
        if sum(item.token_count for item in self.splits) != self.total_token_count:
            raise ValueError("split token counts do not match the build total")
        return self


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    if path.exists() or partial.exists():
        raise DataValidationError(f"refusing to overwrite build file: {path.name}")
    encoded = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    with partial.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def load_shard_build(path: str | Path) -> ShardBuildManifest:
    """Load and validate a complete shard-build manifest."""

    try:
        return ShardBuildManifest.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise DataValidationError(f"invalid shard build manifest: {error}") from error


def validate_shard_build(path: str | Path) -> ShardBuildManifest:
    """Validate all files and provenance relationships in a shard build."""

    build_path = Path(path)
    build = load_shard_build(build_path)
    shard_directory = build_path.parent / build.shard_directory
    if not shard_directory.is_dir():
        raise DataValidationError("shard directory is missing")

    expected_files = {
        filename for stats in build.splits for filename in stats.manifest_files
    }
    actual_files = {item.name for item in shard_directory.glob("*.json")}
    if actual_files != expected_files:
        raise DataValidationError(
            "shard manifest file set does not match build manifest"
        )

    mapped_shards = validate_shard_directory(
        shard_directory,
        expected_tokenizer_hash=build.tokenizer_hash,
        expected_vocab_size=build.tokenizer_vocab_size,
    )
    del mapped_shards

    total_tokens = 0
    total_documents = 0
    for stats in build.splits:
        next_document_index = 0
        split_tokens = 0
        split_documents_count = 0
        for filename in stats.manifest_files:
            manifest = load_manifest(shard_directory / filename)
            if manifest.split != stats.split:
                raise DataValidationError("shard split does not match build manifest")
            if manifest.source_name != build.source_dataset_id:
                raise DataValidationError("shard source does not match build manifest")
            if manifest.source_revision != build.source_revision:
                raise DataValidationError(
                    "shard revision does not match build manifest"
                )
            if manifest.split_seed != build.split_policy.seed:
                raise DataValidationError(
                    "shard split seed does not match build manifest"
                )
            if manifest.token_count > build.max_tokens_per_shard:
                raise DataValidationError("shard exceeds the configured token limit")
            split_tokens += manifest.token_count
            split_documents_count += len(manifest.source_document_hashes)
            next_document_index += len(manifest.source_document_hashes)
            if manifest.cursor.next_document_index != next_document_index:
                raise DataValidationError(
                    "shard cursor does not match document progress"
                )
        if split_tokens != stats.token_count:
            raise DataValidationError("split token count does not match build manifest")
        if split_documents_count != stats.document_count:
            raise DataValidationError(
                "split document count does not match build manifest"
            )
        total_tokens += split_tokens
        total_documents += split_documents_count

    if total_tokens != build.total_token_count:
        raise DataValidationError("total token count does not match build manifest")
    if total_documents != build.source_document_count:
        raise DataValidationError("source document count does not match build manifest")
    return build


def build_token_shards(
    sample_manifest_path: str | Path,
    tokenizer_training_manifest_path: str | Path,
    output_directory: str | Path,
    *,
    split_policy: SplitPolicy = DEFAULT_SPLIT_POLICY,
    max_tokens_per_shard: int = 100_000_000,
) -> ShardBuildManifest:
    """Build and atomically publish a checked sample-to-shards artifact."""

    if not 0 < max_tokens_per_shard <= 100_000_000:
        raise ValueError("max_tokens_per_shard must be in [1, 100000000]")

    sample, texts = load_sample(sample_manifest_path)
    training_path = Path(tokenizer_training_manifest_path)
    training = load_training_manifest(training_path)
    if training.status != "complete":
        raise DataValidationError("tokenizer training is not complete")
    if (
        training.source_dataset_id != sample.dataset_id
        or training.source_revision != sample.revision
        or training.source_sample_sha256 != sample.sample_sha256
        or training.source_content_sha256 != sample.aggregate_content_sha256
    ):
        raise DataValidationError("tokenizer training source does not match sample")

    tokenizer_path = training_path.parent / training.tokenizer_file
    try:
        tokenizer = ByteBPE.load(tokenizer_path)
    except ValueError as error:
        raise DataValidationError(f"invalid tokenizer model: {error}") from error
    if tokenizer.model_hash != training.tokenizer_hash:
        raise DataValidationError("tokenizer hash does not match training manifest")
    if tokenizer.vocab_size != training.realized_vocab_size:
        raise DataValidationError(
            "tokenizer vocabulary does not match training manifest"
        )

    split_items = split_documents(texts, split_policy)
    if sum(len(items) for items in split_items.values()) != sample.document_count:
        raise DataValidationError("split document count does not match sample")

    destination = Path(output_directory)
    partial = destination.with_name(f".{destination.name}.partial")
    if destination.exists() or partial.exists():
        raise DataValidationError("refusing to overwrite shard build directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial.mkdir()
    shard_directory = partial / "shards"
    shard_directory.mkdir()

    stats: list[ShardSplitStats] = []
    for split in SPLIT_ORDER:
        documents = split_items[split]
        manifest_paths = write_token_shards(
            documents,
            tokenizer,
            shard_directory,
            split=split,
            max_tokens_per_shard=max_tokens_per_shard,
            source_name=sample.dataset_id,
            source_revision=sample.revision,
            split_seed=split_policy.seed,
        )
        manifests = [load_manifest(path) for path in manifest_paths]
        stats.append(
            ShardSplitStats(
                split=split,
                document_count=len(documents),
                token_count=sum(item.token_count for item in manifests),
                manifest_files=tuple(path.name for path in manifest_paths),
            )
        )

    split_stats = (stats[0], stats[1], stats[2])
    build = ShardBuildManifest(
        source_dataset_id=sample.dataset_id,
        source_revision=sample.revision,
        source_license=sample.license,
        source_sample_sha256=sample.sample_sha256,
        source_content_sha256=sample.aggregate_content_sha256,
        source_document_count=sample.document_count,
        source_text_bytes=sample.actual_text_bytes,
        tokenizer_hash=tokenizer.model_hash,
        tokenizer_vocab_size=tokenizer.vocab_size,
        split_policy=split_policy,
        max_tokens_per_shard=max_tokens_per_shard,
        splits=split_stats,
        total_token_count=sum(item.token_count for item in split_stats),
    )
    build_path = partial / "build.json"
    _atomic_json(build_path, build.model_dump(mode="json"))
    validate_shard_build(build_path)
    partial.replace(destination)
    return build
