"""Replayable tokenizer training against a checked text sample."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from itertools import islice
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lm_from_zero.data import DataValidationError
from lm_from_zero.sampling import SampleManifest, load_sample
from lm_from_zero.tokenizer.bpe import INITIAL_VOCAB_SIZE, ByteBPE, TokenPair
from lm_from_zero.tokenizer.pretokenizer import PretokenizerMode
from lm_from_zero.tokenizer.trainer import BPETrainingStats, train_bpe

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class TokenizerTrainingConfig(BaseModel):
    """Deterministic tokenizer training and checkpoint policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_vocab_size: Annotated[int, Field(ge=INITIAL_VOCAB_SIZE, le=65536)] = 16_000
    min_frequency: Annotated[int, Field(ge=1)] = 2
    pretokenizer: PretokenizerMode = "gpt2"
    checkpoint_every_merges: Annotated[int, Field(gt=0)] = 250
    max_documents: Annotated[int | None, Field(gt=0)] = None
    max_corpus_bytes: Annotated[int | None, Field(gt=0)] = None


class TokenizerTrainingManifest(BaseModel):
    """Replay contract and measured outcome for tokenizer training."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-tokenizer-training"] = (
        "lm-from-zero-tokenizer-training"
    )
    format_version: Literal[1] = 1
    status: Literal["in_progress", "complete"]
    training_config: TokenizerTrainingConfig
    source_dataset_id: str
    source_revision: str
    source_sample_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    source_content_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    tokenizer_file: str = "tokenizer.json"
    tokenizer_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    realized_vocab_size: Annotated[int, Field(ge=INITIAL_VOCAB_SIZE, le=65536)]
    merge_count: Annotated[int, Field(ge=0)]
    last_pair_frequency: Annotated[int | None, Field(ge=1)] = None
    resumed_from_merge_count: Annotated[int, Field(ge=0)]
    corpus_sha256: Annotated[str | None, Field(pattern=SHA256_PATTERN)] = None
    document_count: Annotated[int | None, Field(ge=0)] = None
    corpus_bytes: Annotated[int | None, Field(ge=0)] = None
    segment_count: Annotated[int | None, Field(ge=0)] = None
    unique_segment_count: Annotated[int | None, Field(ge=0)] = None
    elapsed_seconds: Annotated[float | None, Field(ge=0)] = None

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if Path(self.tokenizer_file).name != self.tokenizer_file:
            raise ValueError("tokenizer_file must not contain directories")
        if self.realized_vocab_size != INITIAL_VOCAB_SIZE + self.merge_count:
            raise ValueError("realized vocabulary and merge count disagree")
        measured = (
            self.corpus_sha256,
            self.document_count,
            self.corpus_bytes,
            self.segment_count,
            self.unique_segment_count,
            self.elapsed_seconds,
        )
        if self.status == "complete" and any(item is None for item in measured):
            raise ValueError("complete training manifests require measured statistics")
        return self


class _TrainingPaused(Exception):
    pass


def _atomic_json_replace(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    if temporary.exists():
        raise DataValidationError(f"incomplete training file exists: {temporary.name}")
    encoded = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_training_manifest(path: str | Path) -> TokenizerTrainingManifest:
    """Load and validate a tokenizer training manifest."""

    try:
        return TokenizerTrainingManifest.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise DataValidationError(
            f"invalid tokenizer training manifest: {error}"
        ) from error


def limit_texts(texts: Iterator[str], config: TokenizerTrainingConfig) -> Iterator[str]:
    selected: Iterator[str] = texts
    if config.max_documents is not None:
        selected = islice(selected, config.max_documents)
    if config.max_corpus_bytes is None:
        yield from selected
        return

    used_bytes = 0
    for text in selected:
        encoded_length = len(text.encode("utf-8"))
        if used_bytes and used_bytes + encoded_length > config.max_corpus_bytes:
            break
        used_bytes += encoded_length
        yield text
        if used_bytes >= config.max_corpus_bytes:
            break


def _manifest_for_checkpoint(
    *,
    status: Literal["in_progress", "complete"],
    config: TokenizerTrainingConfig,
    sample: SampleManifest,
    tokenizer: ByteBPE,
    last_pair_frequency: int | None,
    resumed_from_merge_count: int,
    stats: BPETrainingStats | None,
) -> TokenizerTrainingManifest:
    return TokenizerTrainingManifest(
        status=status,
        training_config=config,
        source_dataset_id=sample.dataset_id,
        source_revision=sample.revision,
        source_sample_sha256=sample.sample_sha256,
        source_content_sha256=sample.aggregate_content_sha256,
        tokenizer_hash=tokenizer.model_hash,
        realized_vocab_size=tokenizer.vocab_size,
        merge_count=len(tokenizer.merges),
        last_pair_frequency=last_pair_frequency,
        resumed_from_merge_count=resumed_from_merge_count,
        corpus_sha256=None if stats is None else stats.corpus_sha256,
        document_count=None if stats is None else stats.document_count,
        corpus_bytes=None if stats is None else stats.corpus_bytes,
        segment_count=None if stats is None else stats.segment_count,
        unique_segment_count=None if stats is None else stats.unique_segment_count,
        elapsed_seconds=None if stats is None else stats.elapsed_seconds,
    )


def train_tokenizer_from_sample(
    sample_manifest_path: str | Path,
    output_directory: str | Path,
    config: TokenizerTrainingConfig,
    *,
    stop_after_new_merges: int | None = None,
) -> TokenizerTrainingManifest:
    """Train or replay-resume from the last atomic tokenizer checkpoint."""

    if stop_after_new_merges is not None and stop_after_new_merges <= 0:
        raise ValueError("stop_after_new_merges must be positive")
    sample, texts = load_sample(sample_manifest_path)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_path = output / "tokenizer.json"
    manifest_path = output / "training.json"
    tokenizer_partial = tokenizer_path.with_name(f".{tokenizer_path.name}.tmp")
    if tokenizer_partial.exists():
        raise DataValidationError(
            f"incomplete tokenizer file exists: {tokenizer_partial.name}"
        )

    initial_merges: tuple[TokenPair, ...] = ()
    resumed_from = 0
    if manifest_path.exists() or tokenizer_path.exists():
        if not manifest_path.is_file() or not tokenizer_path.is_file():
            raise DataValidationError("tokenizer checkpoint is incomplete")
        previous = load_training_manifest(manifest_path)
        if previous.training_config != config:
            raise DataValidationError(
                "training configuration does not match checkpoint"
            )
        if (
            previous.source_sample_sha256 != sample.sample_sha256
            or previous.source_content_sha256 != sample.aggregate_content_sha256
            or previous.source_revision != sample.revision
        ):
            raise DataValidationError("source sample does not match checkpoint")
        tokenizer = ByteBPE.load(tokenizer_path)
        if tokenizer.model_hash != previous.tokenizer_hash:
            raise DataValidationError(
                "tokenizer checkpoint hash does not match manifest"
            )
        if tokenizer.pretokenizer != config.pretokenizer:
            raise DataValidationError(
                "tokenizer pretokenizer does not match checkpoint"
            )
        if previous.status == "complete":
            return previous
        initial_merges = tokenizer.merges
        resumed_from = len(initial_merges)

    last_pair_frequency: int | None = None
    added_merges = 0

    def checkpoint(merges: tuple[TokenPair, ...], pair_frequency: int) -> None:
        nonlocal added_merges, last_pair_frequency
        added_merges += 1
        last_pair_frequency = pair_frequency
        should_save = len(merges) % config.checkpoint_every_merges == 0
        should_pause = (
            stop_after_new_merges is not None and added_merges >= stop_after_new_merges
        )
        if should_save or should_pause:
            tokenizer = ByteBPE(merges=merges, pretokenizer=config.pretokenizer)
            tokenizer.save(tokenizer_path)
            checkpoint_manifest = _manifest_for_checkpoint(
                status="in_progress",
                config=config,
                sample=sample,
                tokenizer=tokenizer,
                last_pair_frequency=pair_frequency,
                resumed_from_merge_count=resumed_from,
                stats=None,
            )
            _atomic_json_replace(
                manifest_path, checkpoint_manifest.model_dump(mode="json")
            )
        if should_pause:
            raise _TrainingPaused

    try:
        result = train_bpe(
            limit_texts(texts, config),
            target_vocab_size=config.target_vocab_size,
            min_frequency=config.min_frequency,
            pretokenizer=config.pretokenizer,
            initial_merges=initial_merges,
            on_merge=checkpoint,
        )
    except _TrainingPaused:
        return load_training_manifest(manifest_path)

    result.tokenizer.save(tokenizer_path)
    final_manifest = _manifest_for_checkpoint(
        status="complete",
        config=config,
        sample=sample,
        tokenizer=result.tokenizer,
        last_pair_frequency=last_pair_frequency,
        resumed_from_merge_count=resumed_from,
        stats=result.stats,
    )
    _atomic_json_replace(manifest_path, final_manifest.model_dump(mode="json"))
    return final_manifest
