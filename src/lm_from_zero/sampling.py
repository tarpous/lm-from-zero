"""Deterministic, size-bounded dataset sampling with checked provenance."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self, cast

from datasets import load_dataset  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from lm_from_zero.data import DataValidationError
from lm_from_zero.progress import ProgressReporter

TINYSTORIES_DATASET_ID = "roneneldan/TinyStories"
TINYSTORIES_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TINYSTORIES_LICENSE = "cdla-sharing-1.0"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
GIT_COMMIT_PATTERN = r"^[0-9a-f]{40}$"


class SamplingConfig(BaseModel):
    """Pinned source and local safety policy for a deterministic sample."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: Annotated[str, Field(min_length=1)] = TINYSTORIES_DATASET_ID
    revision: Annotated[str, Field(pattern=GIT_COMMIT_PATTERN)] = TINYSTORIES_REVISION
    config_name: Annotated[str, Field(min_length=1)] = "default"
    split: Annotated[str, Field(min_length=1)] = "train"
    text_field: Annotated[str, Field(min_length=1)] = "text"
    target_text_bytes: Annotated[int, Field(gt=0)] = 100_000_000
    max_storage_bytes: Annotated[int, Field(gt=0)] = 1_000_000_000
    license: Annotated[str, Field(min_length=1)] = TINYSTORIES_LICENSE

    @model_validator(mode="after")
    def validate_storage_budget(self) -> Self:
        if self.max_storage_bytes <= self.target_text_bytes:
            raise ValueError("max_storage_bytes must exceed target_text_bytes")
        return self


class SampleManifest(BaseModel):
    """Integrity, selection, and source metadata for a text sample."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-text-sample"] = "lm-from-zero-text-sample"
    format_version: Literal[1] = 1
    status: Literal["complete"] = "complete"
    dataset_id: str
    revision: Annotated[str, Field(pattern=GIT_COMMIT_PATTERN)]
    config_name: str
    split: str
    text_field: str
    license: str
    selection: Literal["source-order-prefix-deduplicated"]
    target_text_bytes: Annotated[int, Field(gt=0)]
    actual_text_bytes: Annotated[int, Field(gt=0)]
    sample_file: str
    sample_file_bytes: Annotated[int, Field(gt=0)]
    sample_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    document_hashes_file: str
    document_hashes_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    aggregate_content_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    document_count: Annotated[int, Field(gt=0)]
    source_rows_scanned: Annotated[int, Field(gt=0)]
    next_source_index: Annotated[int, Field(gt=0)]
    duplicate_rows_skipped: Annotated[int, Field(ge=0)]
    empty_rows_skipped: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_internal_consistency(self) -> Self:
        for filename in (self.sample_file, self.document_hashes_file):
            if Path(filename).name != filename:
                raise ValueError("sample artifact fields must contain filenames")
        if self.actual_text_bytes < self.target_text_bytes:
            raise ValueError("actual_text_bytes must reach target_text_bytes")
        if self.next_source_index != self.source_rows_scanned:
            raise ValueError("next_source_index must equal source_rows_scanned")
        accounted_rows = (
            self.document_count + self.duplicate_rows_skipped + self.empty_rows_skipped
        )
        if accounted_rows != self.source_rows_scanned:
            raise ValueError("sample row accounting does not match rows scanned")
        return self


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _storage_size(paths: Sequence[Path]) -> int:
    return sum(_directory_size(path) for path in paths)


def _check_storage_limit(paths: Sequence[Path], maximum: int) -> None:
    used = _storage_size(paths)
    if used > maximum:
        raise DataValidationError(
            f"sample storage safety limit exceeded: {used} > {maximum} bytes"
        )


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    partial = path.with_name(f".{path.name}.partial")
    if path.exists() or partial.exists():
        raise DataValidationError(f"refusing to overwrite sample file: {path.name}")
    encoded = (
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    with partial.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    partial.replace(path)


def sample_text_records(
    records: Iterable[Mapping[str, object]],
    output_directory: str | Path,
    config: SamplingConfig,
    *,
    storage_paths: Sequence[str | Path] = (),
) -> Path:
    """Write a deterministic deduplicated prefix and its complete manifest."""

    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    sample_path = destination / "documents.jsonl"
    hashes_path = destination / "documents.sha256"
    manifest_path = destination / "manifest.json"
    sample_partial = destination / ".documents.jsonl.partial"
    hashes_partial = destination / ".documents.sha256.partial"
    for path in (
        sample_path,
        hashes_path,
        manifest_path,
        sample_partial,
        hashes_partial,
    ):
        if path.exists():
            raise DataValidationError(f"refusing to overwrite sample file: {path.name}")

    checked_storage_paths = [destination]
    checked_storage_paths.extend(Path(path) for path in storage_paths)
    seen_hashes: set[str] = set()
    aggregate_digest = sha256()
    actual_text_bytes = 0
    document_count = 0
    rows_scanned = 0
    duplicate_rows = 0
    empty_rows = 0
    progress = ProgressReporter("sample building")
    progress.phase("scanning source", fields={"target_bytes": config.target_text_bytes})

    try:
        with (
            sample_partial.open("xb") as sample_handle,
            hashes_partial.open("xb") as hashes_handle,
        ):
            for source_index, record in enumerate(records):
                rows_scanned = source_index + 1
                progress.update(
                    rows_scanned,
                    fields={
                        "documents": document_count,
                        "bytes": actual_text_bytes,
                    },
                )
                raw_text = record.get(config.text_field)
                if not isinstance(raw_text, str):
                    raise DataValidationError(
                        f"row {source_index} field {config.text_field!r} is not text"
                    )
                text_bytes = raw_text.encode("utf-8")
                if not text_bytes:
                    empty_rows += 1
                    continue
                content_hash = sha256(text_bytes).hexdigest()
                if content_hash in seen_hashes:
                    duplicate_rows += 1
                    continue

                encoded_record = (
                    json.dumps(
                        {config.text_field: raw_text},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                sample_handle.write(encoded_record)
                hashes_handle.write(f"{content_hash}\n".encode())
                seen_hashes.add(content_hash)
                aggregate_digest.update(len(text_bytes).to_bytes(8, "big"))
                aggregate_digest.update(text_bytes)
                actual_text_bytes += len(text_bytes)
                document_count += 1

                if document_count % 1000 == 0:
                    sample_handle.flush()
                    hashes_handle.flush()
                    _check_storage_limit(
                        checked_storage_paths, config.max_storage_bytes
                    )
                if actual_text_bytes >= config.target_text_bytes:
                    break

            if actual_text_bytes < config.target_text_bytes:
                raise DataValidationError(
                    "source ended before the requested sample size was reached"
                )
            sample_handle.flush()
            hashes_handle.flush()
            os.fsync(sample_handle.fileno())
            os.fsync(hashes_handle.fileno())

        _check_storage_limit(checked_storage_paths, config.max_storage_bytes)
        manifest = SampleManifest(
            dataset_id=config.dataset_id,
            revision=config.revision,
            config_name=config.config_name,
            split=config.split,
            text_field=config.text_field,
            license=config.license,
            selection="source-order-prefix-deduplicated",
            target_text_bytes=config.target_text_bytes,
            actual_text_bytes=actual_text_bytes,
            sample_file=sample_path.name,
            sample_file_bytes=sample_partial.stat().st_size,
            sample_sha256=_file_sha256(sample_partial),
            document_hashes_file=hashes_path.name,
            document_hashes_sha256=_file_sha256(hashes_partial),
            aggregate_content_sha256=aggregate_digest.hexdigest(),
            document_count=document_count,
            source_rows_scanned=rows_scanned,
            next_source_index=rows_scanned,
            duplicate_rows_skipped=duplicate_rows,
            empty_rows_skipped=empty_rows,
        )
        sample_partial.replace(sample_path)
        hashes_partial.replace(hashes_path)
        _atomic_json(manifest_path, manifest.model_dump(mode="json"))
        progress.finish("complete")
        return manifest_path
    except Exception:
        progress.finish("failed")
        sample_partial.unlink(missing_ok=True)
        hashes_partial.unlink(missing_ok=True)
        raise


def stream_hugging_face_sample(
    output_directory: str | Path,
    cache_directory: str | Path,
    config: SamplingConfig,
) -> Path:
    """Stream the pinned Hugging Face split into a checked local sample."""

    cache = Path(cache_directory)
    cache.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        config.dataset_id,
        config.config_name,
        split=config.split,
        revision=config.revision,
        streaming=True,
        cache_dir=str(cache),
    )
    records = cast(Iterable[Mapping[str, object]], dataset)
    return sample_text_records(
        records,
        output_directory,
        config,
        storage_paths=(cache,),
    )


def load_sample(manifest_path: str | Path) -> tuple[SampleManifest, Iterator[str]]:
    """Validate sample artifacts and return a lazy exact-text iterator."""

    path = Path(manifest_path)
    try:
        manifest = SampleManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DataValidationError(f"invalid sample manifest: {error}") from error
    sample_path = path.parent / manifest.sample_file
    hashes_path = path.parent / manifest.document_hashes_file
    for artifact, expected_hash in (
        (sample_path, manifest.sample_sha256),
        (hashes_path, manifest.document_hashes_sha256),
    ):
        if not artifact.is_file():
            raise DataValidationError(f"sample artifact is missing: {artifact.name}")
        if _file_sha256(artifact) != expected_hash:
            raise DataValidationError(
                f"sample artifact checksum mismatch: {artifact.name}"
            )
    if sample_path.stat().st_size != manifest.sample_file_bytes:
        raise DataValidationError("sample file byte length does not match manifest")

    def iter_texts() -> Iterator[str]:
        content_digest = sha256()
        count = 0
        text_bytes = 0
        with (
            sample_path.open(encoding="utf-8") as sample_handle,
            hashes_path.open(encoding="ascii") as hashes_handle,
        ):
            for line, expected_content_hash in zip(
                sample_handle, hashes_handle, strict=True
            ):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DataValidationError("sample contains invalid JSON") from error
                text = payload.get(manifest.text_field)
                if not isinstance(text, str):
                    raise DataValidationError("sample row does not contain text")
                encoded = text.encode("utf-8")
                content_hash = sha256(encoded).hexdigest()
                if content_hash != expected_content_hash.rstrip("\n"):
                    raise DataValidationError("sample document hash mismatch")
                content_digest.update(len(encoded).to_bytes(8, "big"))
                content_digest.update(encoded)
                count += 1
                text_bytes += len(encoded)
                yield text
        if count != manifest.document_count:
            raise DataValidationError("sample document count does not match manifest")
        if text_bytes != manifest.actual_text_bytes:
            raise DataValidationError("sample text byte count does not match manifest")
        if content_digest.hexdigest() != manifest.aggregate_content_sha256:
            raise DataValidationError("sample aggregate content hash mismatch")

    return manifest, iter_texts()
