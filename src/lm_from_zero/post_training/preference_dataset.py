"""Deterministic UltraFeedback preference-mix preparation."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Final, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from lm_from_zero.post_training.chat import (
    DEFAULT_CHAT_TEMPLATE,
    ChatMessage,
    Conversation,
)
from lm_from_zero.post_training.dpo import (
    DPOFormatError,
    PreferencePair,
    render_preference_pair,
)
from lm_from_zero.tokenizer.bpe import ByteBPE

PREFERENCE_DATASET_ID: Final = "HuggingFaceH4/ultrafeedback_binarized"
PREFERENCE_DATASET_REVISION: Final = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"
PREFERENCE_SOURCE_SPLIT: Final = "train_prefs"
PREFERENCE_SOURCE_FILE: Final = "data/train_prefs-00000-of-00001.parquet"
PREFERENCE_RECORD_FORMAT: Final = "lm-from-zero-preference-record"
PREFERENCE_MANIFEST_FORMAT: Final = "lm-from-zero-preference-mix-manifest"
PREFERENCE_HOLDOUT_MANIFEST_FORMAT: Final = "lm-from-zero-preference-holdout-manifest"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
REVISION_PATTERN = r"^[0-9a-f]{40,64}$"


class PreferenceDatasetError(ValueError):
    """Raised when a preference source cannot produce a valid deterministic mix."""


class PreferenceRecord(BaseModel):
    """Canonical preference pair with immutable source coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-preference-record"] = PREFERENCE_RECORD_FORMAT
    format_version: Literal[1] = 1
    dataset_id: Literal["HuggingFaceH4/ultrafeedback_binarized"] = PREFERENCE_DATASET_ID
    dataset_revision: Annotated[str, Field(pattern=REVISION_PATTERN)] = (
        PREFERENCE_DATASET_REVISION
    )
    source_split: Literal["train_prefs"] = PREFERENCE_SOURCE_SPLIT
    source_index: Annotated[int, Field(ge=0)]
    prompt_id: str = Field(min_length=1)
    pair: PreferencePair
    score_chosen: float
    score_rejected: float


class PreferenceMixManifest(BaseModel):
    """Manifest binding source, selection, tokenizer, and canonical records."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-preference-mix-manifest"] = PREFERENCE_MANIFEST_FORMAT
    format_version: Literal[1] = 1
    dataset_id: Literal["HuggingFaceH4/ultrafeedback_binarized"] = PREFERENCE_DATASET_ID
    dataset_revision: Annotated[str, Field(pattern=REVISION_PATTERN)] = (
        PREFERENCE_DATASET_REVISION
    )
    source_split: Literal["train_prefs"] = PREFERENCE_SOURCE_SPLIT
    source_file: Literal["data/train_prefs-00000-of-00001.parquet"] = (
        PREFERENCE_SOURCE_FILE
    )
    source_file_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    tokenizer_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    chat_template_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    max_length: Annotated[int, Field(gt=1)] = 1_024
    truncation_policy: Literal["left"] = "left"
    filtering_policy: Literal[
        "canonical_pair_shared_prompt_finite_scores_renderable"
    ] = "canonical_pair_shared_prompt_finite_scores_renderable"
    selection_seed: int
    selection_policy: Literal["prompt_id_sha256_ascending"] = (
        "prompt_id_sha256_ascending"
    )
    target_pairs: Annotated[int, Field(gt=0)]
    source_rows: Annotated[int, Field(ge=0)]
    valid_pairs: Annotated[int, Field(ge=0)]
    rejected_rows: Annotated[int, Field(ge=0)]
    selected_pairs: Annotated[int, Field(gt=0)]
    truncated_pairs: Annotated[int, Field(ge=0)]
    chosen_truncated_pairs: Annotated[int, Field(ge=0)]
    rejected_truncated_pairs: Annotated[int, Field(ge=0)]
    records_jsonl: Literal["records.jsonl"] = "records.jsonl"
    records_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]

    def canonical_json(self) -> str:
        """Return the stable manifest encoding."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


class PreferenceHoldoutManifest(BaseModel):
    """Manifest for a deterministic preference split excluded from DPO training."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-preference-holdout-manifest"] = (
        PREFERENCE_HOLDOUT_MANIFEST_FORMAT
    )
    format_version: Literal[1] = 1
    dataset_id: Literal["HuggingFaceH4/ultrafeedback_binarized"] = PREFERENCE_DATASET_ID
    dataset_revision: Annotated[str, Field(pattern=REVISION_PATTERN)] = (
        PREFERENCE_DATASET_REVISION
    )
    source_split: Literal["train_prefs"] = PREFERENCE_SOURCE_SPLIT
    source_file: Literal["data/train_prefs-00000-of-00001.parquet"] = (
        PREFERENCE_SOURCE_FILE
    )
    source_file_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    train_manifest_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    train_records_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    tokenizer_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    chat_template_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    max_length: Annotated[int, Field(gt=1)] = 1_024
    truncation_policy: Literal["left"] = "left"
    selection_seed: int
    selection_policy: Literal["prompt_id_sha256_ascending"] = (
        "prompt_id_sha256_ascending"
    )
    source_rows: Annotated[int, Field(ge=0)]
    valid_pairs: Annotated[int, Field(ge=0)]
    rejected_rows: Annotated[int, Field(ge=0)]
    excluded_pairs: Annotated[int, Field(gt=0)]
    target_pairs: Annotated[int, Field(gt=0)]
    selected_pairs: Annotated[int, Field(gt=0)]
    truncated_pairs: Annotated[int, Field(ge=0)]
    chosen_truncated_pairs: Annotated[int, Field(ge=0)]
    rejected_truncated_pairs: Annotated[int, Field(ge=0)]
    records_jsonl: Literal["records.jsonl"] = "records.jsonl"
    records_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]

    def canonical_json(self) -> str:
        """Return the stable holdout manifest encoding."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


@dataclass(frozen=True, slots=True)
class _Candidate:
    rank: str
    record: PreferenceRecord
    chosen_truncated: bool
    rejected_truncated: bool


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise PreferenceDatasetError(
            f"incomplete preference artifact exists: {temporary}"
        )
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _parquet_rows(source_path: Path) -> Iterable[Mapping[str, object]]:
    try:
        import pyarrow.parquet as parquet  # type: ignore[import-untyped]
    except ImportError as error:  # pragma: no cover - environment contract
        raise PreferenceDatasetError(
            "preference preparation requires pyarrow"
        ) from error
    try:
        parquet_file = parquet.ParquetFile(source_path)
        for batch in parquet_file.iter_batches(
            columns=[
                "prompt_id",
                "chosen",
                "rejected",
                "score_chosen",
                "score_rejected",
            ],
            batch_size=1_024,
        ):
            yield from cast(Iterable[Mapping[str, object]], batch.to_pylist())
    except (OSError, ValueError) as error:
        raise PreferenceDatasetError(
            f"cannot read preference parquet: {error}"
        ) from error


def _message_tuple(value: object, *, field_name: str) -> tuple[ChatMessage, ...]:
    if not isinstance(value, list):
        raise PreferenceDatasetError(f"{field_name} must be a list of messages")
    try:
        return tuple(ChatMessage.model_validate(message) for message in value)
    except (TypeError, ValueError) as error:
        raise PreferenceDatasetError(
            f"{field_name} contains invalid messages"
        ) from error


def _record_from_row(
    row: Mapping[str, object], *, source_index: int
) -> PreferenceRecord:
    prompt_id = row.get("prompt_id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise PreferenceDatasetError(
            "preference row prompt_id must be a non-empty string"
        )
    chosen_messages = _message_tuple(row.get("chosen"), field_name="chosen")
    rejected_messages = _message_tuple(row.get("rejected"), field_name="rejected")
    if len(chosen_messages) < 2 or len(rejected_messages) < 2:
        raise PreferenceDatasetError(
            "preference responses require a prompt and response"
        )
    if chosen_messages[:-1] != rejected_messages[:-1]:
        raise PreferenceDatasetError("chosen and rejected prompts must be identical")
    raw_score_chosen = row.get("score_chosen")
    raw_score_rejected = row.get("score_rejected")
    if not isinstance(raw_score_chosen, (int, float)) or isinstance(
        raw_score_chosen, bool
    ):
        raise PreferenceDatasetError("score_chosen must be numeric")
    if not isinstance(raw_score_rejected, (int, float)) or isinstance(
        raw_score_rejected, bool
    ):
        raise PreferenceDatasetError("score_rejected must be numeric")
    try:
        pair = PreferencePair(
            prompt=Conversation(messages=chosen_messages[:-1]),
            chosen=chosen_messages[-1],
            rejected=rejected_messages[-1],
        )
        score_chosen = float(raw_score_chosen)
        score_rejected = float(raw_score_rejected)
    except (KeyError, TypeError, ValueError) as error:
        raise PreferenceDatasetError(
            "preference row failed canonical validation"
        ) from error
    return PreferenceRecord(
        source_index=source_index,
        prompt_id=prompt_id,
        pair=pair,
        score_chosen=score_chosen,
        score_rejected=score_rejected,
    )


def _canonical_record_line(record: PreferenceRecord) -> bytes:
    return (
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _collect_candidates(
    source_path: Path,
    tokenizer: ByteBPE,
    *,
    max_length: int,
    selection_seed: int,
    loader: Callable[[Path], Iterable[Mapping[str, object]]] | None,
) -> tuple[list[_Candidate], int, int]:
    candidates: list[_Candidate] = []
    source_rows = 0
    rejected_rows = 0
    rows = _parquet_rows(source_path) if loader is None else loader(source_path)
    for source_index, row in enumerate(rows):
        source_rows += 1
        try:
            record = _record_from_row(row, source_index=source_index)
            rendered = render_preference_pair(
                record.pair,
                tokenizer,
                max_length=max_length,
                truncation="left",
            )
        except (DPOFormatError, PreferenceDatasetError, TypeError, ValueError):
            rejected_rows += 1
            continue
        rank = sha256(
            f"{selection_seed}:{record.prompt_id}:{source_index}".encode()
        ).hexdigest()
        candidates.append(
            _Candidate(
                rank=rank,
                record=record,
                chosen_truncated=rendered.chosen.truncated,
                rejected_truncated=rendered.rejected.truncated,
            )
        )
    return candidates, source_rows, rejected_rows


def _write_records(
    output: Path,
    selected: list[_Candidate],
) -> tuple[str, int, int, int]:
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "records.jsonl"
    records_temporary = records_path.with_name(f".{records_path.name}.tmp")
    if records_temporary.exists():
        raise PreferenceDatasetError(
            f"incomplete preference records exists: {records_temporary}"
        )
    digest = sha256()
    try:
        with records_temporary.open("xb") as handle:
            for candidate in selected:
                line = _canonical_record_line(candidate.record)
                handle.write(line)
                digest.update(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(records_temporary, records_path)
    except Exception:
        records_temporary.unlink(missing_ok=True)
        raise
    chosen_truncated_pairs = sum(item.chosen_truncated for item in selected)
    rejected_truncated_pairs = sum(item.rejected_truncated for item in selected)
    return (
        digest.hexdigest(),
        sum(item.chosen_truncated or item.rejected_truncated for item in selected),
        chosen_truncated_pairs,
        rejected_truncated_pairs,
    )


def prepare_preference_mix(
    source_parquet: str | Path,
    output_directory: str | Path,
    tokenizer_path: str | Path,
    *,
    target_pairs: int = 50_000,
    selection_seed: int = 1_337,
    max_length: int = 1_024,
    loader: Callable[[Path], Iterable[Mapping[str, object]]] | None = None,
) -> PreferenceMixManifest:
    """Validate, tokenize-check, select, and atomically write preference pairs."""

    if target_pairs <= 0:
        raise PreferenceDatasetError("target_pairs must be positive")
    source_path = Path(source_parquet)
    tokenizer_file = Path(tokenizer_path)
    if not source_path.is_file():
        raise PreferenceDatasetError(
            f"preference parquet does not exist: {source_path}"
        )
    if not tokenizer_file.is_file():
        raise PreferenceDatasetError(f"tokenizer does not exist: {tokenizer_file}")
    try:
        tokenizer = ByteBPE.load(tokenizer_file)
    except (OSError, ValueError) as error:
        raise PreferenceDatasetError(
            f"cannot load preference tokenizer: {error}"
        ) from error

    candidates, source_rows, rejected_rows = _collect_candidates(
        source_path,
        tokenizer,
        max_length=max_length,
        selection_seed=selection_seed,
        loader=loader,
    )

    if len(candidates) < target_pairs:
        raise PreferenceDatasetError(
            f"source yielded {len(candidates)} valid preference pairs, "
            f"requested {target_pairs}"
        )
    selected = sorted(
        candidates,
        key=lambda item: (item.rank, item.record.prompt_id, item.record.source_index),
    )[:target_pairs]

    output = Path(output_directory)
    (
        records_digest,
        truncated_pairs,
        chosen_truncated_pairs,
        rejected_truncated_pairs,
    ) = _write_records(output, selected)
    manifest_path = output / "manifest.json"
    manifest = PreferenceMixManifest(
        source_file_sha256=_sha256_file(source_path),
        tokenizer_hash=tokenizer.model_hash,
        chat_template_hash=DEFAULT_CHAT_TEMPLATE.template_hash,
        max_length=max_length,
        selection_seed=selection_seed,
        target_pairs=target_pairs,
        source_rows=source_rows,
        valid_pairs=len(candidates),
        rejected_rows=rejected_rows,
        selected_pairs=len(selected),
        truncated_pairs=truncated_pairs,
        chosen_truncated_pairs=chosen_truncated_pairs,
        rejected_truncated_pairs=rejected_truncated_pairs,
        records_sha256=records_digest,
    )
    _atomic_write(manifest_path, (manifest.canonical_json() + "\n").encode())
    return manifest


def prepare_preference_holdout(
    source_parquet: str | Path,
    training_manifest_path: str | Path,
    output_directory: str | Path,
    tokenizer_path: str | Path,
    *,
    target_pairs: int | None = None,
    loader: Callable[[Path], Iterable[Mapping[str, object]]] | None = None,
) -> PreferenceHoldoutManifest:
    """Prepare all or a deterministic prefix of pairs excluded from DPO training."""

    source_path = Path(source_parquet)
    train_manifest_file = Path(training_manifest_path)
    tokenizer_file = Path(tokenizer_path)
    output = Path(output_directory)
    if not source_path.is_file():
        raise PreferenceDatasetError(
            f"preference parquet does not exist: {source_path}"
        )
    if not train_manifest_file.is_file():
        raise PreferenceDatasetError(
            f"training preference manifest does not exist: {train_manifest_file}"
        )
    if not tokenizer_file.is_file():
        raise PreferenceDatasetError(f"tokenizer does not exist: {tokenizer_file}")
    if output.resolve() == train_manifest_file.parent.resolve():
        raise PreferenceDatasetError("holdout output must differ from training mix")
    try:
        tokenizer = ByteBPE.load(tokenizer_file)
        train_manifest = PreferenceMixManifest.model_validate_json(
            train_manifest_file.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise PreferenceDatasetError(
            f"cannot load preference training artifacts: {error}"
        ) from error
    source_hash = _sha256_file(source_path)
    if source_hash != train_manifest.source_file_sha256:
        raise PreferenceDatasetError("source parquet does not match training manifest")
    if tokenizer.model_hash != train_manifest.tokenizer_hash:
        raise PreferenceDatasetError("tokenizer does not match training manifest")
    if train_manifest.chat_template_hash != DEFAULT_CHAT_TEMPLATE.template_hash:
        raise PreferenceDatasetError("training manifest chat template is unsupported")
    train_records_path = train_manifest_file.parent / train_manifest.records_jsonl
    if _sha256_file(train_records_path) != train_manifest.records_sha256:
        raise PreferenceDatasetError(
            "training preference records do not match manifest"
        )
    excluded_keys: set[tuple[int, str]] = set()
    record_count = 0
    with train_records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = PreferenceRecord.model_validate_json(line)
            excluded_keys.add((record.source_index, record.prompt_id))
            record_count += 1
    if record_count != train_manifest.selected_pairs:
        raise PreferenceDatasetError("training record count disagrees with manifest")

    candidates, source_rows, rejected_rows = _collect_candidates(
        source_path,
        tokenizer,
        max_length=train_manifest.max_length,
        selection_seed=train_manifest.selection_seed,
        loader=loader,
    )
    ordered = sorted(
        (
            candidate
            for candidate in candidates
            if (candidate.record.source_index, candidate.record.prompt_id)
            not in excluded_keys
        ),
        key=lambda item: (item.rank, item.record.prompt_id, item.record.source_index),
    )
    if target_pairs is None:
        target_pairs = len(ordered)
    if target_pairs <= 0:
        raise PreferenceDatasetError("target_pairs must be positive")
    if len(ordered) < target_pairs:
        raise PreferenceDatasetError(
            f"source yielded {len(ordered)} holdout pairs, requested {target_pairs}"
        )
    selected = ordered[:target_pairs]
    (
        records_digest,
        truncated_pairs,
        chosen_truncated_pairs,
        rejected_truncated_pairs,
    ) = _write_records(output, selected)
    manifest = PreferenceHoldoutManifest(
        source_file_sha256=source_hash,
        train_manifest_sha256=_sha256_file(train_manifest_file),
        train_records_sha256=train_manifest.records_sha256,
        tokenizer_hash=tokenizer.model_hash,
        chat_template_hash=DEFAULT_CHAT_TEMPLATE.template_hash,
        max_length=train_manifest.max_length,
        selection_seed=train_manifest.selection_seed,
        source_rows=source_rows,
        valid_pairs=len(candidates),
        rejected_rows=rejected_rows,
        excluded_pairs=record_count,
        target_pairs=target_pairs,
        selected_pairs=len(selected),
        truncated_pairs=truncated_pairs,
        chosen_truncated_pairs=chosen_truncated_pairs,
        rejected_truncated_pairs=rejected_truncated_pairs,
        records_sha256=records_digest,
    )
    _atomic_write(output / "manifest.json", (manifest.canonical_json() + "\n").encode())
    return manifest
