"""Deterministic, provenance-bound SmolTalk2 SFT mix preparation."""

from __future__ import annotations

import gc
import json
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from lm_from_zero.post_training.chat import ChatMessage, Conversation

DATASET_ID = "HuggingFaceTB/smoltalk2"
DATASET_CONFIG = "SFT"
DATASET_REVISION = "fc6cc2103c066455aade5d7fbb346039ae36ca5e"
SFT_MIX_FORMAT = "lm-from-zero-sft-mix-manifest"
SFT_RECORD_FORMAT = "lm-from-zero-sft-record"
SFT_RECORDS_FILENAME = "records.jsonl"
SFT_MANIFEST_FILENAME = "manifest.json"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
REVISION_PATTERN = r"^[0-9a-f]{40,64}$"


class SFTDatasetError(ValueError):
    """Raised when the pinned SFT source cannot produce the requested mix."""


class SFTSourceSpec(BaseModel):
    """Pinned source split and the published SmolTalk2 mix weight."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    split: str = Field(min_length=1)
    available_examples: Annotated[int, Field(gt=0)]
    weight: Annotated[float, Field(gt=0)]


class SFTSourceSummary(BaseModel):
    """Measured extraction counts for one source split."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    split: str = Field(min_length=1)
    available_examples: Annotated[int, Field(gt=0)]
    weight: Annotated[float, Field(gt=0)]
    requested_examples: Annotated[int, Field(ge=0)]
    selected_examples: Annotated[int, Field(ge=0)]
    scanned_examples: Annotated[int, Field(ge=0)]
    rejected_examples: Annotated[int, Field(ge=0)]


class SFTRecord(BaseModel):
    """Canonical raw conversation record with immutable source coordinates."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-sft-record"] = "lm-from-zero-sft-record"
    format_version: Literal[1] = 1
    dataset_id: Literal["HuggingFaceTB/smoltalk2"] = "HuggingFaceTB/smoltalk2"
    dataset_config: Literal["SFT"] = "SFT"
    dataset_revision: Annotated[str, Field(pattern=REVISION_PATTERN)] = DATASET_REVISION
    source_split: str = Field(min_length=1)
    source_index: Annotated[int, Field(ge=0)]
    messages: tuple[ChatMessage, ...] = Field(min_length=1)


class SFTMixManifest(BaseModel):
    """Canonical manifest for one deterministic 100k-example SFT mix."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-sft-mix-manifest"] = "lm-from-zero-sft-mix-manifest"
    format_version: Literal[1] = 1
    dataset_id: Literal["HuggingFaceTB/smoltalk2"] = "HuggingFaceTB/smoltalk2"
    dataset_config: Literal["SFT"] = "SFT"
    dataset_revision: Annotated[str, Field(pattern=REVISION_PATTERN)] = DATASET_REVISION
    selection_seed: int
    selection_policy: Literal["source_order_first_valid"] = "source_order_first_valid"
    target_examples: Annotated[int, Field(gt=0)]
    selected_examples: Annotated[int, Field(gt=0)]
    records_jsonl: str = SFT_RECORDS_FILENAME
    records_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    sources: tuple[SFTSourceSummary, ...]

    def canonical_json(self) -> str:
        """Return the stable manifest encoding."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


SMOLTALK2_NO_THINK_SOURCES: tuple[SFTSourceSpec, ...] = (
    SFTSourceSpec(
        split="smoltalk_smollm3_everyday_conversations_no_think",
        available_examples=2_260,
        weight=1.0,
    ),
    SFTSourceSpec(
        split="smoltalk_smollm3_systemchats_30k_no_think",
        available_examples=33_997,
        weight=1.0,
    ),
    SFTSourceSpec(
        split="tulu_3_sft_personas_instruction_following_no_think",
        available_examples=29_970,
        weight=1.0,
    ),
    SFTSourceSpec(
        split="hermes_function_calling_v1_no_think",
        available_examples=8_961,
        weight=1.0,
    ),
    SFTSourceSpec(
        split="smoltalk_smollm3_smol_magpie_ultra_no_think",
        available_examples=406_843,
        weight=0.5,
    ),
    SFTSourceSpec(
        split="smoltalk_multilingual_8languages_lang_5_no_think",
        available_examples=254_047,
        weight=1.0,
    ),
    SFTSourceSpec(split="table_gpt_no_think", available_examples=13_203, weight=1.0),
    SFTSourceSpec(
        split="OpenHermes_2.5_no_think",
        available_examples=384_900,
        weight=0.5,
    ),
    SFTSourceSpec(
        split="OpenThoughts3_1.2M_no_think_no_think",
        available_examples=435_193,
        weight=0.4,
    ),
    SFTSourceSpec(
        split="Mixture_of_Thoughts_science_no_think",
        available_examples=86_110,
        weight=1.0,
    ),
    SFTSourceSpec(
        split="smoltalk_smollm3_explore_instruct_rewriting_no_think",
        available_examples=30_391,
        weight=1.0,
    ),
    SFTSourceSpec(
        split="smoltalk_smollm3_smol_rewrite_no_think",
        available_examples=53_262,
        weight=1.0,
    ),
    SFTSourceSpec(
        split="smoltalk_smollm3_smol_summarize_no_think",
        available_examples=96_061,
        weight=1.0,
    ),
    SFTSourceSpec(
        split="LongAlign_64k_context_lang_annotated_lang_6_no_think",
        available_examples=6_249,
        weight=1.0,
    ),
    SFTSourceSpec(split="xlam_traces_no_think", available_examples=59_962, weight=1.0),
)


@dataclass(frozen=True, slots=True)
class _SourceExtraction:
    summary: SFTSourceSummary
    records: tuple[SFTRecord, ...]


def allocate_sft_counts(
    sources: Iterable[SFTSourceSpec],
    target_examples: int,
) -> dict[str, int]:
    """Allocate a target by weighted available-example mass exactly."""

    source_list = tuple(sources)
    if not source_list or target_examples <= 0:
        raise SFTDatasetError("SFT allocation requires sources and a positive target")
    total_mass = sum(
        Decimal(str(source.available_examples)) * Decimal(str(source.weight))
        for source in source_list
    )
    if total_mass < target_examples:
        raise SFTDatasetError("SFT target exceeds weighted source capacity")
    raw_counts = {
        source.split: Decimal(target_examples)
        * Decimal(str(source.available_examples))
        * Decimal(str(source.weight))
        / total_mass
        for source in source_list
    }
    counts = {
        split: int(value.to_integral_value(rounding=ROUND_FLOOR))
        for split, value in raw_counts.items()
    }
    remaining = target_examples - sum(counts.values())
    order = sorted(
        raw_counts,
        key=lambda split: (-(raw_counts[split] - counts[split]), split),
    )
    for split in order[:remaining]:
        counts[split] += 1
    return counts


def _canonical_record_line(record: SFTRecord) -> bytes:
    return (
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _record_from_row(
    row: Mapping[str, object],
    *,
    split: str,
    source_index: int,
) -> SFTRecord:
    raw_messages = row.get("messages")
    if not isinstance(raw_messages, list):
        raise SFTDatasetError("SFT row messages must be a list")
    messages = tuple(ChatMessage.model_validate(message) for message in raw_messages)
    conversation = Conversation(messages=messages)
    if conversation.messages[-1].role != "assistant":
        raise SFTDatasetError("SFT row must end with an assistant turn")
    return SFTRecord(
        source_split=split,
        source_index=source_index,
        messages=conversation.messages,
    )


def _hub_source_loader(split: str) -> Iterable[Mapping[str, object]]:
    from datasets import load_dataset  # type: ignore[import-untyped]

    return cast(
        Iterable[Mapping[str, object]],
        load_dataset(
            DATASET_ID,
            DATASET_CONFIG,
            split=split,
            revision=DATASET_REVISION,
            streaming=True,
        ),
    )


def _extract_source(
    source: SFTSourceSpec,
    requested_examples: int,
    loader: Callable[[str], Iterable[Mapping[str, object]]],
) -> _SourceExtraction:
    selected: list[SFTRecord] = []
    scanned = 0
    rejected = 0
    if requested_examples:
        rows = loader(source.split)
        try:
            for source_index, row in enumerate(rows):
                scanned += 1
                try:
                    record = _record_from_row(
                        row,
                        split=source.split,
                        source_index=source_index,
                    )
                except (SFTDatasetError, ValueError, TypeError):
                    rejected += 1
                    continue
                selected.append(record)
                if len(selected) == requested_examples:
                    break
        finally:
            del rows
            gc.collect()
    if len(selected) != requested_examples:
        raise SFTDatasetError(
            f"source {source.split} yielded {len(selected)} valid rows, "
            f"requested {requested_examples}"
        )
    return _SourceExtraction(
        summary=SFTSourceSummary(
            split=source.split,
            available_examples=source.available_examples,
            weight=source.weight,
            requested_examples=requested_examples,
            selected_examples=len(selected),
            scanned_examples=scanned,
            rejected_examples=rejected,
        ),
        records=tuple(selected),
    )


def _atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise SFTDatasetError(f"incomplete SFT artifact exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def prepare_sft_mix(
    output_directory: str | Path,
    *,
    target_examples: int = 100_000,
    selection_seed: int = 1_337,
    loader: Callable[[str], Iterable[Mapping[str, object]]] | None = None,
) -> SFTMixManifest:
    """Stream, validate, and atomically write the pinned no-think SFT mix."""

    # The source-order policy is deterministic by revision; retain the seed in
    # the manifest so a future sampling policy can be distinguished cleanly.
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    load_source = _hub_source_loader if loader is None else loader
    allocations = allocate_sft_counts(SMOLTALK2_NO_THINK_SOURCES, target_examples)
    extractions = tuple(
        _extract_source(source, allocations[source.split], load_source)
        for source in SMOLTALK2_NO_THINK_SOURCES
    )
    records_path = output / SFT_RECORDS_FILENAME
    manifest_path = output / SFT_MANIFEST_FILENAME
    records_temporary = records_path.with_name(f".{records_path.name}.tmp")
    if records_temporary.exists():
        raise SFTDatasetError(f"incomplete SFT records exists: {records_temporary}")
    digest = sha256()
    try:
        with records_temporary.open("xb") as handle:
            for extraction in extractions:
                for record in extraction.records:
                    line = _canonical_record_line(record)
                    handle.write(line)
                    digest.update(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(records_temporary, records_path)
    except Exception:
        records_temporary.unlink(missing_ok=True)
        raise
    manifest = SFTMixManifest(
        selection_seed=selection_seed,
        target_examples=target_examples,
        selected_examples=sum(item.summary.selected_examples for item in extractions),
        records_sha256=digest.hexdigest(),
        sources=tuple(item.summary for item in extractions),
    )
    _atomic_write(manifest_path, (manifest.canonical_json() + "\n").encode())
    return manifest
