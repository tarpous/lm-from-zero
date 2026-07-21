"""Deterministic shard-window batches with exact serializable cursors."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from lm_from_zero.data import (
    DataValidationError,
    Split,
    load_manifest,
    validate_shard,
)
from lm_from_zero.sharding import ShardBuildManifest, validate_shard_build

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class CausalBatchConfig(BaseModel):
    """Static ordering and batch-shape contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    split: Split = "train"
    sequence_length: Annotated[int, Field(gt=1)] = 1_024
    micro_batch_size: Annotated[int, Field(gt=0)] = 8
    seed: int = 1_337
    rank: Annotated[int, Field(ge=0)] = 0
    world_size: Annotated[int, Field(gt=0)] = 1
    shuffle: bool = True

    @model_validator(mode="after")
    def validate_rank(self) -> Self:
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        return self


class BatchCursor(BaseModel):
    """Exact next rank-local window and immutable consumption totals."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-batch-cursor"] = "lm-from-zero-batch-cursor"
    format_version: Literal[1] = 1
    build_manifest_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    tokenizer_hash: Annotated[str, Field(pattern=SHA256_PATTERN)]
    split: Split
    sequence_length: Annotated[int, Field(gt=1)]
    seed: int
    rank: Annotated[int, Field(ge=0)]
    world_size: Annotated[int, Field(gt=0)]
    shuffle: bool
    epoch: Annotated[int, Field(ge=0)] = 0
    next_local_window: Annotated[int, Field(ge=0)] = 0
    sequences_consumed: Annotated[int, Field(ge=0)] = 0
    tokens_consumed: Annotated[int, Field(ge=0)] = 0

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        if self.tokens_consumed != self.sequences_consumed * self.sequence_length:
            raise ValueError("cursor token and sequence totals disagree")
        return self


@dataclass(frozen=True, slots=True)
class CausalBatch:
    """CPU token IDs and the exact cursor transition that produced them."""

    input_ids: Tensor
    labels: Tensor
    cursor_before: BatchCursor
    cursor_after: BatchCursor


@dataclass(frozen=True, slots=True)
class _Window:
    shard_index: int
    start: int

    @property
    def stable_id(self) -> str:
        return f"{self.shard_index}:{self.start}"


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ShardBatchSource:
    """Validated memory-mapped token windows with deterministic rank order."""

    def __init__(
        self,
        build_manifest_path: str | Path,
        config: CausalBatchConfig,
    ) -> None:
        self.config = config
        self.build_path = Path(build_manifest_path)
        self.build: ShardBuildManifest = validate_shard_build(self.build_path)
        if config.sequence_length > self.build.max_tokens_per_shard:
            raise DataValidationError("sequence length exceeds the shard token limit")
        self.build_manifest_sha256 = _file_sha256(self.build_path)
        split_stats = next(
            item for item in self.build.splits if item.split == config.split
        )
        if not split_stats.manifest_files:
            raise DataValidationError(f"the {config.split} split contains no shards")

        shard_directory = self.build_path.parent / self.build.shard_directory
        arrays: list[np.ndarray] = []
        windows: list[_Window] = []
        for shard_index, filename in enumerate(split_stats.manifest_files):
            manifest_path = shard_directory / filename
            manifest = load_manifest(manifest_path)
            array = validate_shard(
                manifest_path,
                expected_tokenizer_hash=self.build.tokenizer_hash,
                expected_vocab_size=self.build.tokenizer_vocab_size,
            )
            arrays.append(array)
            for start in range(
                0,
                manifest.token_count - config.sequence_length + 1,
                config.sequence_length,
            ):
                windows.append(_Window(shard_index=shard_index, start=start))
        if not windows:
            raise DataValidationError("split does not contain one complete sequence")
        self._arrays = tuple(arrays)
        self._windows = tuple(windows)
        self._rank_order_cache: dict[int, tuple[_Window, ...]] = {}
        if not self._rank_windows(0):
            raise DataValidationError(
                "rank has no windows under the configured world size"
            )

    @property
    def window_count(self) -> int:
        """Return complete non-overlapping windows across every rank."""

        return len(self._windows)

    def _ordered_windows(self, epoch: int) -> tuple[_Window, ...]:
        if not self.config.shuffle:
            return self._windows

        def sort_key(window: _Window) -> tuple[bytes, str]:
            payload = f"{self.config.seed}:{epoch}:{window.stable_id}".encode("ascii")
            return sha256(payload).digest(), window.stable_id

        return tuple(sorted(self._windows, key=sort_key))

    def _rank_windows(self, epoch: int) -> tuple[_Window, ...]:
        cached = self._rank_order_cache.get(epoch)
        if cached is None:
            ordered = self._ordered_windows(epoch)
            cached = ordered[self.config.rank :: self.config.world_size]
            self._rank_order_cache[epoch] = cached
        return cached

    def rank_window_ids(self, epoch: int) -> tuple[str, ...]:
        """Expose stable IDs for rank-overlap and ordering verification."""

        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        return tuple(window.stable_id for window in self._rank_windows(epoch))

    def initial_cursor(self) -> BatchCursor:
        """Return a cursor bound to this exact build and ordering policy."""

        return BatchCursor(
            build_manifest_sha256=self.build_manifest_sha256,
            tokenizer_hash=self.build.tokenizer_hash,
            split=self.config.split,
            sequence_length=self.config.sequence_length,
            seed=self.config.seed,
            rank=self.config.rank,
            world_size=self.config.world_size,
            shuffle=self.config.shuffle,
        )

    def _validate_cursor(self, cursor: BatchCursor) -> None:
        expected = self.initial_cursor()
        binding_fields = (
            "build_manifest_sha256",
            "tokenizer_hash",
            "split",
            "sequence_length",
            "seed",
            "rank",
            "world_size",
            "shuffle",
        )
        for field_name in binding_fields:
            if getattr(cursor, field_name) != getattr(expected, field_name):
                raise DataValidationError(
                    f"cursor {field_name} does not match the batch source"
                )
        if cursor.next_local_window > len(self._rank_windows(cursor.epoch)):
            raise DataValidationError("cursor points beyond the rank-local epoch")

    def next_batch(self, cursor: BatchCursor | None = None) -> CausalBatch:
        """Read the next batch and return its exact immutable cursor transition."""

        before = self.initial_cursor() if cursor is None else cursor
        self._validate_cursor(before)
        epoch = before.epoch
        next_local = before.next_local_window
        rows: list[Tensor] = []
        while len(rows) < self.config.micro_batch_size:
            rank_windows = self._rank_windows(epoch)
            if next_local == len(rank_windows):
                epoch += 1
                next_local = 0
                rank_windows = self._rank_windows(epoch)
            window = rank_windows[next_local]
            array = self._arrays[window.shard_index]
            values = np.array(
                array[window.start : window.start + self.config.sequence_length],
                dtype=np.int64,
                copy=True,
            )
            rows.append(torch.from_numpy(values))
            next_local += 1

        input_ids = torch.stack(rows)
        sequences = before.sequences_consumed + self.config.micro_batch_size
        after = before.model_copy(
            update={
                "epoch": epoch,
                "next_local_window": next_local,
                "sequences_consumed": sequences,
                "tokens_consumed": sequences * self.config.sequence_length,
            }
        )
        return CausalBatch(
            input_ids=input_ids,
            labels=input_ids.clone(),
            cursor_before=before,
            cursor_after=after,
        )
