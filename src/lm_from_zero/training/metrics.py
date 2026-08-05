"""Durable JSONL, TensorBoard, and atomic Parquet training metrics."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, TextIO, cast

import polars as pl
from pydantic import BaseModel, ConfigDict, Field


class TrainingMetricsError(RuntimeError):
    """Raised when training metrics are corrupt or cannot be published."""


class OptimizerMetricRecord(BaseModel):
    """Canonical optimizer-step metrics shared by every output format."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    event: Literal["optimizer_step"] = "optimizer_step"
    recorded_at_utc: datetime
    optimizer_step: Annotated[int, Field(gt=0)]
    measurement_optimizer_steps: Annotated[int, Field(gt=0)] = 1
    loss: Annotated[float, Field(ge=0)]
    learning_rate: Annotated[float, Field(gt=0)]
    gradient_norm: Annotated[float, Field(ge=0)]
    tokens_consumed: Annotated[int, Field(gt=0)]
    elapsed_seconds: Annotated[float, Field(gt=0)]
    tokens_per_second: Annotated[float, Field(gt=0)]
    peak_cuda_memory_allocated_bytes: Annotated[int | None, Field(gt=0)] = None
    peak_cuda_memory_reserved_bytes: Annotated[int | None, Field(gt=0)] = None


class ScalarWriter(Protocol):
    """Small SummaryWriter surface used by the training loop."""

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int,
    ) -> None: ...

    def flush(self) -> None: ...

    def close(self) -> None: ...


WriterFactory = Callable[[Path, int | None], ScalarWriter]
Clock = Callable[[], float]


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )


def append_training_event(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    recorded_at_utc: datetime | None = None,
) -> dict[str, Any]:
    """Append and fsync one canonical training event."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record["recorded_at_utc"] = (
        datetime.now(UTC) if recorded_at_utc is None else recorded_at_utc
    ).isoformat()
    encoded = _canonical_json(record)
    try:
        with destination.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise TrainingMetricsError("cannot append the training JSONL log") from error
    return record


def load_optimizer_metrics(path: str | Path) -> tuple[OptimizerMetricRecord, ...]:
    """Validate canonical JSONL and retain the last record for each step."""

    source = Path(path)
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise TrainingMetricsError("cannot read the training JSONL log") from error
    latest: dict[int, OptimizerMetricRecord] = {}
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise TrainingMetricsError("training JSONL is invalid") from error
        if not isinstance(payload, dict) or line != _canonical_json(payload):
            raise TrainingMetricsError("training JSONL is not canonical")
        if payload.get("event") != "optimizer_step":
            continue
        try:
            record = OptimizerMetricRecord.model_validate(payload)
        except ValueError as error:
            raise TrainingMetricsError("optimizer-step metrics are invalid") from error
        latest[record.optimizer_step] = record
    if not latest:
        raise TrainingMetricsError("training JSONL has no optimizer-step metrics")
    return tuple(latest[step] for step in sorted(latest))


def materialize_metrics_parquet(
    jsonl_path: str | Path,
    parquet_path: str | Path,
) -> Path:
    """Atomically write a typed, deduplicated Parquet metric table."""

    records = load_optimizer_metrics(jsonl_path)
    rows = [record.model_dump(mode="python") for record in records]
    frame = pl.DataFrame(rows).sort("optimizer_step")
    destination = Path(parquet_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise TrainingMetricsError("incomplete Parquet metrics file exists")
    try:
        frame.write_parquet(
            temporary,
            compression="zstd",
            statistics=True,
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise TrainingMetricsError("cannot publish Parquet metrics") from error
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _default_writer_factory(
    directory: Path,
    purge_step: int | None,
) -> ScalarWriter:
    from torch.utils.tensorboard import SummaryWriter

    return cast(
        ScalarWriter,
        SummaryWriter(
            log_dir=str(directory),
            purge_step=purge_step,
            max_queue=10,
            flush_secs=30,
            filename_suffix=".rank-zero",
        ),
    )


class TrainingMetricSinks:
    """Coordinate durable events, live scalars, and Parquet snapshots."""

    def __init__(
        self,
        *,
        jsonl_path: str | Path,
        tensorboard_directory: str | Path | None,
        parquet_path: str | Path | None,
        resume_optimizer_step: int,
        durable_every_steps: int = 50,
        durable_every_seconds: float = 5.0,
        writer_factory: WriterFactory = _default_writer_factory,
        clock: Clock = time.monotonic,
        duration_clock: Clock = time.perf_counter,
    ) -> None:
        if resume_optimizer_step < 0:
            raise ValueError("resume optimizer step cannot be negative")
        if durable_every_steps <= 0:
            raise ValueError("durable metric step interval must be positive")
        if durable_every_seconds <= 0:
            raise ValueError("durable metric time interval must be positive")
        self.jsonl_path = Path(jsonl_path)
        self.parquet_path = None if parquet_path is None else Path(parquet_path)
        self._durable_every_steps = durable_every_steps
        self._durable_every_seconds = durable_every_seconds
        self._clock = clock
        self._duration_clock = duration_clock
        self._optimizer_steps_since_durable_sync = 0
        self._last_durable_sync = clock()
        self._metric_fsync_seconds = 0.0
        self._closed = False
        self._jsonl_handle = self._open_jsonl_handle()
        self._writer = (
            None
            if tensorboard_directory is None
            else writer_factory(
                Path(tensorboard_directory),
                resume_optimizer_step + 1 if resume_optimizer_step else None,
            )
        )

    def _open_jsonl_handle(self) -> TextIO:
        try:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            return self.jsonl_path.open("a", encoding="utf-8", newline="\n")
        except OSError as error:
            raise TrainingMetricsError(
                "cannot append the training JSONL log"
            ) from error

    def _sync_jsonl(self, *, now: float) -> None:
        started_at = self._duration_clock()
        try:
            self._jsonl_handle.flush()
            os.fsync(self._jsonl_handle.fileno())
        except OSError as error:
            raise TrainingMetricsError("cannot sync the training JSONL log") from error
        self._metric_fsync_seconds += self._duration_clock() - started_at
        self._optimizer_steps_since_durable_sync = 0
        self._last_durable_sync = now

    @property
    def metric_fsync_seconds(self) -> float:
        """Return cumulative time spent flushing and fsyncing canonical JSONL."""

        return self._metric_fsync_seconds

    def durable_sync(self) -> None:
        """Force pending canonical JSONL events to durable storage."""

        if self._closed:
            raise TrainingMetricsError("training metric sinks are closed")
        self._sync_jsonl(now=self._clock())

    def append_event(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Append one event and durably sync it on the configured schedule."""

        if self._closed:
            raise TrainingMetricsError("training metric sinks are closed")
        record = dict(payload)
        record["recorded_at_utc"] = datetime.now(UTC).isoformat()
        encoded = _canonical_json(record)
        try:
            self._jsonl_handle.write(encoded)
            self._jsonl_handle.write("\n")
        except OSError as error:
            raise TrainingMetricsError(
                "cannot append the training JSONL log"
            ) from error

        if record.get("event") == "optimizer_step":
            self._optimizer_steps_since_durable_sync += 1
        now = self._clock()
        if (
            self._optimizer_steps_since_durable_sync >= self._durable_every_steps
            or now - self._last_durable_sync >= self._durable_every_seconds
        ):
            self._sync_jsonl(now=now)
        return record

    def log_optimizer_step(self, payload: Mapping[str, Any]) -> None:
        """Persist one step and mirror its scalar fields to TensorBoard."""

        record = OptimizerMetricRecord.model_validate(
            self.append_event({"event": "optimizer_step", **payload})
        )
        if self._writer is None:
            return
        step = record.optimizer_step
        scalars = {
            "train/elapsed_seconds": record.elapsed_seconds,
            "train/gradient_norm": record.gradient_norm,
            "train/learning_rate": record.learning_rate,
            "train/loss": record.loss,
            "train/tokens_consumed": float(record.tokens_consumed),
            "train/tokens_per_second": record.tokens_per_second,
        }
        if record.peak_cuda_memory_allocated_bytes is not None:
            scalars["train/peak_cuda_memory_allocated_bytes"] = float(
                record.peak_cuda_memory_allocated_bytes
            )
        if record.peak_cuda_memory_reserved_bytes is not None:
            scalars["train/peak_cuda_memory_reserved_bytes"] = float(
                record.peak_cuda_memory_reserved_bytes
            )
        for tag, value in scalars.items():
            self._writer.add_scalar(tag, value, step)

    def snapshot(self) -> None:
        """Flush TensorBoard and atomically refresh Parquet when configured."""

        if self._closed:
            raise TrainingMetricsError("training metric sinks are closed")
        self.durable_sync()
        if self._writer is not None:
            self._writer.flush()
        if self.parquet_path is not None:
            materialize_metrics_parquet(self.jsonl_path, self.parquet_path)

    def close(self) -> None:
        """Publish the final Parquet table and close TensorBoard."""

        if self._closed:
            return
        try:
            self.snapshot()
        finally:
            if self._writer is not None:
                self._writer.close()
            self._jsonl_handle.close()
            self._closed = True

    def abort(self) -> None:
        """Close live telemetry while leaving JSONL available for rebuilding."""

        if self._closed:
            return
        try:
            self.durable_sync()
            if self._writer is not None:
                self._writer.flush()
        finally:
            if self._writer is not None:
                self._writer.close()
            self._jsonl_handle.close()
            self._closed = True
