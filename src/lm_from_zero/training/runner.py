"""Deterministic single-process dense pretraining and dry-run planning."""

from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal, Self, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from lm_from_zero.models import Olmo2Config, Olmo2ForCausalLM
from lm_from_zero.training.checkpointing import (
    CheckpointBinding,
    CheckpointCadence,
    apply_checkpoint_retention,
    create_checkpoint_binding,
    restore_checkpoint,
    save_checkpoint,
)
from lm_from_zero.training.data import (
    BatchCursor,
    CausalBatch,
    CausalBatchConfig,
    ShardBatchSource,
)
from lm_from_zero.training.optimization import (
    OptimizationConfig,
    build_adamw,
    clip_gradients,
    set_learning_rate,
)

Precision = Literal["fp32", "bf16"]
DeviceKind = Literal["cpu", "cuda"]


class TrainingRunError(RuntimeError):
    """Raised when a training run is incompatible or cannot safely continue."""


class DenseTrainingConfig(BaseModel):
    """Resolved single-process training contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-dense-training"] = "lm-from-zero-dense-training"
    format_version: Literal[1] = 1
    model: Olmo2Config
    batch: CausalBatchConfig
    optimization: OptimizationConfig
    gradient_accumulation_steps: Annotated[int, Field(gt=0)] = 1
    checkpoint_every_steps: Annotated[int, Field(gt=0)] = 250
    checkpoint_every_seconds: Annotated[float, Field(gt=0)] = 15 * 60
    keep_latest_checkpoints: Annotated[int, Field(gt=0)] = 3
    device: DeviceKind = "cpu"
    precision: Precision = "fp32"
    compile_model: bool = False
    seed: int = 1_337

    @model_validator(mode="after")
    def validate_single_process_contract(self) -> Self:
        if self.batch.rank != 0 or self.batch.world_size != 1:
            raise ValueError(
                "the single-process runner requires rank 0 and world size 1"
            )
        if self.batch.sequence_length > self.model.max_position_embeddings:
            raise ValueError("batch sequence length exceeds the model context")
        if self.device == "cpu" and self.compile_model:
            raise ValueError("CPU compilation is outside the default runner contract")
        return self

    @property
    def tokens_per_optimizer_step(self) -> int:
        """Return the fixed effective token batch."""

        return (
            self.batch.sequence_length
            * self.batch.micro_batch_size
            * self.gradient_accumulation_steps
        )

    @property
    def total_training_tokens(self) -> int:
        """Return the configured token budget."""

        return self.tokens_per_optimizer_step * self.optimization.total_steps

    def canonical_json(self) -> str:
        """Return the canonical resolved training configuration."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def config_hash(self) -> str:
        """Return the resolved training-configuration hash."""

        return sha256(self.canonical_json().encode()).hexdigest()


class DenseRunPlan(BaseModel):
    """Auditable dry-run estimate produced before allocating the model."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    training_config_sha256: str
    model_config_sha256: str
    shard_manifest_sha256: str
    tokenizer_sha256: str
    parameter_count: Annotated[int, Field(gt=0)]
    optimizer_steps: Annotated[int, Field(gt=0)]
    micro_batches_per_step: Annotated[int, Field(gt=0)]
    tokens_per_optimizer_step: Annotated[int, Field(gt=0)]
    total_training_tokens: Annotated[int, Field(gt=0)]
    estimated_training_flops: Annotated[int, Field(gt=0)]
    estimated_checkpoint_bytes: Annotated[int, Field(gt=0)]
    estimated_seconds: float | None
    checkpoint_directory: str
    device: DeviceKind
    precision: Precision
    compile_model: bool


class OptimizerStepMetrics(BaseModel):
    """Metrics from one completed optimizer step."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    optimizer_step: Annotated[int, Field(gt=0)]
    loss: Annotated[float, Field(ge=0)]
    learning_rate: Annotated[float, Field(gt=0)]
    gradient_norm: Annotated[float, Field(ge=0)]
    tokens_consumed: Annotated[int, Field(ge=0)]


class DenseTrainingResult(BaseModel):
    """Terminal state from a bounded training invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    optimizer_step: Annotated[int, Field(ge=0)]
    cursor: BatchCursor
    last_checkpoint: Path
    metrics: tuple[OptimizerStepMetrics, ...]


def seed_training(seed: int, *, cuda: bool) -> None:
    """Seed Python, NumPy, Torch CPU, and optionally every CUDA device."""

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def _validate_source_binding(
    config: DenseTrainingConfig, source: ShardBatchSource
) -> None:
    if source.config != config.batch:
        raise TrainingRunError("batch source configuration does not match the run")
    if source.build.tokenizer_hash != config.model.tokenizer_hash:
        raise TrainingRunError("model tokenizer does not match the shard build")
    if source.build.tokenizer_vocab_size != config.model.vocab_size:
        raise TrainingRunError("model vocabulary size does not match the shard build")


def create_dense_run_plan(
    config: DenseTrainingConfig,
    source: ShardBatchSource,
    checkpoint_directory: str | Path,
    *,
    estimated_tokens_per_second: float | None = None,
) -> DenseRunPlan:
    """Validate inputs and estimate compute, disk, and optional wall time."""

    _validate_source_binding(config, source)
    if estimated_tokens_per_second is not None and estimated_tokens_per_second <= 0:
        raise ValueError("estimated token throughput must be positive")
    parameters = config.model.parameter_breakdown().total
    forward_flops = config.model.forward_flops(
        config.batch.sequence_length
    ).total_flops_per_token
    estimated_seconds = (
        None
        if estimated_tokens_per_second is None
        else config.total_training_tokens / estimated_tokens_per_second
    )
    return DenseRunPlan(
        training_config_sha256=config.config_hash,
        model_config_sha256=config.model.config_hash,
        shard_manifest_sha256=source.build_manifest_sha256,
        tokenizer_sha256=source.build.tokenizer_hash,
        parameter_count=parameters,
        optimizer_steps=config.optimization.total_steps,
        micro_batches_per_step=config.gradient_accumulation_steps,
        tokens_per_optimizer_step=config.tokens_per_optimizer_step,
        total_training_tokens=config.total_training_tokens,
        estimated_training_flops=(3 * forward_flops * config.total_training_tokens),
        estimated_checkpoint_bytes=12 * parameters,
        estimated_seconds=estimated_seconds,
        checkpoint_directory=str(Path(checkpoint_directory)),
        device=config.device,
        precision=config.precision,
        compile_model=config.compile_model,
    )


def _autocast_context(config: DenseTrainingConfig) -> torch.autocast:
    return torch.autocast(
        device_type=config.device,
        dtype=torch.bfloat16,
        enabled=config.precision == "bf16",
    )


def train_accumulated_step(
    model: Olmo2ForCausalLM,
    optimizer: torch.optim.Optimizer,
    batches: Sequence[CausalBatch],
    config: DenseTrainingConfig,
    *,
    zero_based_step: int,
) -> OptimizerStepMetrics:
    """Apply one optimizer update from equal-size causal microbatches."""

    if len(batches) != config.gradient_accumulation_steps:
        raise TrainingRunError("microbatch count does not match accumulation policy")
    if zero_based_step < 0:
        raise ValueError("optimizer step cannot be negative")
    device = torch.device(config.device)
    optimizer.zero_grad(set_to_none=True)
    learning_rate = set_learning_rate(
        optimizer,
        config.optimization,
        zero_based_step,
    )
    detached_losses: list[Tensor] = []
    final_cursor = batches[-1].cursor_after
    for batch in batches:
        input_ids = batch.input_ids.to(device)
        labels = batch.labels.to(device)
        with _autocast_context(config):
            loss = cast(Tensor, model(input_ids, labels=labels).loss)
        if not bool(torch.isfinite(loss)):
            raise TrainingRunError("training loss is not finite")
        torch.autograd.backward(loss / config.gradient_accumulation_steps)
        detached_losses.append(loss.detach().float())
    gradient_norm = clip_gradients(model, config.optimization.gradient_clip_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise TrainingRunError("gradient norm is not finite")
    optimizer.step()
    mean_loss = torch.stack(detached_losses).mean()
    return OptimizerStepMetrics(
        optimizer_step=zero_based_step + 1,
        loss=float(mean_loss),
        learning_rate=learning_rate,
        gradient_norm=float(gradient_norm),
        tokens_consumed=final_cursor.tokens_consumed,
    )


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record["recorded_at_utc"] = datetime.now(UTC).isoformat()
    encoded = json.dumps(
        record,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(encoded)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


class DenseTrainer:
    """Run bounded dense pretraining with exact checkpoint resume."""

    def __init__(
        self,
        *,
        model: Olmo2ForCausalLM,
        source: ShardBatchSource,
        config: DenseTrainingConfig,
        checkpoint_directory: str | Path,
        repository: str | Path,
        jsonl_log: str | Path,
    ) -> None:
        _validate_source_binding(config, source)
        if model.config != config.model:
            raise TrainingRunError(
                "model instance does not match the run configuration"
            )
        if config.device == "cuda" and not torch.cuda.is_available():
            raise TrainingRunError(
                "CUDA training was requested but CUDA is unavailable"
            )
        self.model = model.to(torch.device(config.device))
        self.forward_model = (
            cast(Olmo2ForCausalLM, torch.compile(self.model))
            if config.compile_model
            else self.model
        )
        self.source = source
        self.config = config
        self.checkpoint_directory = Path(checkpoint_directory)
        self.jsonl_log = Path(jsonl_log)
        self.optimizer, _ = build_adamw(self.model, config.optimization)
        self.binding: CheckpointBinding = create_checkpoint_binding(
            architecture="olmo2",
            resolved_model_config=config.model.model_dump(mode="json"),
            tokenizer_sha256=source.build.tokenizer_hash,
            shard_manifest_sha256=source.build_manifest_sha256,
            rank=config.batch.rank,
            world_size=config.batch.world_size,
            repository=repository,
        )

    def _restore(
        self, resume_from: str | Path | None
    ) -> tuple[int, BatchCursor, Path | None]:
        if resume_from is None:
            seed_training(self.config.seed, cuda=self.config.device == "cuda")
            return 0, self.source.initial_cursor(), None
        restored = restore_checkpoint(
            resume_from,
            model=self.model,
            optimizer=self.optimizer,
            expected_binding=self.binding,
        )
        optimizer_step = restored.manifest.progress.optimizer_step
        expected_scheduler = {
            "next_optimizer_step": optimizer_step,
            "training_config_sha256": self.config.config_hash,
        }
        if restored.scheduler_state != expected_scheduler:
            raise TrainingRunError("checkpoint scheduler state is incompatible")
        return optimizer_step, restored.manifest.cursor, restored.path

    def _save(
        self,
        *,
        optimizer_step: int,
        cursor: BatchCursor,
        parent: Path | None,
    ) -> Path:
        checkpoint = save_checkpoint(
            self.checkpoint_directory,
            model=self.model,
            optimizer=self.optimizer,
            cursor=cursor,
            binding=self.binding,
            optimizer_step=optimizer_step,
            scheduler_step=optimizer_step,
            scheduler_state={
                "next_optimizer_step": optimizer_step,
                "training_config_sha256": self.config.config_hash,
            },
            parent_checkpoint=parent,
        )
        apply_checkpoint_retention(
            self.checkpoint_directory,
            keep_latest=self.config.keep_latest_checkpoints,
        )
        return checkpoint

    def run(
        self,
        *,
        resume_from: str | Path | None = None,
        stop_after_optimizer_step: int | None = None,
    ) -> DenseTrainingResult:
        """Train to the configured step budget and publish a final checkpoint."""

        optimizer_step, cursor, parent = self._restore(resume_from)
        if optimizer_step > self.config.optimization.total_steps:
            raise TrainingRunError("checkpoint is beyond the configured step budget")
        target_step = (
            self.config.optimization.total_steps
            if stop_after_optimizer_step is None
            else stop_after_optimizer_step
        )
        if not optimizer_step < target_step <= self.config.optimization.total_steps:
            raise TrainingRunError("bounded stop must follow resume and fit the budget")
        _append_jsonl(
            self.jsonl_log,
            {
                "event": "run_start" if resume_from is None else "run_resume",
                "optimizer_step": optimizer_step,
                "training_config": self.config.model_dump(mode="json"),
                "training_config_sha256": self.config.config_hash,
            },
        )
        cadence = CheckpointCadence(
            last_saved_time_seconds=time.monotonic(),
            step_interval=self.config.checkpoint_every_steps,
            time_interval_seconds=self.config.checkpoint_every_seconds,
            last_saved_step=optimizer_step if parent is not None else None,
        )
        metrics: list[OptimizerStepMetrics] = []
        self.forward_model.train()
        while optimizer_step < target_step:
            batches: list[CausalBatch] = []
            for _ in range(self.config.gradient_accumulation_steps):
                batch = self.source.next_batch(cursor)
                batches.append(batch)
                cursor = batch.cursor_after
            step_metrics = train_accumulated_step(
                self.forward_model,
                self.optimizer,
                batches,
                self.config,
                zero_based_step=optimizer_step,
            )
            optimizer_step += 1
            metrics.append(step_metrics)
            _append_jsonl(
                self.jsonl_log,
                {"event": "optimizer_step", **step_metrics.model_dump(mode="json")},
            )
            now = time.monotonic()
            reasons = cadence.due_reasons(optimizer_step, now)
            if reasons:
                parent = self._save(
                    optimizer_step=optimizer_step,
                    cursor=cursor,
                    parent=parent,
                )
                cadence.mark_saved(optimizer_step, now)
                _append_jsonl(
                    self.jsonl_log,
                    {
                        "checkpoint": parent.name,
                        "event": "checkpoint",
                        "optimizer_step": optimizer_step,
                        "reasons": sorted(reasons),
                    },
                )

        parent = self._save(
            optimizer_step=optimizer_step,
            cursor=cursor,
            parent=parent,
        )
        _append_jsonl(
            self.jsonl_log,
            {
                "checkpoint": parent.name,
                "event": (
                    "run_complete"
                    if optimizer_step == self.config.optimization.total_steps
                    else "run_stopped"
                ),
                "optimizer_step": optimizer_step,
                "tokens_consumed": cursor.tokens_consumed,
            },
        )
        return DenseTrainingResult(
            optimizer_step=optimizer_step,
            cursor=cursor,
            last_checkpoint=parent,
            metrics=tuple(metrics),
        )


def optimizer_steps_for_token_budget(
    target_tokens: int,
    *,
    sequence_length: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Round a token budget upward to complete optimizer steps."""

    values = (
        target_tokens,
        sequence_length,
        micro_batch_size,
        gradient_accumulation_steps,
    )
    if any(value <= 0 for value in values):
        raise ValueError("token budget and batch dimensions must be positive")
    tokens_per_step = sequence_length * micro_batch_size * gradient_accumulation_steps
    return math.ceil(target_tokens / tokens_per_step)
