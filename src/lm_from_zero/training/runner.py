"""Deterministic single-process and DDP objective-adapted pretraining."""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Sequence
from contextlib import nullcontext
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Self, cast

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from lm_from_zero.models import (
    Mamba2Config,
    Mamba2ForCausalLM,
    MaskedDiffusionConfig,
    MaskedDiffusionForMaskedLM,
    Olmo2Config,
    Olmo2ForCausalLM,
    base_pretraining_eligible_mask,
    corrupt_for_diffusion,
)
from lm_from_zero.training.checkpointing import (
    CheckpointBinding,
    CheckpointCadence,
    apply_checkpoint_retention,
    capture_rng_state,
    create_checkpoint_binding,
    restore_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from lm_from_zero.training.data import (
    BatchCursor,
    CausalBatch,
    CausalBatchConfig,
    ShardBatchSource,
)
from lm_from_zero.training.distributed import DistributedContext, DistributedError
from lm_from_zero.training.metrics import TrainingMetricSinks
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
    """Resolved process-local training contract."""

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
    def validate_training_contract(self) -> Self:
        if self.batch.sequence_length > self.model.max_position_embeddings:
            raise ValueError("batch sequence length exceeds the model context")
        if self.device == "cpu" and self.compile_model:
            raise ValueError("CPU compilation is outside the default runner contract")
        return self

    @property
    def tokens_per_optimizer_step(self) -> int:
        """Return the fixed global effective token batch."""

        return self.local_tokens_per_optimizer_step * self.batch.world_size

    @property
    def local_tokens_per_optimizer_step(self) -> int:
        """Return the fixed token batch processed by this rank."""

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
        """Return a rank-independent canonical training configuration."""

        payload = self.model_dump(mode="json")
        batch = cast(dict[str, Any], payload["batch"])
        batch["rank"] = 0
        return json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def config_hash(self) -> str:
        """Return the resolved training-configuration hash."""

        return sha256(self.canonical_json().encode()).hexdigest()


class Mamba2TrainingConfig(BaseModel):
    """Resolved process-local Mamba-2 training contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-mamba2-training"] = "lm-from-zero-mamba2-training"
    format_version: Literal[1] = 1
    model: Mamba2Config
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
    def validate_training_contract(self) -> Self:
        if self.batch.sequence_length > self.model.max_position_embeddings:
            raise ValueError("batch sequence length exceeds the model context")
        if self.device == "cpu" and self.compile_model:
            raise ValueError("CPU compilation is outside the default runner contract")
        return self

    @property
    def tokens_per_optimizer_step(self) -> int:
        """Return the fixed global effective token batch."""

        return self.local_tokens_per_optimizer_step * self.batch.world_size

    @property
    def local_tokens_per_optimizer_step(self) -> int:
        """Return the fixed token batch processed by this rank."""

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
        """Return a rank-independent canonical training configuration."""

        payload = self.model_dump(mode="json")
        batch = cast(dict[str, Any], payload["batch"])
        batch["rank"] = 0
        return json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def config_hash(self) -> str:
        """Return the resolved training-configuration hash."""

        return sha256(self.canonical_json().encode()).hexdigest()


class DiffusionTrainingConfig(BaseModel):
    """Resolved process-local masked-diffusion training contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-diffusion-training"] = (
        "lm-from-zero-diffusion-training"
    )
    format_version: Literal[1] = 1
    model: MaskedDiffusionConfig
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
    def validate_training_contract(self) -> Self:
        if self.batch.sequence_length > self.model.max_position_embeddings:
            raise ValueError("batch sequence length exceeds the model context")
        if self.device == "cpu" and self.compile_model:
            raise ValueError("CPU compilation is outside the default runner contract")
        return self

    @property
    def tokens_per_optimizer_step(self) -> int:
        """Return the fixed global effective token batch."""

        return self.local_tokens_per_optimizer_step * self.batch.world_size

    @property
    def local_tokens_per_optimizer_step(self) -> int:
        """Return the fixed token batch processed by this rank."""

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
        """Return a rank-independent canonical training configuration."""

        payload = self.model_dump(mode="json")
        batch = cast(dict[str, Any], payload["batch"])
        batch["rank"] = 0
        return json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def config_hash(self) -> str:
        """Return the resolved training-configuration hash."""

        return sha256(self.canonical_json().encode()).hexdigest()


TrainingConfig = DenseTrainingConfig | Mamba2TrainingConfig | DiffusionTrainingConfig
PretrainingModel = Olmo2ForCausalLM | Mamba2ForCausalLM | MaskedDiffusionForMaskedLM


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
    world_size: Annotated[int, Field(gt=0)]
    local_tokens_per_optimizer_step: Annotated[int, Field(gt=0)]
    tokens_per_optimizer_step: Annotated[int, Field(gt=0)]
    total_training_tokens: Annotated[int, Field(gt=0)]
    estimated_training_flops: Annotated[int, Field(gt=0)]
    reference_training_flops: Annotated[int | None, Field(gt=0)] = None
    training_flop_ratio: Annotated[float | None, Field(gt=0)] = None
    estimated_checkpoint_bytes: Annotated[int, Field(gt=0)]
    estimated_seconds: float | None
    checkpoint_directory: str
    jsonl_log: str | None
    tensorboard_directory: str | None
    parquet_log: str | None
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
    elapsed_seconds: Annotated[float, Field(gt=0)]
    tokens_per_second: Annotated[float, Field(gt=0)]
    peak_cuda_memory_allocated_bytes: Annotated[int | None, Field(gt=0)] = None
    peak_cuda_memory_reserved_bytes: Annotated[int | None, Field(gt=0)] = None


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


def _validate_source_binding(config: TrainingConfig, source: ShardBatchSource) -> None:
    if source.config != config.batch:
        raise TrainingRunError("batch source configuration does not match the run")
    if source.build.tokenizer_hash != config.model.tokenizer_hash:
        raise TrainingRunError("model tokenizer does not match the shard build")
    if source.build.tokenizer_vocab_size != config.model.vocab_size:
        raise TrainingRunError("model vocabulary size does not match the shard build")


def create_dense_run_plan(
    config: TrainingConfig,
    source: ShardBatchSource,
    checkpoint_directory: str | Path,
    *,
    estimated_tokens_per_second: float | None = None,
    jsonl_log: str | Path | None = None,
    tensorboard_directory: str | Path | None = None,
    parquet_log: str | Path | None = None,
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
        world_size=config.batch.world_size,
        local_tokens_per_optimizer_step=config.local_tokens_per_optimizer_step,
        tokens_per_optimizer_step=config.tokens_per_optimizer_step,
        total_training_tokens=config.total_training_tokens,
        estimated_training_flops=(3 * forward_flops * config.total_training_tokens),
        estimated_checkpoint_bytes=12 * parameters,
        estimated_seconds=estimated_seconds,
        checkpoint_directory=str(Path(checkpoint_directory)),
        jsonl_log=None if jsonl_log is None else str(Path(jsonl_log)),
        tensorboard_directory=(
            None if tensorboard_directory is None else str(Path(tensorboard_directory))
        ),
        parquet_log=None if parquet_log is None else str(Path(parquet_log)),
        device=config.device,
        precision=config.precision,
        compile_model=config.compile_model,
    )


def create_mamba2_run_plan(
    config: Mamba2TrainingConfig,
    source: ShardBatchSource,
    checkpoint_directory: str | Path,
    *,
    estimated_tokens_per_second: float | None = None,
    jsonl_log: str | Path | None = None,
    tensorboard_directory: str | Path | None = None,
    parquet_log: str | Path | None = None,
    reference_training_flops: int | None = None,
) -> DenseRunPlan:
    """Validate and estimate a Mamba-2 run with the common causal policy."""

    plan = create_dense_run_plan(
        config,
        source,
        checkpoint_directory,
        estimated_tokens_per_second=estimated_tokens_per_second,
        jsonl_log=jsonl_log,
        tensorboard_directory=tensorboard_directory,
        parquet_log=parquet_log,
    )
    if reference_training_flops is None:
        return plan
    if reference_training_flops <= 0:
        raise ValueError("reference training FLOPs must be positive")
    return plan.model_copy(
        update={
            "reference_training_flops": reference_training_flops,
            "training_flop_ratio": (
                plan.estimated_training_flops / reference_training_flops
            ),
        }
    )


def create_diffusion_run_plan(
    config: DiffusionTrainingConfig,
    source: ShardBatchSource,
    checkpoint_directory: str | Path,
    *,
    estimated_tokens_per_second: float | None = None,
    jsonl_log: str | Path | None = None,
    tensorboard_directory: str | Path | None = None,
    parquet_log: str | Path | None = None,
    reference_training_flops: int | None = None,
) -> DenseRunPlan:
    """Validate and estimate diffusion pretraining under matched FLOPs."""

    plan = create_dense_run_plan(
        config,
        source,
        checkpoint_directory,
        estimated_tokens_per_second=estimated_tokens_per_second,
        jsonl_log=jsonl_log,
        tensorboard_directory=tensorboard_directory,
        parquet_log=parquet_log,
    )
    if reference_training_flops is None:
        return plan
    if reference_training_flops <= 0:
        raise ValueError("reference training FLOPs must be positive")
    return plan.model_copy(
        update={
            "reference_training_flops": reference_training_flops,
            "training_flop_ratio": (
                plan.estimated_training_flops / reference_training_flops
            ),
        }
    )


def _autocast_context(config: TrainingConfig) -> torch.autocast:
    return torch.autocast(
        device_type=config.device,
        dtype=torch.bfloat16,
        enabled=config.precision == "bf16",
    )


class TrainingObjective(Protocol):
    """Prepare an architecture-specific batch and return one scalar loss."""

    def loss(
        self, model: nn.Module, batch: CausalBatch, device: torch.device
    ) -> Tensor:
        """Run the model objective for one local microbatch."""


class CausalTrainingObjective:
    """Shifted next-token objective for dense and Mamba-2 models."""

    def loss(
        self, model: nn.Module, batch: CausalBatch, device: torch.device
    ) -> Tensor:
        input_ids = batch.input_ids.to(device)
        labels = batch.labels.to(device)
        loss = cast(Any, model)(input_ids, labels=labels).loss
        if not isinstance(loss, Tensor):
            raise TrainingRunError("causal model did not return a training loss")
        return loss


class DiffusionTrainingObjective:
    """Continuous-time masking adapter for the bidirectional denoiser."""

    def __init__(self, config: MaskedDiffusionConfig) -> None:
        self.config = config

    def loss(
        self, model: nn.Module, batch: CausalBatch, device: torch.device
    ) -> Tensor:
        original = batch.input_ids.to(device)
        attention_mask = original != self.config.pad_token_id
        eligible = base_pretraining_eligible_mask(
            original,
            attention_mask,
            pad_token_id=self.config.pad_token_id,
            bos_token_id=self.config.bos_token_id,
        )
        corrupted = corrupt_for_diffusion(
            original,
            eligible,
            mask_token_id=self.config.mask_token_id,
            epsilon=self.config.corruption_epsilon,
        )
        loss = cast(Any, model)(
            corrupted.input_ids,
            attention_mask=attention_mask,
            labels=corrupted.labels,
            eligible_mask=corrupted.eligible_mask,
            time=corrupted.time,
        ).loss
        if not isinstance(loss, Tensor):
            raise TrainingRunError("diffusion model did not return a training loss")
        return loss


def training_objective(config: TrainingConfig) -> TrainingObjective:
    """Resolve the objective adapter without branching through model internals."""

    if isinstance(config, DiffusionTrainingConfig):
        return DiffusionTrainingObjective(config.model)
    return CausalTrainingObjective()


def train_accumulated_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Sequence[CausalBatch],
    config: TrainingConfig,
    *,
    zero_based_step: int,
    distributed: DistributedContext | None = None,
    objective: TrainingObjective | None = None,
) -> OptimizerStepMetrics:
    """Apply one globally reduced update from equal-size local microbatches."""

    if len(batches) != config.gradient_accumulation_steps:
        raise TrainingRunError("microbatch count does not match accumulation policy")
    if zero_based_step < 0:
        raise ValueError("optimizer step cannot be negative")
    process = DistributedContext.current() if distributed is None else distributed
    try:
        process.validate_topology(
            rank=config.batch.rank,
            world_size=config.batch.world_size,
        )
    except DistributedError as error:
        raise TrainingRunError(str(error)) from error
    device = process.torch_device(config.device)
    if config.device == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    learning_rate = set_learning_rate(
        optimizer,
        config.optimization,
        zero_based_step,
    )
    detached_losses: list[Tensor] = []
    final_cursor = batches[-1].cursor_after
    resolved_objective = training_objective(config) if objective is None else objective
    for batch_index, batch in enumerate(batches):
        synchronization = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel)
            and batch_index + 1 < len(batches)
            else nullcontext()
        )
        with synchronization:
            with _autocast_context(config):
                loss = resolved_objective.loss(model, batch, device)
            if not bool(torch.isfinite(loss)):
                raise TrainingRunError("training loss is not finite")
            torch.autograd.backward(loss / config.gradient_accumulation_steps)
        detached_losses.append(loss.detach().float())
    gradient_norm = clip_gradients(model, config.optimization.gradient_clip_norm)
    if not bool(torch.isfinite(gradient_norm)):
        raise TrainingRunError("gradient norm is not finite")
    optimizer.step()
    if config.device == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = process.reduce_float(
        time.perf_counter() - started,
        reduction="max",
        device=device,
    )
    mean_loss = process.reduce_tensor_mean(torch.stack(detached_losses).mean())
    reduced_gradient_norm = process.reduce_float(
        float(gradient_norm),
        reduction="mean",
        device=device,
    )
    peak_allocated = (
        torch.cuda.max_memory_allocated(device) if config.device == "cuda" else None
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(device) if config.device == "cuda" else None
    )
    if peak_allocated is not None:
        peak_allocated = round(
            process.reduce_float(
                float(peak_allocated),
                reduction="max",
                device=device,
            )
        )
    if peak_reserved is not None:
        peak_reserved = round(
            process.reduce_float(
                float(peak_reserved),
                reduction="max",
                device=device,
            )
        )
    return OptimizerStepMetrics(
        optimizer_step=zero_based_step + 1,
        loss=float(mean_loss),
        learning_rate=learning_rate,
        gradient_norm=reduced_gradient_norm,
        tokens_consumed=final_cursor.tokens_consumed * process.world_size,
        elapsed_seconds=elapsed_seconds,
        tokens_per_second=config.tokens_per_optimizer_step / elapsed_seconds,
        peak_cuda_memory_allocated_bytes=peak_allocated,
        peak_cuda_memory_reserved_bytes=peak_reserved,
    )


def _portable_rng_state() -> dict[str, Any]:
    """Encode RNG tensors as lists for reliable object collectives."""

    state = capture_rng_state()
    numpy_state = cast(dict[str, Any], state["numpy"])
    return {
        "numpy": {
            **numpy_state,
            "keys": cast(Tensor, numpy_state["keys"]).tolist(),
        },
        "python": state["python"],
        "torch_cpu": cast(Tensor, state["torch_cpu"]).tolist(),
        "torch_cuda": [
            item.tolist() for item in cast(list[Tensor], state["torch_cuda"])
        ],
    }


def _restore_portable_rng_state(value: object) -> None:
    """Reconstruct tensor RNG fields and delegate strict validation."""

    if not isinstance(value, dict):
        raise TrainingRunError("checkpoint rank RNG state is invalid")
    try:
        numpy_state = cast(dict[str, Any], value["numpy"])
        restored = {
            "numpy": {
                **numpy_state,
                "keys": torch.tensor(numpy_state["keys"], dtype=torch.uint32),
            },
            "python": value["python"],
            "torch_cpu": torch.tensor(value["torch_cpu"], dtype=torch.uint8),
            "torch_cuda": [
                torch.tensor(item, dtype=torch.uint8)
                for item in cast(list[object], value["torch_cuda"])
            ],
        }
        restore_rng_state(restored)
    except (KeyError, TypeError, ValueError) as error:
        raise TrainingRunError("checkpoint rank RNG state is invalid") from error


class CausalTrainer:
    """Run bounded objective-adapted pretraining with exact checkpoint resume."""

    def __init__(
        self,
        *,
        model: PretrainingModel,
        source: ShardBatchSource,
        config: TrainingConfig,
        checkpoint_directory: str | Path,
        repository: str | Path,
        jsonl_log: str | Path,
        tensorboard_directory: str | Path | None = None,
        parquet_log: str | Path | None = None,
        distributed: DistributedContext | None = None,
    ) -> None:
        _validate_source_binding(config, source)
        self.distributed = (
            DistributedContext.current() if distributed is None else distributed
        )
        try:
            self.distributed.validate_topology(
                rank=config.batch.rank,
                world_size=config.batch.world_size,
            )
        except DistributedError as error:
            raise TrainingRunError(str(error)) from error
        if model.config != config.model:
            raise TrainingRunError(
                "model instance does not match the run configuration"
            )
        if config.device == "cuda" and not torch.cuda.is_available():
            raise TrainingRunError(
                "CUDA training was requested but CUDA is unavailable"
            )
        device = self.distributed.torch_device(config.device)
        self.model = model.to(device)
        compiled_model: nn.Module = (
            cast(nn.Module, torch.compile(self.model))
            if config.compile_model
            else self.model
        )
        self.forward_model = (
            DistributedDataParallel(
                compiled_model,
                device_ids=(
                    [self.distributed.local_rank] if config.device == "cuda" else None
                ),
                output_device=(
                    self.distributed.local_rank if config.device == "cuda" else None
                ),
            )
            if self.distributed.enabled
            else compiled_model
        )
        self.source = source
        self.config = config
        self.objective = training_objective(config)
        self.checkpoint_directory = Path(checkpoint_directory)
        self.jsonl_log = Path(jsonl_log)
        self.tensorboard_directory = (
            None if tensorboard_directory is None else Path(tensorboard_directory)
        )
        self.parquet_log = None if parquet_log is None else Path(parquet_log)
        self.optimizer, _ = build_adamw(self.model, config.optimization)
        self.binding: CheckpointBinding = create_checkpoint_binding(
            architecture=config.model.architecture,
            resolved_model_config=config.model.model_dump(mode="json"),
            tokenizer_sha256=source.build.tokenizer_hash,
            shard_manifest_sha256=source.build_manifest_sha256,
            rank=0,
            world_size=config.batch.world_size,
            repository=repository,
        )

    def _base_scheduler_state(self, optimizer_step: int) -> dict[str, Any]:
        return {
            "next_optimizer_step": optimizer_step,
            "training_config_sha256": self.config.config_hash,
        }

    def _rank_recovery_state(self, cursor: BatchCursor) -> dict[str, Any]:
        return {
            "cursor": cursor.model_dump(mode="json"),
            "rank": self.distributed.rank,
            "rng_state": _portable_rng_state(),
        }

    def _restore_distributed_state(
        self,
        scheduler_state: dict[str, Any],
        optimizer_step: int,
    ) -> BatchCursor:
        expected = self._base_scheduler_state(optimizer_step)
        if self.distributed.world_size == 1:
            if scheduler_state != expected:
                raise TrainingRunError("checkpoint scheduler state is incompatible")
            raise AssertionError("single-process cursor must come from the manifest")
        if set(scheduler_state) != {*expected, "rank_states"} or any(
            scheduler_state[key] != value for key, value in expected.items()
        ):
            raise TrainingRunError("checkpoint scheduler state is incompatible")
        raw_states = scheduler_state["rank_states"]
        if (
            not isinstance(raw_states, list)
            or len(raw_states) != self.distributed.world_size
        ):
            raise TrainingRunError("checkpoint rank recovery state is incomplete")
        cursors: dict[int, BatchCursor] = {}
        rng_states: dict[int, object] = {}
        for raw_state in raw_states:
            if not isinstance(raw_state, dict):
                raise TrainingRunError("checkpoint rank recovery state is invalid")
            if set(raw_state) != {"cursor", "rank", "rng_state"}:
                raise TrainingRunError("checkpoint rank recovery state is invalid")
            rank = raw_state["rank"]
            if not isinstance(rank, int) or rank in cursors:
                raise TrainingRunError("checkpoint rank recovery state is invalid")
            try:
                cursor = BatchCursor.model_validate(raw_state["cursor"])
            except ValueError as error:
                raise TrainingRunError("checkpoint rank cursor is invalid") from error
            cursors[rank] = cursor
            rng_states[rank] = raw_state["rng_state"]
        if set(cursors) != set(range(self.distributed.world_size)):
            raise TrainingRunError("checkpoint rank recovery state is incomplete")
        local_cursor = cursors[self.distributed.rank]
        expected_cursor = self.source.initial_cursor()
        immutable_fields = (
            "build_manifest_sha256",
            "tokenizer_hash",
            "split",
            "sequence_length",
            "seed",
            "rank",
            "world_size",
            "shuffle",
        )
        if any(
            getattr(local_cursor, field) != getattr(expected_cursor, field)
            for field in immutable_fields
        ):
            raise TrainingRunError("checkpoint rank cursor does not match the run")
        expected_tokens = optimizer_step * self.config.local_tokens_per_optimizer_step
        if local_cursor.tokens_consumed != expected_tokens:
            raise TrainingRunError("checkpoint rank cursor progress is incompatible")
        _restore_portable_rng_state(rng_states[self.distributed.rank])
        return local_cursor

    def _restore(
        self, resume_from: str | Path | None
    ) -> tuple[int, BatchCursor, Path | None]:
        if resume_from is None:
            seed_training(
                self.config.seed + self.distributed.rank,
                cuda=self.config.device == "cuda",
            )
            return 0, self.source.initial_cursor(), None
        restored = restore_checkpoint(
            resume_from,
            model=self.model,
            optimizer=self.optimizer,
            expected_binding=self.binding,
        )
        optimizer_step = restored.manifest.progress.optimizer_step
        if self.distributed.world_size == 1:
            expected_scheduler = self._base_scheduler_state(optimizer_step)
            if restored.scheduler_state != expected_scheduler:
                raise TrainingRunError("checkpoint scheduler state is incompatible")
            cursor = restored.manifest.cursor
        else:
            cursor = self._restore_distributed_state(
                restored.scheduler_state,
                optimizer_step,
            )
        return optimizer_step, cursor, restored.path

    def _save(
        self,
        *,
        optimizer_step: int,
        cursor: BatchCursor,
        parent: Path | None,
    ) -> Path:
        rank_states = self.distributed.all_gather_object(
            self._rank_recovery_state(cursor)
        )
        checkpoint = self.checkpoint_directory / f"step-{optimizer_step:012d}"
        failure: str | None = None
        if self.distributed.is_primary:
            try:
                rank_zero_state = cast(dict[str, Any], rank_states[0])
                rank_zero_cursor = BatchCursor.model_validate(rank_zero_state["cursor"])
                scheduler_state = self._base_scheduler_state(optimizer_step)
                if self.distributed.enabled:
                    scheduler_state["rank_states"] = list(rank_states)
                checkpoint = save_checkpoint(
                    self.checkpoint_directory,
                    model=self.model,
                    optimizer=self.optimizer,
                    cursor=rank_zero_cursor,
                    binding=self.binding,
                    optimizer_step=optimizer_step,
                    scheduler_step=optimizer_step,
                    scheduler_state=scheduler_state,
                    parent_checkpoint=parent,
                )
                apply_checkpoint_retention(
                    self.checkpoint_directory,
                    keep_latest=self.config.keep_latest_checkpoints,
                )
            except Exception as error:
                failure = f"{type(error).__name__}: {error}"
        failure = cast(
            str | None,
            self.distributed.broadcast_primary_object(failure),
        )
        if failure is not None:
            raise TrainingRunError(
                f"rank-zero checkpoint publication failed: {failure}"
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
        sinks = (
            TrainingMetricSinks(
                jsonl_path=self.jsonl_log,
                tensorboard_directory=self.tensorboard_directory,
                parquet_path=self.parquet_log,
                resume_optimizer_step=optimizer_step,
            )
            if self.distributed.is_primary
            else None
        )
        try:
            return self._run_loop(
                optimizer_step=optimizer_step,
                cursor=cursor,
                parent=parent,
                target_step=target_step,
                resumed=resume_from is not None,
                sinks=sinks,
            )
        except BaseException:
            if sinks is not None:
                sinks.abort()
            raise

    def _run_loop(
        self,
        *,
        optimizer_step: int,
        cursor: BatchCursor,
        parent: Path | None,
        target_step: int,
        resumed: bool,
        sinks: TrainingMetricSinks | None,
    ) -> DenseTrainingResult:
        """Execute a validated bounded loop with optional rank-zero sinks."""

        if sinks is not None:
            sinks.append_event(
                {
                    "event": "run_resume" if resumed else "run_start",
                    "optimizer_step": optimizer_step,
                    "training_config": self.config.model_dump(mode="json"),
                    "training_config_sha256": self.config.config_hash,
                }
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
                distributed=self.distributed,
                objective=self.objective,
            )
            optimizer_step += 1
            metrics.append(step_metrics)
            if sinks is not None:
                sinks.log_optimizer_step(step_metrics.model_dump(mode="json"))
            now = time.monotonic()
            primary_reasons = (
                sorted(cadence.due_reasons(optimizer_step, now))
                if self.distributed.is_primary
                else None
            )
            reasons = cast(
                list[str],
                self.distributed.broadcast_primary_object(primary_reasons),
            )
            if reasons:
                parent = self._save(
                    optimizer_step=optimizer_step,
                    cursor=cursor,
                    parent=parent,
                )
                cadence.mark_saved(optimizer_step, now)
                if sinks is not None:
                    sinks.append_event(
                        {
                            "checkpoint": parent.name,
                            "event": "checkpoint",
                            "optimizer_step": optimizer_step,
                            "reasons": reasons,
                        },
                    )
                    sinks.snapshot()

        parent = self._save(
            optimizer_step=optimizer_step,
            cursor=cursor,
            parent=parent,
        )
        if sinks is not None:
            sinks.append_event(
                {
                    "checkpoint": parent.name,
                    "event": (
                        "run_complete"
                        if optimizer_step == self.config.optimization.total_steps
                        else "run_stopped"
                    ),
                    "optimizer_step": optimizer_step,
                    "tokens_consumed": (
                        cursor.tokens_consumed * self.distributed.world_size
                    ),
                },
            )
            sinks.close()
        return DenseTrainingResult(
            optimizer_step=optimizer_step,
            cursor=cursor,
            last_checkpoint=parent,
            metrics=tuple(metrics),
        )


DenseTrainer = CausalTrainer
Mamba2Trainer = CausalTrainer
DiffusionTrainer = CausalTrainer


def optimizer_steps_for_token_budget(
    target_tokens: int,
    *,
    sequence_length: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int = 1,
) -> int:
    """Round a token budget upward to complete optimizer steps."""

    values = (
        target_tokens,
        sequence_length,
        micro_batch_size,
        gradient_accumulation_steps,
        world_size,
    )
    if any(value <= 0 for value in values):
        raise ValueError("token budget and batch dimensions must be positive")
    tokens_per_step = (
        sequence_length * micro_batch_size * gradient_accumulation_steps * world_size
    )
    return math.ceil(target_tokens / tokens_per_step)
