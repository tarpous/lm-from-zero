"""Dry-run resolution for one fresh-process acceleration calibration cell."""

from __future__ import annotations

import json
import time
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn

from lm_from_zero.acceleration_calibration import (
    AccelerationCalibrationError,
    AccelerationCalibrationPlan,
    Architecture,
    CalibrationCell,
    CalibrationNumericalTrace,
    CalibrationResult,
    compare_numerical_traces,
    load_numerical_trace,
    load_plan,
    load_result,
    resolve_cell,
    write_artifact,
    write_numerical_trace,
)
from lm_from_zero.acceleration_runtime import (
    capture_process_cpu_times,
    process_cpu_utilization,
    profile_cuda_sdpa_backend,
    read_compile_graph_break_counters,
    reset_compile_graph_break_counters,
)
from lm_from_zero.acceleration_statistics import (
    resolve_parameter_group_measurements,
    snapshot_parameter_groups,
    update_rms_values,
)
from lm_from_zero.models import (
    Mamba2Config,
    Mamba2ForCausalLM,
    MaskedDiffusionConfig,
    MaskedDiffusionForMaskedLM,
    Olmo2Config,
    Olmo2ForCausalLM,
)
from lm_from_zero.progress import ProgressReporter
from lm_from_zero.sharding import validate_shard_build
from lm_from_zero.tokenizer.bpe import ByteBPE
from lm_from_zero.training import (
    BatchCursor,
    CausalBatch,
    CausalBatchConfig,
    CausalTrainer,
    OptimizationConfig,
    ShardBatchSource,
    TrainingMetricSinks,
    partition_parameters,
    seed_training,
    train_accumulated_step,
)
from lm_from_zero.training.checkpointing import (
    CheckpointError,
    GitMetadata,
    capture_git_metadata,
)
from lm_from_zero.training.runner import (
    CudaEventTimer,
    DenseTrainingConfig,
    DiffusionTrainingConfig,
    Mamba2TrainingConfig,
    TrainingConfig,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40,64}$")]


class AccelerationExecutionError(RuntimeError):
    """Raised when a calibration cell cannot be resolved or safely executed."""


class CalibrationCellDryRun(BaseModel):
    """Exact non-allocating execution contract for one cell repetition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-acceleration-calibration-cell-dry-run"] = (
        "lm-from-zero-acceleration-calibration-cell-dry-run"
    )
    format_version: Literal[3] = 3
    plan_sha256: Sha256
    cell_sha256: Sha256
    repository_revision: GitRevision
    expected_cuda_device_name: str = Field(min_length=1)
    architecture: Architecture
    cell_id: str
    repetition: Annotated[int, Field(gt=0)]
    shard_manifest_sha256: Sha256
    tokenizer_sha256: Sha256
    training_config_sha256: Sha256
    training_config: dict[str, object]
    warmup_optimizer_steps: Annotated[int, Field(ge=0)]
    measured_optimizer_steps: Annotated[int, Field(gt=0)]
    total_optimizer_steps: Annotated[int, Field(gt=1)]
    checkpoint_directory: str
    jsonl_log: str
    numerical_trace_path: str
    result_path: str
    requires_clean_git: Literal[True] = True
    requires_cuda: Literal[True] = True
    execution_ready: Literal[True] = True
    execution_blockers: tuple[()] = ()

    def canonical_json(self) -> str:
        """Return compact deterministic JSON for CLI review."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


class CudaDeviceMetadata(BaseModel):
    """Non-allocating CUDA facts required by a calibration cell."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    device_count: Annotated[int, Field(ge=0)]
    selected_device_index: Annotated[int | None, Field(ge=0)] = None
    device_name: str | None = None
    compute_capability: (
        tuple[Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]] | None
    ) = None
    bf16_supported: bool = False


class CalibrationExecutionPreflight(BaseModel):
    """Evidence that exact-input guards passed before CUDA allocation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-acceleration-calibration-preflight"] = (
        "lm-from-zero-acceleration-calibration-preflight"
    )
    format_version: Literal[1] = 1
    plan_sha256: Sha256
    cell_sha256: Sha256
    architecture: Architecture
    cell_id: str
    repetition: Annotated[int, Field(gt=0)]
    repository_revision: GitRevision
    shard_manifest_sha256: Sha256
    tokenizer_sha256: Sha256
    model_config_sha256: Sha256
    cuda_device_index: Annotated[int, Field(ge=0)]
    cuda_device_name: str = Field(min_length=1)
    cuda_compute_capability: tuple[
        Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]
    ]
    result_path: str
    allocation_performed: Literal[False] = False


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def capture_cuda_device_metadata(device_index: int = 0) -> CudaDeviceMetadata:
    """Probe CUDA support and one device without allocating model tensors."""

    if not torch.cuda.is_available():
        return CudaDeviceMetadata(available=False, device_count=0)
    device_count = torch.cuda.device_count()
    if not 0 <= device_index < device_count:
        return CudaDeviceMetadata(available=True, device_count=device_count)
    properties = torch.cuda.get_device_properties(device_index)
    return CudaDeviceMetadata(
        available=True,
        device_count=device_count,
        selected_device_index=device_index,
        device_name=properties.name,
        compute_capability=(properties.major, properties.minor),
        bf16_supported=torch.cuda.is_bf16_supported(),
    )


def inspect_repository_state(repository: str | Path) -> GitMetadata:
    """Return the current source revision and dirty flag for plan binding."""

    try:
        return capture_git_metadata(repository)
    except CheckpointError as error:
        raise AccelerationExecutionError(
            "could not validate the calibration repository"
        ) from error


def _resolved_path(path: str | Path, repository: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = repository / candidate
    return candidate.resolve(strict=False)


def _training_config(
    plan: AccelerationCalibrationPlan,
    cell: CalibrationCell,
    build_manifest: Path,
) -> TrainingConfig:
    build = validate_shard_build(build_manifest)
    if _file_sha256(build_manifest) != plan.shard_manifest_sha256:
        raise AccelerationExecutionError("shard manifest hash does not match the plan")
    if build.tokenizer_hash != plan.tokenizer_sha256:
        raise AccelerationExecutionError("tokenizer hash does not match the plan")
    total_steps = plan.warmup_optimizer_steps + plan.measured_optimizer_steps
    batch = CausalBatchConfig(
        sequence_length=plan.sequence_length,
        micro_batch_size=plan.micro_batch_size,
        seed=plan.seed,
    )
    settings = cell.settings
    common: dict[str, object] = {
        "batch": batch,
        "optimization": OptimizationConfig(total_steps=total_steps),
        "gradient_accumulation_steps": plan.gradient_accumulation_steps,
        "device": "cuda",
        "precision": "bf16",
        "compile_model": settings.compile_mode != "disabled",
        "compile_mode": (
            "default" if settings.compile_mode == "disabled" else settings.compile_mode
        ),
        "adamw_backend": (
            "fused" if settings.optimizer_backend == "fused-adamw" else "auto"
        ),
        "loss_backend": (
            "linear" if settings.output_loss == "linear-cross-entropy" else "full"
        ),
        "float32_matmul_precision": settings.matmul_precision,
        "telemetry_every_steps": (
            plan.telemetry_interval_steps if settings.telemetry_mode == "sampled" else 1
        ),
        "checkpoint_every_steps": None,
        "checkpoint_every_seconds": 15 * 60,
        "seed": plan.seed,
    }
    if cell.architecture == "dense":
        common.update(
            model=Olmo2Config(tokenizer_hash=build.tokenizer_hash),
            sdpa_backend=settings.sdpa_backend,
        )
        config: TrainingConfig = DenseTrainingConfig.model_validate(common)
    elif cell.architecture == "mamba2":
        common["model"] = Mamba2Config(tokenizer_hash=build.tokenizer_hash)
        config = Mamba2TrainingConfig.model_validate(common)
    else:
        common.update(
            model=MaskedDiffusionConfig(tokenizer_hash=build.tokenizer_hash),
            sdpa_backend=settings.sdpa_backend,
            diffusion_padding_free_attention=(
                settings.diffusion_padding_free_attention
            ),
        )
        config = DiffusionTrainingConfig.model_validate(common)
    if config.model.config_hash != cell.model_config_sha256:
        raise AccelerationExecutionError("model configuration does not match the cell")
    return config


def resolve_calibration_cell_dry_run(
    plan_path: str | Path,
    build_manifest: str | Path,
    architecture: Architecture,
    cell_id: str,
    repetition: int,
) -> CalibrationCellDryRun:
    """Resolve one cell without allocating a model, optimizer, or CUDA state."""

    plan = load_plan(plan_path)
    if not 1 <= repetition <= plan.repetitions:
        raise AccelerationExecutionError("repetition is outside the calibration plan")
    cell = resolve_cell(plan, architecture, cell_id)
    build_path = Path(build_manifest)
    config = _training_config(plan, cell, build_path)
    repetition_root = Path(cell.result_directory) / f"repetition-{repetition:02d}"
    return CalibrationCellDryRun(
        plan_sha256=plan.artifact_sha256,
        cell_sha256=cell.artifact_sha256,
        repository_revision=plan.repository_revision,
        expected_cuda_device_name=plan.expected_cuda_device_name,
        architecture=architecture,
        cell_id=cell_id,
        repetition=repetition,
        shard_manifest_sha256=plan.shard_manifest_sha256,
        tokenizer_sha256=plan.tokenizer_sha256,
        training_config_sha256=config.config_hash,
        training_config=config.model_dump(mode="json"),
        warmup_optimizer_steps=plan.warmup_optimizer_steps,
        measured_optimizer_steps=plan.measured_optimizer_steps,
        total_optimizer_steps=config.optimization.total_steps,
        checkpoint_directory=(repetition_root / "checkpoints").as_posix(),
        jsonl_log=(repetition_root / "events.jsonl").as_posix(),
        numerical_trace_path=(repetition_root / "numerical-trace.json").as_posix(),
        result_path=(repetition_root / "result.json").as_posix(),
    )


def validate_calibration_execution_preflight(
    dry_run: CalibrationCellDryRun,
    *,
    plan_path: str | Path,
    build_manifest: str | Path,
    tokenizer_model: str | Path,
    repository: str | Path,
    result_path: str | Path,
    expected_cuda_device_name: str | None = None,
    cuda_device_index: int = 0,
    repository_state: GitMetadata | None = None,
    cuda_state: CudaDeviceMetadata | None = None,
) -> CalibrationExecutionPreflight:
    """Validate immutable inputs and the host before allocating training state.

    ``repository_state`` and ``cuda_state`` are injectable snapshots for tests.
    Production callers should omit them so the state is captured immediately
    before execution.
    """

    if not isinstance(dry_run.training_config.get("model"), dict):
        raise AccelerationExecutionError(
            "model configuration input is missing from the reviewed dry run"
        )
    repository_path = Path(repository).resolve(strict=False)
    plan_file = _resolved_path(plan_path, repository_path)
    build_file = _resolved_path(build_manifest, repository_path)
    tokenizer_file = _resolved_path(tokenizer_model, repository_path)
    for label, path in (
        ("calibration plan", plan_file),
        ("shard build manifest", build_file),
        ("tokenizer model", tokenizer_file),
    ):
        if not path.is_file():
            raise AccelerationExecutionError(f"{label} is missing: {path}")

    try:
        plan = load_plan(plan_file)
    except (AccelerationCalibrationError, OSError, ValueError) as error:
        raise AccelerationExecutionError(
            f"invalid calibration plan: {error}"
        ) from error
    if plan.artifact_sha256 != dry_run.plan_sha256:
        raise AccelerationExecutionError(
            "calibration plan artifact does not match the resolved cell"
        )
    if (
        expected_cuda_device_name is not None
        and expected_cuda_device_name != plan.expected_cuda_device_name
    ):
        raise AccelerationExecutionError(
            "requested CUDA device name does not match the calibration plan"
        )

    try:
        resolved = resolve_calibration_cell_dry_run(
            plan_file,
            build_file,
            dry_run.architecture,
            dry_run.cell_id,
            dry_run.repetition,
        )
    except (AccelerationCalibrationError, OSError, ValueError) as error:
        raise AccelerationExecutionError(
            f"invalid shard or model inputs: {error}"
        ) from error
    if resolved != dry_run:
        raise AccelerationExecutionError(
            "resolved shard or model inputs do not match the reviewed dry run"
        )

    try:
        tokenizer = ByteBPE.load(tokenizer_file)
    except (OSError, ValueError) as error:
        raise AccelerationExecutionError(f"invalid tokenizer model: {error}") from error
    if tokenizer.model_hash != dry_run.tokenizer_sha256:
        raise AccelerationExecutionError("tokenizer model hash does not match the plan")
    if tokenizer.vocab_size != plan.tokenizer_vocab_size:
        raise AccelerationExecutionError(
            "tokenizer vocabulary size does not match the plan"
        )

    planned_result = _resolved_path(dry_run.result_path, repository_path)
    requested_result = _resolved_path(result_path, repository_path)
    if requested_result != planned_result:
        raise AccelerationExecutionError(
            "result path does not match the planned cell repetition"
        )
    if requested_result.exists():
        raise AccelerationExecutionError(
            "refusing to overwrite an existing calibration result"
        )
    if requested_result.parent.exists():
        raise AccelerationExecutionError(
            "refusing to reuse a partial calibration repetition directory"
        )

    try:
        current_repository = (
            capture_git_metadata(repository_path)
            if repository_state is None
            else repository_state
        )
    except CheckpointError as error:
        raise AccelerationExecutionError(
            "could not validate the calibration repository"
        ) from error
    if current_repository.dirty:
        raise AccelerationExecutionError(
            "calibration execution requires a clean Git worktree"
        )
    if current_repository.revision != plan.repository_revision:
        raise AccelerationExecutionError(
            "repository revision does not match the planned revision"
        )

    detected_cuda = (
        capture_cuda_device_metadata(cuda_device_index)
        if cuda_state is None
        else cuda_state
    )
    if not detected_cuda.available:
        raise AccelerationExecutionError("CUDA is not available")
    if detected_cuda.selected_device_index != cuda_device_index:
        raise AccelerationExecutionError("requested CUDA device does not exist")
    if detected_cuda.device_name != plan.expected_cuda_device_name:
        raise AccelerationExecutionError(
            "CUDA device does not match the planned calibration hardware"
        )
    if detected_cuda.compute_capability is None:
        raise AccelerationExecutionError("CUDA compute capability is unavailable")
    if not detected_cuda.bf16_supported:
        raise AccelerationExecutionError(
            "CUDA device does not support the planned bf16 precision"
        )

    cell = resolve_cell(plan, dry_run.architecture, dry_run.cell_id)
    return CalibrationExecutionPreflight(
        plan_sha256=dry_run.plan_sha256,
        cell_sha256=dry_run.cell_sha256,
        architecture=dry_run.architecture,
        cell_id=dry_run.cell_id,
        repetition=dry_run.repetition,
        repository_revision=current_repository.revision,
        shard_manifest_sha256=dry_run.shard_manifest_sha256,
        tokenizer_sha256=dry_run.tokenizer_sha256,
        model_config_sha256=cell.model_config_sha256,
        cuda_device_index=cuda_device_index,
        cuda_device_name=detected_cuda.device_name,
        cuda_compute_capability=detected_cuda.compute_capability,
        result_path=dry_run.result_path,
    )


def _model_for_config(config: TrainingConfig) -> nn.Module:
    if isinstance(config, DenseTrainingConfig):
        return Olmo2ForCausalLM(config.model)
    if isinstance(config, Mamba2TrainingConfig):
        return Mamba2ForCausalLM(config.model)
    return MaskedDiffusionForMaskedLM(config.model)


def _next_batches(
    source: ShardBatchSource,
    cursor: BatchCursor,
    accumulation_steps: int,
) -> tuple[list[CausalBatch], BatchCursor]:
    batches: list[CausalBatch] = []
    current = cursor
    for _ in range(accumulation_steps):
        batch = source.next_batch(current)
        batches.append(batch)
        current = batch.cursor_after
    return batches, current


def _baseline_paths(
    plan: AccelerationCalibrationPlan,
    architecture: Architecture,
    repetition: int,
    repository: Path,
) -> tuple[Path, Path]:
    baseline = resolve_cell(plan, architecture, "baseline")
    root = _resolved_path(
        Path(baseline.result_directory) / f"repetition-{repetition:02d}",
        repository,
    )
    return root / "result.json", root / "numerical-trace.json"


def _validate_baseline_evidence(
    plan: AccelerationCalibrationPlan,
    cell: CalibrationCell,
    repetition: int,
    repository: Path,
) -> tuple[CalibrationResult | None, CalibrationNumericalTrace | None]:
    if cell.cell_id == "baseline":
        return None, None
    result_path, trace_path = _baseline_paths(
        plan, cell.architecture, repetition, repository
    )
    try:
        result = load_result(result_path)
        trace = load_numerical_trace(trace_path)
    except AccelerationCalibrationError as error:
        raise AccelerationExecutionError(
            "the matching baseline repetition must complete before a candidate"
        ) from error
    if (
        result.plan_sha256 != plan.artifact_sha256
        or result.cell_id != "baseline"
        or result.architecture != cell.architecture
        or result.repetition != repetition
        or result.numerical_trace_sha256 != trace.artifact_sha256
    ):
        raise AccelerationExecutionError(
            "the matching baseline result and numerical trace are incompatible"
        )
    return result, trace


def _evaluation_batch(
    build_manifest: Path,
    config: TrainingConfig,
) -> CausalBatch:
    evaluation_batch = config.batch.model_copy(
        update={"split": "validation", "shuffle": False}
    )
    source = ShardBatchSource(build_manifest, evaluation_batch)
    return source.next_batch(source.initial_cursor())


def _evaluate_once(
    trainer: CausalTrainer,
    config: TrainingConfig,
    batch: CausalBatch,
) -> float:
    trainer.forward_model.eval()
    torch.cuda.synchronize()
    started = time.perf_counter()
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        trainer.objective.loss(
            trainer.forward_model,
            batch,
            trainer.distributed.torch_device("cuda"),
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    trainer.forward_model.train()
    return elapsed


def _profile_sdpa(
    trainer: CausalTrainer,
    config: TrainingConfig,
    batch: CausalBatch,
) -> str | None:
    if isinstance(config, Mamba2TrainingConfig):
        return None

    def forward() -> object:
        with (
            torch.inference_mode(),
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        ):
            return trainer.objective.loss(
                trainer.forward_model,
                batch,
                trainer.distributed.torch_device("cuda"),
            )

    return profile_cuda_sdpa_backend(forward)


def _execute_preflighted_calibration(
    *,
    plan: AccelerationCalibrationPlan,
    cell: CalibrationCell,
    dry_run: CalibrationCellDryRun,
    preflight: CalibrationExecutionPreflight,
    build_manifest: Path,
    repository: Path,
    baseline_result: CalibrationResult | None,
    baseline_trace: CalibrationNumericalTrace | None,
) -> CalibrationResult:  # pragma: no cover - approval-gated CUDA path
    """Run one already-guarded CUDA cell and atomically publish its evidence."""

    if cell.architecture == "dense":
        config: TrainingConfig = DenseTrainingConfig.model_validate(
            dry_run.training_config
        )
    elif cell.architecture == "mamba2":
        config = Mamba2TrainingConfig.model_validate(dry_run.training_config)
    else:
        config = DiffusionTrainingConfig.model_validate(dry_run.training_config)

    seed_training(config.seed, cuda=True)
    reset_compile_graph_break_counters()
    source = ShardBatchSource(build_manifest, config.batch)
    model = _model_for_config(config)
    trainer = CausalTrainer(
        model=cast("Any", model),
        source=source,
        config=config,
        checkpoint_directory=_resolved_path(dry_run.checkpoint_directory, repository),
        repository=repository,
        jsonl_log=_resolved_path(dry_run.jsonl_log, repository),
    )
    progress = ProgressReporter(
        f"calibration {cell.architecture}/{cell.cell_id}",
        enabled=trainer.distributed.is_primary,
    )
    progress.phase("warm-up", total=plan.warmup_optimizer_steps)
    cursor = source.initial_cursor()
    partition = partition_parameters(trainer.model)
    parity_snapshot = snapshot_parameter_groups(partition)
    parity_losses: list[float] = []
    parity_gradients: list[float] = []
    cold_compiled_step_seconds = 0.0
    warm_compiled_step_seconds = 0.0

    for step in range(plan.warmup_optimizer_steps):
        batches, cursor = _next_batches(
            source, cursor, config.gradient_accumulation_steps
        )
        collect_parity = step < 3
        if collect_parity:
            torch.cuda.synchronize()
            step_started = time.perf_counter()
        step_metrics = train_accumulated_step(
            trainer.forward_model,
            trainer.optimizer,
            batches,
            config,
            zero_based_step=step,
            distributed=trainer.distributed,
            objective=trainer.objective,
            collect_telemetry=collect_parity,
            reset_peak_memory=False,
        )
        if collect_parity:
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - step_started
            assert step_metrics is not None
            parity_losses.append(step_metrics.loss)
            parity_gradients.append(step_metrics.gradient_norm)
            if step == 0 and config.compile_model:
                cold_compiled_step_seconds = step_seconds
            if step == 2 and config.compile_model:
                warm_compiled_step_seconds = step_seconds
        progress.update(step + 1)

    parity_measurements = resolve_parameter_group_measurements(parity_snapshot)
    numerical_trace = CalibrationNumericalTrace(
        plan_sha256=plan.artifact_sha256,
        cell_sha256=cell.artifact_sha256,
        repository_revision=plan.repository_revision,
        architecture=cell.architecture,
        cell_id=cell.cell_id,
        repetition=dry_run.repetition,
        loss_values=tuple(parity_losses),
        gradient_norm_values=tuple(parity_gradients),
        update_rms_values=update_rms_values(parity_measurements),
    )
    if baseline_trace is None:
        loss_delta = gradient_delta = update_delta = None
    else:
        loss_delta, gradient_delta, update_delta = compare_numerical_traces(
            baseline_trace, numerical_trace
        )

    measurement_snapshot = snapshot_parameter_groups(partition)
    timer = CudaEventTimer()
    sinks = TrainingMetricSinks(
        jsonl_path=_resolved_path(dry_run.jsonl_log, repository),
        tensorboard_directory=None,
        parquet_path=None,
        resume_optimizer_step=plan.warmup_optimizer_steps,
        durable_every_steps=config.metrics_durable_every_steps,
        durable_every_seconds=config.metrics_durable_every_seconds,
    )
    sinks.append_event(
        {
            "event": "calibration_measurement_start",
            "optimizer_step": plan.warmup_optimizer_steps,
            "plan_sha256": plan.artifact_sha256,
            "cell_sha256": cell.artifact_sha256,
        }
    )
    cpu_data_seconds = 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started_cpu = capture_process_cpu_times()
    telemetry_started = started_cpu.wall_seconds
    telemetry_window_steps = 0
    try:
        progress.phase("measured training", total=plan.measured_optimizer_steps)
        for offset in range(plan.measured_optimizer_steps):
            data_started = time.perf_counter()
            batches, cursor = _next_batches(
                source, cursor, config.gradient_accumulation_steps
            )
            cpu_data_seconds += time.perf_counter() - data_started
            optimizer_step = plan.warmup_optimizer_steps + offset
            telemetry_window_steps += 1
            collect = (
                (offset + 1) % config.telemetry_every_steps == 0
                or offset + 1 == plan.measured_optimizer_steps
            )
            metrics = train_accumulated_step(
                trainer.forward_model,
                trainer.optimizer,
                batches,
                config,
                zero_based_step=optimizer_step,
                distributed=trainer.distributed,
                objective=trainer.objective,
                collect_telemetry=collect,
                measurement_started_seconds=telemetry_started,
                measurement_optimizer_steps=telemetry_window_steps,
                reset_peak_memory=False,
                cuda_event_timing=timer,
            )
            if metrics is not None:
                sinks.log_optimizer_step(metrics.model_dump(mode="json"))
                telemetry_started = time.perf_counter()
                telemetry_window_steps = 0
            progress.update(offset + 1)
        cuda_timing = timer.resolve(reset=True)
        sinks.durable_sync()
        ended_cpu = capture_process_cpu_times()
    except BaseException:
        sinks.abort()
        raise

    measured_end_to_end_seconds = ended_cpu.wall_seconds - started_cpu.wall_seconds
    measured_tokens = plan.measured_optimizer_steps * config.tokens_per_optimizer_step
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    parameter_groups = resolve_parameter_group_measurements(measurement_snapshot)
    graph_breaks = read_compile_graph_break_counters()
    evaluation_batch = _evaluation_batch(build_manifest, config)
    progress.phase("evaluation")
    evaluation_seconds = _evaluate_once(trainer, config, evaluation_batch)
    observed_sdpa_backend = _profile_sdpa(trainer, config, evaluation_batch)

    sinks.durable_sync()
    checkpoint_started = time.perf_counter()
    progress.phase("publishing checkpoint")
    checkpoint = trainer._save(
        optimizer_step=config.optimization.total_steps,
        cursor=cursor,
        parent=None,
    )
    checkpoint_seconds = time.perf_counter() - checkpoint_started
    sinks.append_event(
        {
            "checkpoint": checkpoint.name,
            "event": "calibration_complete",
            "optimizer_step": config.optimization.total_steps,
        }
    )
    sinks.close()

    trace_path = _resolved_path(dry_run.numerical_trace_path, repository)
    result_path = _resolved_path(dry_run.result_path, repository)
    write_numerical_trace(trace_path, numerical_trace)
    checkpoint_manifest = checkpoint / "manifest.json"
    event_log = _resolved_path(dry_run.jsonl_log, repository)
    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise AccelerationExecutionError("CUDA runtime version is unavailable")
    result = CalibrationResult(
        plan_sha256=plan.artifact_sha256,
        cell_sha256=cell.artifact_sha256,
        repository_revision=preflight.repository_revision,
        cuda_device_name=preflight.cuda_device_name,
        cuda_compute_capability=preflight.cuda_compute_capability,
        torch_version=torch.__version__,
        cuda_version=cuda_version,
        architecture=cell.architecture,
        cell_id=cell.cell_id,
        repetition=dry_run.repetition,
        shard_manifest_sha256=plan.shard_manifest_sha256,
        model_config_sha256=cell.model_config_sha256,
        training_config_sha256=dry_run.training_config_sha256,
        seed=plan.seed,
        sequence_length=plan.sequence_length,
        micro_batch_size=plan.micro_batch_size,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        world_size=plan.world_size,
        warmup_optimizer_steps=plan.warmup_optimizer_steps,
        measured_optimizer_steps=plan.measured_optimizer_steps,
        measured_end_to_end_seconds=measured_end_to_end_seconds,
        measured_cuda_compute_seconds=cuda_timing.compute_milliseconds / 1_000,
        tokens_per_second=measured_tokens / measured_end_to_end_seconds,
        cold_compiled_step_seconds=cold_compiled_step_seconds,
        warm_compiled_step_seconds=warm_compiled_step_seconds,
        estimated_compile_overhead_seconds=max(
            0.0, cold_compiled_step_seconds - warm_compiled_step_seconds
        ),
        optimizer_seconds=cuda_timing.optimizer_milliseconds / 1_000,
        evaluation_seconds=evaluation_seconds,
        checkpoint_seconds=checkpoint_seconds,
        metric_fsync_seconds=sinks.metric_fsync_seconds,
        cpu_data_seconds=cpu_data_seconds,
        process_cpu_utilization=process_cpu_utilization(started_cpu, ended_cpu),
        peak_cuda_allocated_bytes=peak_allocated,
        peak_cuda_reserved_bytes=peak_reserved,
        graph_breaks=graph_breaks.graph_breaks,
        graph_break_reasons=graph_breaks.reasons,
        observed_sdpa_backend=observed_sdpa_backend,
        baseline_result_sha256=(
            None if baseline_result is None else baseline_result.artifact_sha256
        ),
        numerical_trace_sha256=numerical_trace.artifact_sha256,
        checkpoint_manifest_sha256=_file_sha256(checkpoint_manifest),
        event_log_sha256=_file_sha256(event_log),
        parameter_groups=parameter_groups,
        maximum_loss_absolute_delta=loss_delta,
        maximum_gradient_absolute_delta=gradient_delta,
        maximum_update_absolute_delta=update_delta,
    )
    write_artifact(result_path, result)
    progress.finish("complete")
    return result


def execute_calibration_cell(
    plan_path: str | Path,
    build_manifest: str | Path,
    tokenizer_model: str | Path,
    repository: str | Path,
    architecture: Architecture,
    cell_id: str,
    repetition: int,
) -> CalibrationResult:
    """Guard, execute, and atomically record one fresh-process CUDA cell."""

    repository_path = Path(repository).resolve(strict=False)
    plan_file = _resolved_path(plan_path, repository_path)
    build_file = _resolved_path(build_manifest, repository_path)
    tokenizer_file = _resolved_path(tokenizer_model, repository_path)
    dry_run = resolve_calibration_cell_dry_run(
        plan_file, build_file, architecture, cell_id, repetition
    )
    preflight = validate_calibration_execution_preflight(
        dry_run,
        plan_path=plan_file,
        build_manifest=build_file,
        tokenizer_model=tokenizer_file,
        repository=repository_path,
        result_path=dry_run.result_path,
    )
    plan = load_plan(plan_file)
    cell = resolve_cell(plan, architecture, cell_id)
    baseline_result, baseline_trace = _validate_baseline_evidence(
        plan, cell, repetition, repository_path
    )
    return _execute_preflighted_calibration(
        plan=plan,
        cell=cell,
        dry_run=dry_run,
        preflight=preflight,
        build_manifest=build_file,
        repository=repository_path,
        baseline_result=baseline_result,
        baseline_trace=baseline_trace,
    )


def require_executable_calibration_cell(dry_run: CalibrationCellDryRun) -> None:
    """Reject a reviewed cell if any execution blocker remains."""

    if not dry_run.execution_ready or dry_run.execution_blockers:
        raise AccelerationExecutionError("; ".join(dry_run.execution_blockers))


def resolve_from_plan(
    plan_path: str | Path,
    build_manifest: str | Path,
    architecture: Architecture,
    cell_id: str,
    repetition: int,
) -> CalibrationCellDryRun:
    """Descriptive alias used by the CLI."""

    try:
        return resolve_calibration_cell_dry_run(
            plan_path,
            build_manifest,
            architecture,
            cell_id,
            repetition,
        )
    except AccelerationCalibrationError as error:
        raise AccelerationExecutionError(str(error)) from error
