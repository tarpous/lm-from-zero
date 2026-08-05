"""Deterministic Milestone 6A acceleration calibration contracts.

This module plans and validates measurements; it deliberately does not execute
GPU work.  The staged matrix changes one candidate at a time after establishing
the sampled-telemetry hot path, avoiding an uninformative Cartesian sweep.
"""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from statistics import median
from typing import Annotated, Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lm_from_zero.models import Mamba2Config, MaskedDiffusionConfig, Olmo2Config
from lm_from_zero.sharding import validate_shard_build

Architecture = Literal["dense", "mamba2", "diffusion"]
CalibrationStage = Literal[
    "baseline",
    "hot-path",
    "compiler",
    "optimizer",
    "attention",
    "math",
    "loss",
]
CompileMode = Literal[
    "disabled",
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
ExcludeDisabledCompileMode = Literal[
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
]
OptimizerBackend = Literal["adamw", "fused-adamw"]
SdpaBackend = Literal["auto", "flash"]
MatmulPrecision = Literal["highest", "high"]
OutputLoss = Literal["materialized-cross-entropy", "linear-cross-entropy"]
TelemetryMode = Literal["per-step", "sampled"]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
GitRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
ModelConfig = Olmo2Config | Mamba2Config | MaskedDiffusionConfig

ARCHITECTURES: tuple[Architecture, ...] = ("dense", "mamba2", "diffusion")
COMPILE_MODES: tuple[ExcludeDisabledCompileMode, ...] = (
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
)


class AccelerationCalibrationError(RuntimeError):
    """Raised when calibration evidence violates its frozen plan."""


class CanonicalArtifact(Protocol):
    """Structural contract for atomic canonical-JSON artifacts."""

    def canonical_json(self) -> str:
        """Return compact, key-sorted JSON."""


class _CanonicalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    def canonical_json(self) -> str:
        """Return stable machine-readable evidence."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def artifact_sha256(self) -> str:
        """Hash the canonical representation used for artifact binding."""

        return sha256(self.canonical_json().encode()).hexdigest()


class CalibrationSettings(_CanonicalModel):
    """All implementation choices controlled by one calibration cell."""

    compile_mode: CompileMode = "disabled"
    optimizer_backend: OptimizerBackend = "adamw"
    sdpa_backend: SdpaBackend = "auto"
    matmul_precision: MatmulPrecision = "highest"
    output_loss: OutputLoss = "materialized-cross-entropy"
    telemetry_mode: TelemetryMode = "per-step"
    diffusion_padding_free_attention: bool = False


CellSpec = tuple[
    str,
    CalibrationStage,
    str | None,
    CalibrationSettings,
]


class CalibrationCell(_CanonicalModel):
    """One architecture-specific, one-factor calibration experiment."""

    architecture: Architecture
    cell_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    stage: CalibrationStage
    parent_cell_id: str | None
    settings: CalibrationSettings
    model_config_sha256: Sha256
    parameter_count: Annotated[int, Field(gt=0)]
    result_directory: str


class AccelerationCalibrationPlan(_CanonicalModel):
    """Frozen synchronized plan for all Milestone 6A measurements."""

    format: Literal["lm-from-zero-acceleration-calibration-plan"] = (
        "lm-from-zero-acceleration-calibration-plan"
    )
    format_version: Literal[2] = 2
    repository_revision: GitRevision
    expected_cuda_device_name: str = Field(min_length=1)
    shard_manifest_sha256: Sha256
    tokenizer_sha256: Sha256
    tokenizer_vocab_size: Literal[16_000]
    seed: int
    sequence_length: Literal[1_024]
    micro_batch_size: Annotated[int, Field(gt=0)]
    gradient_accumulation_steps: Annotated[int, Field(gt=0)]
    world_size: Literal[1] = 1
    warmup_optimizer_steps: Annotated[int, Field(ge=3)]
    measured_optimizer_steps: Annotated[int, Field(gt=0)]
    minimum_measured_seconds: Annotated[float, Field(gt=0)]
    repetitions: Annotated[int, Field(gt=0)]
    telemetry_interval_steps: Annotated[int, Field(gt=1)]
    promotion_minimum_speedup: Annotated[float, Field(ge=1)]
    maximum_loss_absolute_delta: Annotated[float, Field(ge=0)]
    maximum_gradient_absolute_delta: Annotated[float, Field(ge=0)]
    maximum_update_absolute_delta: Annotated[float, Field(ge=0)]
    artifact_root: str
    cells: tuple[CalibrationCell, ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        expected = _cell_specs(self.telemetry_interval_steps)
        realized = [(cell.architecture, cell.cell_id) for cell in self.cells]
        expected_keys = [
            (architecture, cell_id)
            for architecture in ARCHITECTURES
            for cell_id, _, _, _ in expected[architecture]
        ]
        if realized != expected_keys:
            raise ValueError(
                "calibration cells must match the staged architecture matrix"
            )
        if len(realized) != len(set(realized)):
            raise ValueError("calibration cells must be unique")
        model_bindings: dict[Architecture, tuple[str, int]] = {}
        for cell in self.cells:
            binding = (cell.model_config_sha256, cell.parameter_count)
            previous = model_bindings.setdefault(cell.architecture, binding)
            if previous != binding:
                raise ValueError(
                    "model binding must be constant within each architecture"
                )
            expected_spec = next(
                item for item in expected[cell.architecture] if item[0] == cell.cell_id
            )
            _, stage, parent, settings = expected_spec
            if (cell.stage, cell.parent_cell_id, cell.settings) != (
                stage,
                parent,
                settings,
            ):
                raise ValueError("calibration cell disagrees with its staged settings")
        paths = [cell.result_directory for cell in self.cells]
        if len(paths) != len(set(paths)):
            raise ValueError("calibration result directories must be unique")
        return self


class CalibrationNumericalTrace(_CanonicalModel):
    """Small untimed warm-up trace used for baseline parity comparisons."""

    format: Literal["lm-from-zero-acceleration-numerical-trace"] = (
        "lm-from-zero-acceleration-numerical-trace"
    )
    format_version: Literal[1] = 1
    plan_sha256: Sha256
    cell_sha256: Sha256
    repository_revision: GitRevision
    architecture: Architecture
    cell_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    repetition: Annotated[int, Field(gt=0)]
    loss_values: Annotated[tuple[float, ...], Field(min_length=1)]
    gradient_norm_values: Annotated[tuple[float, ...], Field(min_length=1)]
    update_rms_values: Annotated[tuple[float, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        if len(self.loss_values) != len(self.gradient_norm_values):
            raise ValueError("loss and gradient traces must have equal length")
        return self


class ParameterGroupMeasurement(_CanonicalModel):
    """Measured parameter/update scale for one optimizer partition."""

    name: str = Field(min_length=1)
    parameter_count: Annotated[int, Field(gt=0)]
    weight_rms: Annotated[float, Field(ge=0)]
    gradient_rms: Annotated[float, Field(ge=0)]
    update_rms: Annotated[float, Field(ge=0)]
    angular_learning_rate: Annotated[float, Field(ge=0)]
    effective_learning_rate: Annotated[float, Field(ge=0)]


class CalibrationResult(_CanonicalModel):
    """One recorded repetition for a planned calibration cell."""

    format: Literal["lm-from-zero-acceleration-calibration-result"] = (
        "lm-from-zero-acceleration-calibration-result"
    )
    format_version: Literal[2] = 2
    plan_sha256: Sha256
    cell_sha256: Sha256
    repository_revision: GitRevision
    cuda_device_name: str = Field(min_length=1)
    cuda_compute_capability: tuple[
        Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]
    ]
    torch_version: str = Field(min_length=1)
    cuda_version: str = Field(min_length=1)
    architecture: Architecture
    cell_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    repetition: Annotated[int, Field(gt=0)]
    shard_manifest_sha256: Sha256
    model_config_sha256: Sha256
    training_config_sha256: Sha256
    seed: int
    sequence_length: Annotated[int, Field(gt=0)]
    micro_batch_size: Annotated[int, Field(gt=0)]
    gradient_accumulation_steps: Annotated[int, Field(gt=0)]
    world_size: Annotated[int, Field(gt=0)]
    warmup_optimizer_steps: Annotated[int, Field(ge=0)]
    measured_optimizer_steps: Annotated[int, Field(gt=0)]
    measured_end_to_end_seconds: Annotated[float, Field(gt=0)]
    measured_cuda_compute_seconds: Annotated[float, Field(gt=0)]
    tokens_per_second: Annotated[float, Field(gt=0)]
    cold_compiled_step_seconds: Annotated[float, Field(ge=0)]
    warm_compiled_step_seconds: Annotated[float, Field(ge=0)]
    estimated_compile_overhead_seconds: Annotated[float, Field(ge=0)]
    optimizer_seconds: Annotated[float, Field(ge=0)]
    evaluation_seconds: Annotated[float, Field(ge=0)]
    checkpoint_seconds: Annotated[float, Field(ge=0)]
    metric_fsync_seconds: Annotated[float, Field(ge=0)]
    cpu_data_seconds: Annotated[float, Field(ge=0)]
    process_cpu_utilization: Annotated[float, Field(ge=0)]
    peak_cuda_allocated_bytes: Annotated[int, Field(ge=0)]
    peak_cuda_reserved_bytes: Annotated[int, Field(ge=0)]
    graph_breaks: Annotated[int, Field(ge=0)]
    observed_sdpa_backend: str | None = None
    graph_break_reasons: tuple[str, ...] = ()
    baseline_result_sha256: Sha256 | None = None
    numerical_trace_sha256: Sha256
    checkpoint_manifest_sha256: Sha256
    event_log_sha256: Sha256
    parameter_groups: Annotated[
        tuple[ParameterGroupMeasurement, ...], Field(min_length=1)
    ]
    maximum_loss_absolute_delta: Annotated[float | None, Field(ge=0)] = None
    maximum_gradient_absolute_delta: Annotated[float | None, Field(ge=0)] = None
    maximum_update_absolute_delta: Annotated[float | None, Field(ge=0)] = None

    @model_validator(mode="after")
    def validate_measurement(self) -> Self:
        measured_tokens = (
            self.measured_optimizer_steps
            * self.sequence_length
            * self.micro_batch_size
            * self.gradient_accumulation_steps
            * self.world_size
        )
        expected_throughput = measured_tokens / self.measured_end_to_end_seconds
        tolerance = max(1e-9, expected_throughput * 1e-9)
        if abs(self.tokens_per_second - expected_throughput) > tolerance:
            raise ValueError("tokens per second must derive from end-to-end wall time")
        if self.measured_cuda_compute_seconds > self.measured_end_to_end_seconds:
            raise ValueError("CUDA time cannot exceed end-to-end time")
        parity_values = (
            self.maximum_loss_absolute_delta,
            self.maximum_gradient_absolute_delta,
            self.maximum_update_absolute_delta,
        )
        if self.cell_id == "baseline":
            if self.baseline_result_sha256 is not None or any(
                value is not None for value in parity_values
            ):
                raise ValueError("baseline cannot reference baseline parity evidence")
        elif self.baseline_result_sha256 is None or any(
            value is None for value in parity_values
        ):
            raise ValueError("candidate requires complete baseline parity evidence")
        return self


class CalibrationCellSummary(_CanonicalModel):
    """Median performance and promotion decision for one cell."""

    architecture: Architecture
    cell_id: str
    median_tokens_per_second: Annotated[float, Field(gt=0)]
    baseline_median_tokens_per_second: Annotated[float, Field(gt=0)]
    speedup_over_baseline: Annotated[float, Field(gt=0)]
    maximum_loss_absolute_delta: Annotated[float, Field(ge=0)]
    maximum_gradient_absolute_delta: Annotated[float, Field(ge=0)]
    maximum_update_absolute_delta: Annotated[float, Field(ge=0)]
    numerical_parity_passed: bool
    promoted: bool

    @model_validator(mode="after")
    def validate_decision(self) -> Self:
        if self.cell_id == "baseline" and self.promoted:
            raise ValueError("baseline cannot be promoted")
        if self.promoted and not self.numerical_parity_passed:
            raise ValueError("a promoted cell must pass numerical parity")
        return self


class AccelerationCalibrationReport(_CanonicalModel):
    """Derived Milestone 6A report built only from recorded results."""

    format: Literal["lm-from-zero-acceleration-calibration-report"] = (
        "lm-from-zero-acceleration-calibration-report"
    )
    format_version: Literal[1] = 1
    plan_sha256: Sha256
    result_count: Annotated[int, Field(gt=0)]
    promotion_minimum_speedup: Annotated[float, Field(ge=1)]
    summaries: tuple[CalibrationCellSummary, ...]


def _settings(**updates: object) -> CalibrationSettings:
    payload = CalibrationSettings(compile_mode="default").model_dump(mode="python")
    payload.update(updates)
    return CalibrationSettings.model_validate(payload)


def _cell_specs(
    telemetry_interval_steps: int,
) -> dict[Architecture, list[CellSpec]]:
    del telemetry_interval_steps  # The interval is synchronized at plan level.
    common: list[CellSpec] = [
        ("baseline", "baseline", None, _settings()),
        (
            "sampled-telemetry",
            "hot-path",
            "baseline",
            _settings(telemetry_mode="sampled"),
        ),
    ]
    sampled = CalibrationSettings(compile_mode="default", telemetry_mode="sampled")
    compiler_modes: tuple[CompileMode, ...] = (
        "disabled",
        "reduce-overhead",
        "max-autotune",
        "max-autotune-no-cudagraphs",
    )
    for mode in compiler_modes:
        common.append(
            (
                f"compile-{mode}",
                "compiler",
                "sampled-telemetry",
                sampled.model_copy(update={"compile_mode": mode}),
            )
        )
    common.append(
        (
            "fused-adamw",
            "optimizer",
            "sampled-telemetry",
            sampled.model_copy(update={"optimizer_backend": "fused-adamw"}),
        )
    )
    sdpa: CellSpec = (
        "flash-sdpa",
        "attention",
        "sampled-telemetry",
        sampled.model_copy(update={"sdpa_backend": "flash"}),
    )
    diffusion_sdpa: CellSpec = (
        "flash-sdpa",
        "attention",
        "sampled-telemetry",
        sampled.model_copy(
            update={
                "sdpa_backend": "flash",
                "diffusion_padding_free_attention": True,
            }
        ),
    )
    linear_loss: CellSpec = (
        "linear-cross-entropy",
        "loss",
        "sampled-telemetry",
        sampled.model_copy(update={"output_loss": "linear-cross-entropy"}),
    )
    tf32: CellSpec = (
        "tf32-high",
        "math",
        "sampled-telemetry",
        sampled.model_copy(update={"matmul_precision": "high"}),
    )
    return {
        "dense": [*common, sdpa, linear_loss],
        "mamba2": [*common, tf32],
        "diffusion": [*common, diffusion_sdpa, linear_loss],
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def create_plan(
    build_manifest: str | Path,
    *,
    artifact_root: str | Path = "artifacts/acceleration-calibration/results",
    seed: int = 1_337,
    sequence_length: Literal[1_024] = 1_024,
    micro_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    warmup_optimizer_steps: int = 50,
    measured_optimizer_steps: int = 100,
    minimum_measured_seconds: float = 60.0,
    repetitions: int = 3,
    telemetry_interval_steps: int = 50,
    promotion_minimum_speedup: float = 1.10,
    maximum_loss_absolute_delta: float = 1e-4,
    maximum_gradient_absolute_delta: float = 1e-3,
    maximum_update_absolute_delta: float = 1e-3,
    repository_revision: str,
    expected_cuda_device_name: str = "NVIDIA GeForce RTX 4080 SUPER",
) -> AccelerationCalibrationPlan:
    """Create the deterministic, synchronized Milestone 6A calibration plan."""

    build_path = Path(build_manifest)
    build = validate_shard_build(build_path)
    if build.tokenizer_vocab_size != 16_000:
        raise AccelerationCalibrationError("calibration requires the 16K tokenizer")
    if sequence_length != 1_024:
        raise AccelerationCalibrationError("calibration requires sequence length 1024")

    model_configs: dict[Architecture, ModelConfig] = {
        "dense": Olmo2Config(tokenizer_hash=build.tokenizer_hash),
        "mamba2": Mamba2Config(tokenizer_hash=build.tokenizer_hash),
        "diffusion": MaskedDiffusionConfig(tokenizer_hash=build.tokenizer_hash),
    }
    root = Path(artifact_root)
    cells: list[CalibrationCell] = []
    specs = _cell_specs(telemetry_interval_steps)
    for architecture in ARCHITECTURES:
        model = model_configs[architecture]
        for cell_id, stage, parent, settings in specs[architecture]:
            cells.append(
                CalibrationCell(
                    architecture=architecture,
                    cell_id=cell_id,
                    stage=stage,
                    parent_cell_id=parent,
                    settings=settings,
                    model_config_sha256=model.config_hash,
                    parameter_count=model.parameter_breakdown().total,
                    result_directory=(root / architecture / cell_id).as_posix(),
                )
            )
    return AccelerationCalibrationPlan(
        repository_revision=repository_revision,
        expected_cuda_device_name=expected_cuda_device_name,
        shard_manifest_sha256=_file_sha256(build_path),
        tokenizer_sha256=build.tokenizer_hash,
        tokenizer_vocab_size=16_000,
        seed=seed,
        sequence_length=sequence_length,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_optimizer_steps=warmup_optimizer_steps,
        measured_optimizer_steps=measured_optimizer_steps,
        minimum_measured_seconds=minimum_measured_seconds,
        repetitions=repetitions,
        telemetry_interval_steps=telemetry_interval_steps,
        promotion_minimum_speedup=promotion_minimum_speedup,
        maximum_loss_absolute_delta=maximum_loss_absolute_delta,
        maximum_gradient_absolute_delta=maximum_gradient_absolute_delta,
        maximum_update_absolute_delta=maximum_update_absolute_delta,
        artifact_root=root.as_posix(),
        cells=tuple(cells),
    )


def write_artifact(path: str | Path, artifact: CanonicalArtifact) -> None:
    """Atomically publish one canonical plan or report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise AccelerationCalibrationError("incomplete calibration artifact exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(artifact.canonical_json().encode())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_plan(path: str | Path, plan: AccelerationCalibrationPlan) -> None:
    """Atomically publish a calibration plan."""

    write_artifact(path, plan)


def write_report(path: str | Path, report: AccelerationCalibrationReport) -> None:
    """Atomically publish a derived calibration report."""

    write_artifact(path, report)


def write_numerical_trace(path: str | Path, trace: CalibrationNumericalTrace) -> None:
    """Atomically publish an untimed numerical-parity trace."""

    write_artifact(path, trace)


def load_plan(path: str | Path) -> AccelerationCalibrationPlan:
    """Load one canonical plan and reject noncanonical or malformed input."""

    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8").rstrip("\n")
        payload = json.loads(raw)
        plan = AccelerationCalibrationPlan.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise AccelerationCalibrationError("calibration plan is invalid") from error
    if raw != plan.canonical_json():
        raise AccelerationCalibrationError("calibration plan is not canonical")
    return plan


def load_results(directory: str | Path) -> tuple[CalibrationResult, ...]:
    """Load canonical per-repetition result artifacts in stable path order."""

    root = Path(directory)
    if not root.is_dir():
        raise AccelerationCalibrationError("calibration results directory is missing")
    results: list[CalibrationResult] = []
    for source in sorted(root.glob("*/*/repetition-*/result.json")):
        try:
            raw = source.read_text(encoding="utf-8").rstrip("\n")
            result = CalibrationResult.model_validate(json.loads(raw))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            raise AccelerationCalibrationError(
                "calibration result is invalid"
            ) from error
        if raw != result.canonical_json():
            raise AccelerationCalibrationError("calibration result is not canonical")
        results.append(result)
    if not results:
        raise AccelerationCalibrationError("calibration results directory is empty")
    return tuple(results)


def load_result(path: str | Path) -> CalibrationResult:
    """Load one canonical result artifact."""

    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8").rstrip("\n")
        result = CalibrationResult.model_validate(json.loads(raw))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise AccelerationCalibrationError("calibration result is invalid") from error
    if raw != result.canonical_json():
        raise AccelerationCalibrationError("calibration result is not canonical")
    return result


def load_numerical_trace(path: str | Path) -> CalibrationNumericalTrace:
    """Load one canonical numerical trace and reject malformed input."""

    source = Path(path)
    try:
        raw = source.read_text(encoding="utf-8").rstrip("\n")
        trace = CalibrationNumericalTrace.model_validate(json.loads(raw))
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise AccelerationCalibrationError(
            "calibration numerical trace is invalid"
        ) from error
    if raw != trace.canonical_json():
        raise AccelerationCalibrationError(
            "calibration numerical trace is not canonical"
        )
    return trace


def compare_numerical_traces(
    baseline: CalibrationNumericalTrace,
    candidate: CalibrationNumericalTrace,
) -> tuple[float, float, float]:
    """Return maximum absolute loss, gradient, and update deltas."""

    if baseline.cell_id != "baseline":
        raise AccelerationCalibrationError("parity reference is not a baseline trace")
    bindings = (
        "plan_sha256",
        "repository_revision",
        "architecture",
        "repetition",
    )
    if any(getattr(baseline, name) != getattr(candidate, name) for name in bindings):
        raise AccelerationCalibrationError("numerical trace bindings do not match")
    pairs = (
        (baseline.loss_values, candidate.loss_values),
        (baseline.gradient_norm_values, candidate.gradient_norm_values),
        (baseline.update_rms_values, candidate.update_rms_values),
    )
    if any(len(left) != len(right) for left, right in pairs):
        raise AccelerationCalibrationError("numerical trace lengths do not match")
    deltas = tuple(
        max(
            abs(left_value - right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
        for left, right in pairs
    )
    return deltas[0], deltas[1], deltas[2]


def resolve_cell(
    plan: AccelerationCalibrationPlan,
    architecture: Architecture,
    cell_id: str,
) -> CalibrationCell:
    """Resolve exactly one planned architecture/cell pair."""

    matches = [
        cell
        for cell in plan.cells
        if cell.architecture == architecture and cell.cell_id == cell_id
    ]
    if len(matches) != 1:
        raise AccelerationCalibrationError("calibration cell is not planned")
    return matches[0]


def _validate_result_binding(
    plan: AccelerationCalibrationPlan,
    cell: CalibrationCell,
    result: CalibrationResult,
) -> None:
    expected = {
        "plan_sha256": plan.artifact_sha256,
        "cell_sha256": cell.artifact_sha256,
        "repository_revision": plan.repository_revision,
        "cuda_device_name": plan.expected_cuda_device_name,
        "architecture": cell.architecture,
        "cell_id": cell.cell_id,
        "shard_manifest_sha256": plan.shard_manifest_sha256,
        "model_config_sha256": cell.model_config_sha256,
        "seed": plan.seed,
        "sequence_length": plan.sequence_length,
        "micro_batch_size": plan.micro_batch_size,
        "gradient_accumulation_steps": plan.gradient_accumulation_steps,
        "world_size": plan.world_size,
        "warmup_optimizer_steps": plan.warmup_optimizer_steps,
    }
    actual = {name: getattr(result, name) for name in expected}
    if actual != expected:
        raise AccelerationCalibrationError(
            f"result binding mismatch for {cell.architecture}/{cell.cell_id}"
        )
    if result.repetition > plan.repetitions:
        raise AccelerationCalibrationError("result repetition exceeds the plan")
    if (
        result.measured_optimizer_steps < plan.measured_optimizer_steps
        and result.measured_end_to_end_seconds < plan.minimum_measured_seconds
    ):
        raise AccelerationCalibrationError(
            "result measurement window is shorter than both planned alternatives"
        )
    if (
        cell.settings.sdpa_backend == "flash"
        and result.observed_sdpa_backend != "flash"
    ):
        raise AccelerationCalibrationError(
            "Flash-SDPA cell lacks profiler-confirmed Flash evidence"
        )


def build_report(
    plan: AccelerationCalibrationPlan,
    results: tuple[CalibrationResult, ...] | list[CalibrationResult],
) -> AccelerationCalibrationReport:
    """Validate complete evidence and derive median promotion decisions."""

    cells = {(cell.architecture, cell.cell_id): cell for cell in plan.cells}
    indexed: dict[tuple[Architecture, str, int], CalibrationResult] = {}
    for result in results:
        cell = cells.get((result.architecture, result.cell_id))
        if cell is None:
            raise AccelerationCalibrationError("result names an unplanned cell")
        _validate_result_binding(plan, cell, result)
        key = (result.architecture, result.cell_id, result.repetition)
        if key in indexed:
            raise AccelerationCalibrationError("duplicate calibration result")
        indexed[key] = result

    expected = {
        (cell.architecture, cell.cell_id, repetition)
        for cell in plan.cells
        for repetition in range(1, plan.repetitions + 1)
    }
    realized = set(indexed)
    if realized != expected:
        missing = sorted(expected - realized)
        extra = sorted(realized - expected)
        raise AccelerationCalibrationError(
            f"calibration evidence is incomplete; missing={missing!r}, extra={extra!r}"
        )

    for key, result in indexed.items():
        architecture, cell_id, repetition = key
        if cell_id == "baseline":
            continue
        baseline = indexed[(architecture, "baseline", repetition)]
        if result.baseline_result_sha256 != baseline.artifact_sha256:
            raise AccelerationCalibrationError(
                "candidate parity evidence references the wrong baseline result"
            )

    baseline_medians: dict[Architecture, float] = {}
    for architecture in ARCHITECTURES:
        baseline_medians[architecture] = float(
            median(
                indexed[(architecture, "baseline", repetition)].tokens_per_second
                for repetition in range(1, plan.repetitions + 1)
            )
        )

    summaries: list[CalibrationCellSummary] = []
    for cell in plan.cells:
        repetitions = [
            indexed[(cell.architecture, cell.cell_id, repetition)]
            for repetition in range(1, plan.repetitions + 1)
        ]
        median_throughput = float(
            median(item.tokens_per_second for item in repetitions)
        )
        baseline_throughput = baseline_medians[cell.architecture]
        loss_deltas = [item.maximum_loss_absolute_delta for item in repetitions]
        gradient_deltas = [item.maximum_gradient_absolute_delta for item in repetitions]
        update_deltas = [item.maximum_update_absolute_delta for item in repetitions]
        if cell.cell_id == "baseline":
            maximum_loss_delta = 0.0
            maximum_gradient_delta = 0.0
            maximum_update_delta = 0.0
        else:
            if any(
                item is None
                for item in [*loss_deltas, *gradient_deltas, *update_deltas]
            ):
                raise AccelerationCalibrationError(
                    "candidate result is missing numerical parity evidence"
                )
            maximum_loss_delta = max(item for item in loss_deltas if item is not None)
            maximum_gradient_delta = max(
                item for item in gradient_deltas if item is not None
            )
            maximum_update_delta = max(
                item for item in update_deltas if item is not None
            )
        parity = (
            maximum_loss_delta <= plan.maximum_loss_absolute_delta
            and maximum_gradient_delta <= plan.maximum_gradient_absolute_delta
            and maximum_update_delta <= plan.maximum_update_absolute_delta
        )
        speedup = median_throughput / baseline_throughput
        summaries.append(
            CalibrationCellSummary(
                architecture=cell.architecture,
                cell_id=cell.cell_id,
                median_tokens_per_second=median_throughput,
                baseline_median_tokens_per_second=baseline_throughput,
                speedup_over_baseline=speedup,
                maximum_loss_absolute_delta=maximum_loss_delta,
                maximum_gradient_absolute_delta=maximum_gradient_delta,
                maximum_update_absolute_delta=maximum_update_delta,
                numerical_parity_passed=parity,
                promoted=(
                    cell.cell_id != "baseline"
                    and parity
                    and speedup >= plan.promotion_minimum_speedup
                ),
            )
        )
    return AccelerationCalibrationReport(
        plan_sha256=plan.artifact_sha256,
        result_count=len(results),
        promotion_minimum_speedup=plan.promotion_minimum_speedup,
        summaries=tuple(summaries),
    )


# Descriptive aliases for callers that prefer module-qualified names.
create_acceleration_calibration_plan = create_plan
write_acceleration_calibration_plan = write_plan
