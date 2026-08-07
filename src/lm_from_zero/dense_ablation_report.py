"""Aggregate completed Milestone-8 dense-ablation artifacts."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from statistics import fmean, stdev
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lm_from_zero.dense_ablations import (
    DenseAblationPlan,
    DenseModelVariant,
    OptimizerVariant,
    VariantChange,
    VariantName,
)
from lm_from_zero.evaluation import CausalEvaluationResult
from lm_from_zero.generation.causal import CausalGenerationRecord
from lm_from_zero.training import (
    DenseRunPlan,
    DenseTrainingResult,
    OptimizerMetricRecord,
    validate_checkpoint,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
GitRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class DenseAblationReportError(RuntimeError):
    """Raised when M8 artifacts cannot satisfy the aggregation contract."""


class MetricSummary(BaseModel):
    """Mean and sample standard deviation across the declared seeds."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    mean: float = Field(ge=0)
    std: float = Field(ge=0)


class DenseAblationSeedResult(BaseModel):
    """Validated terminal evidence for one clean M8 job."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    variant: VariantName
    change: VariantChange
    model_variant: DenseModelVariant
    optimizer_variant: OptimizerVariant
    variant_spec_sha256: Sha256
    seed: int
    parameter_count: Annotated[int, Field(gt=0)]
    optimizer_step: Annotated[int, Field(gt=0)]
    tokens_consumed: Annotated[int, Field(gt=0)]
    terminal_loss: float = Field(ge=0)
    terminal_gradient_norm: float = Field(ge=0)
    terminal_tokens_per_second: float = Field(gt=0)
    peak_cuda_memory_reserved_bytes: Annotated[int | None, Field(gt=0)] = None
    training_config_sha256: Sha256
    model_config_sha256: Sha256
    shard_manifest_sha256: Sha256
    tokenizer_sha256: Sha256
    git_revision: GitRevision
    git_dirty: bool
    checkpoint_id: str
    checkpoint_manifest_sha256: Sha256
    model_artifact_sha256: Sha256
    recovery_artifact_sha256: Sha256
    events_sha256: Sha256
    artifact_directory: str
    checkpoint_directory: str
    events_jsonl: str


class DenseAblationVariantSummary(BaseModel):
    """Seed aggregate for one declared M8 variant."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    variant: VariantName
    change: VariantChange
    model_variant: DenseModelVariant
    optimizer_variant: OptimizerVariant
    variant_spec_sha256: Sha256
    seeds: tuple[int, ...]
    parameter_count: Annotated[int, Field(gt=0)]
    optimizer_step: Annotated[int, Field(gt=0)]
    tokens_consumed: Annotated[int, Field(gt=0)]
    terminal_loss: MetricSummary
    terminal_gradient_norm: MetricSummary
    terminal_tokens_per_second: MetricSummary
    peak_cuda_memory_reserved_bytes: MetricSummary | None
    terminal_loss_rank: Annotated[int, Field(gt=0)]
    throughput_rank: Annotated[int, Field(gt=0)]


class DenseAblationReport(BaseModel):
    """Canonical, portable comparison of the completed M8 variant matrix."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-dense-ablation-report"] = (
        "lm-from-zero-dense-ablation-report"
    )
    format_version: Literal[1] = 1
    plan_sha256: Sha256
    plan_format: Literal["lm-from-zero-dense-ablations-plan"]
    artifact_root: str
    source_revision: GitRevision
    source_dirty: bool
    job_count: Literal[21] = 21
    jobs: tuple[DenseAblationSeedResult, ...]
    variants: tuple[DenseAblationVariantSummary, ...]
    finalists_by_terminal_loss: tuple[VariantName, ...]
    fastest_variant: VariantName
    recommended_variants: tuple[VariantName, ...]
    selection_rule: str

    def canonical_bytes(self) -> bytes:
        """Return the stable serialized report."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


class DenseAblationDownstreamRecord(BaseModel):
    """Evaluation and generation evidence bound to one M8 checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: VariantName
    model_variant: DenseModelVariant
    optimizer_variant: OptimizerVariant
    seed: int
    checkpoint_id: str
    checkpoint_manifest_sha256: Sha256
    git_revision: GitRevision
    git_dirty: bool
    evaluation_jsonl: str
    generation_jsonl: str
    evaluation: CausalEvaluationResult
    generation: CausalGenerationRecord


class DenseAblationDownstreamReport(BaseModel):
    """Canonical downstream evidence for the selected M8 variants."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-dense-ablation-downstream-report"] = (
        "lm-from-zero-dense-ablation-downstream-report"
    )
    format_version: Literal[1] = 1
    ablation_report_sha256: Sha256
    plan_sha256: Sha256
    source_revision: GitRevision
    source_dirty: bool
    seed: int
    validation_batch_count: Literal[24] = 24
    validation_sequence_count: Literal[192] = 192
    validation_predicted_token_count: Literal[196_416] = 196_416
    generation_prompt: Literal["Once upon a time"] = "Once upon a time"
    generation_token_count: Literal[16] = 16
    records: tuple[DenseAblationDownstreamRecord, ...]

    def canonical_bytes(self) -> bytes:
        """Return the stable serialized downstream report."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise DenseAblationReportError(f"cannot hash artifact: {path}") from error
    return digest.hexdigest()


def _read_text(path: Path) -> str:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise DenseAblationReportError(f"cannot read artifact: {path}") from error
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
    try:
        return raw.decode(encoding)
    except UnicodeDecodeError as error:
        raise DenseAblationReportError(f"artifact is not valid text: {path}") from error


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _load_events(path: Path) -> tuple[dict[str, Any], ...]:
    try:
        lines = _read_text(path).splitlines()
    except DenseAblationReportError as error:
        raise DenseAblationReportError(f"cannot read event log: {path}") from error
    if not lines:
        raise DenseAblationReportError(f"event log is empty: {path}")
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise DenseAblationReportError(f"event log is invalid: {path}") from error
        if not isinstance(payload, dict) or line != _canonical_json(payload):
            raise DenseAblationReportError(f"event log is not canonical: {path}")
        events.append(payload)
    return tuple(events)


def _load_run_outputs(path: Path) -> tuple[DenseRunPlan, DenseTrainingResult]:
    try:
        lines = _read_text(path).splitlines()
    except DenseAblationReportError as error:
        raise DenseAblationReportError(f"cannot read run output: {path}") from error
    if len(lines) != 2:
        raise DenseAblationReportError(
            f"run output must contain one plan and one result: {path}"
        )
    try:
        return (
            DenseRunPlan.model_validate_json(lines[0]),
            DenseTrainingResult.model_validate_json(lines[1]),
        )
    except ValueError as error:
        raise DenseAblationReportError(f"run output is invalid: {path}") from error


def _load_canonical_line(path: Path) -> str:
    lines = _read_text(path).splitlines()
    if len(lines) != 1:
        raise DenseAblationReportError(f"evidence must contain one record: {path}")
    return lines[0]


def _load_evaluation(path: Path) -> CausalEvaluationResult:
    line = _load_canonical_line(path)
    try:
        result = CausalEvaluationResult.model_validate_json(line)
    except ValueError as error:
        raise DenseAblationReportError(
            f"evaluation evidence is invalid: {path}"
        ) from error
    if line != result.canonical_json():
        raise DenseAblationReportError(f"evaluation evidence is not canonical: {path}")
    return result


def _load_generation(path: Path) -> CausalGenerationRecord:
    line = _load_canonical_line(path)
    try:
        record = CausalGenerationRecord.model_validate_json(line)
    except ValueError as error:
        raise DenseAblationReportError(
            f"generation evidence is invalid: {path}"
        ) from error
    if line != record.canonical_json():
        raise DenseAblationReportError(f"generation evidence is not canonical: {path}")
    return record


def _metric_summary(values: list[float]) -> MetricSummary:
    if not values:
        raise DenseAblationReportError("cannot summarize an empty metric set")
    return MetricSummary(
        mean=fmean(values),
        std=stdev(values) if len(values) > 1 else 0.0,
    )


def _path_text(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_seed_result(
    *,
    plan: DenseAblationPlan,
    job: Any,
    variant_spec: Any,
    artifact_root: Path,
) -> DenseAblationSeedResult:
    job_root = artifact_root / job.variant / f"seed-{job.seed}"
    events_path = job_root / "events.jsonl"
    run_output_path = job_root / "result.jsonl"
    checkpoint_path = (
        job_root / "checkpoints" / f"step-{job.expected_optimizer_step:012d}"
    )
    events = _load_events(events_path)
    run_plan, run_result = _load_run_outputs(run_output_path)
    try:
        checkpoint = validate_checkpoint(checkpoint_path)
    except Exception as error:
        raise DenseAblationReportError(
            f"checkpoint validation failed for {job.variant}/seed-{job.seed}"
        ) from error

    starts = [event for event in events if event.get("event") == "run_start"]
    resumes = [event for event in events if event.get("event") == "run_resume"]
    completes = [event for event in events if event.get("event") == "run_complete"]
    if len(starts) != 1 or resumes or len(completes) != 1:
        raise DenseAblationReportError(
            f"run lifecycle is invalid for {job.variant}/seed-{job.seed}"
        )
    start = starts[0]
    start_config = start.get("training_config")
    if (
        start.get("optimizer_step") != 0
        or not isinstance(start_config, dict)
        or start.get("training_config_sha256") != run_plan.training_config_sha256
        or start_config.get("seed") != job.seed
        or start_config.get("model_variant", "baseline") != job.model_variant
        or start_config.get("optimizer_variant", "adamw") != job.optimizer_variant
    ):
        raise DenseAblationReportError(
            f"run-start metadata disagrees with plan for {job.variant}/seed-{job.seed}"
        )

    try:
        metrics = tuple(
            OptimizerMetricRecord.model_validate(event)
            for event in events
            if event.get("event") == "optimizer_step"
        )
    except ValueError as error:
        raise DenseAblationReportError(
            f"optimizer metrics are invalid for {job.variant}/seed-{job.seed}"
        ) from error
    if not metrics or [item.optimizer_step for item in metrics] != sorted(
        item.optimizer_step for item in metrics
    ):
        raise DenseAblationReportError(
            f"optimizer metrics are not ordered for {job.variant}/seed-{job.seed}"
        )
    terminal = metrics[-1]
    complete = completes[0]
    if (
        terminal.optimizer_step != job.expected_optimizer_step
        or terminal.tokens_consumed != job.expected_tokens
        or complete.get("optimizer_step") != terminal.optimizer_step
        or complete.get("tokens_consumed") != terminal.tokens_consumed
        or run_plan.optimizer_steps != job.expected_optimizer_step
        or run_plan.total_training_tokens != job.expected_tokens
        or run_result.optimizer_step != terminal.optimizer_step
        or run_result.cursor.tokens_consumed != terminal.tokens_consumed
        or checkpoint.progress.optimizer_step != terminal.optimizer_step
        or checkpoint.progress.tokens_consumed != terminal.tokens_consumed
    ):
        raise DenseAblationReportError(
            f"terminal boundaries disagree for {job.variant}/seed-{job.seed}"
        )

    binding = checkpoint.binding
    if (
        job.variant_spec_sha256 != variant_spec.spec_sha256
        or run_plan.seed != job.seed
        or run_plan.model_variant != job.model_variant
        or run_plan.optimizer_variant != job.optimizer_variant
        or run_plan.parameter_count <= 0
        or binding.model_config_sha256 != run_plan.model_config_sha256
        or binding.shard_manifest_sha256 != run_plan.shard_manifest_sha256
        or binding.tokenizer_sha256 != run_plan.tokenizer_sha256
        or checkpoint.lineage.checkpoint_id
        != f"step-{job.expected_optimizer_step:012d}"
    ):
        raise DenseAblationReportError(
            f"binding metadata disagrees with plan for {job.variant}/seed-{job.seed}"
        )
    if checkpoint.binding.git.dirty is not False:
        raise DenseAblationReportError(
            f"source metadata is not clean for {job.variant}/seed-{job.seed}"
        )

    return DenseAblationSeedResult(
        variant=job.variant,
        change=variant_spec.change,
        model_variant=job.model_variant,
        optimizer_variant=job.optimizer_variant,
        variant_spec_sha256=job.variant_spec_sha256,
        seed=job.seed,
        parameter_count=run_plan.parameter_count,
        optimizer_step=terminal.optimizer_step,
        tokens_consumed=terminal.tokens_consumed,
        terminal_loss=terminal.loss,
        terminal_gradient_norm=terminal.gradient_norm,
        terminal_tokens_per_second=terminal.tokens_per_second,
        peak_cuda_memory_reserved_bytes=terminal.peak_cuda_memory_reserved_bytes,
        training_config_sha256=run_plan.training_config_sha256,
        model_config_sha256=run_plan.model_config_sha256,
        shard_manifest_sha256=run_plan.shard_manifest_sha256,
        tokenizer_sha256=run_plan.tokenizer_sha256,
        git_revision=checkpoint.binding.git.revision,
        git_dirty=checkpoint.binding.git.dirty,
        checkpoint_id=checkpoint.lineage.checkpoint_id,
        checkpoint_manifest_sha256=_sha256_file(checkpoint_path / "manifest.json"),
        model_artifact_sha256=checkpoint.model_artifact.sha256,
        recovery_artifact_sha256=checkpoint.recovery_artifact.sha256,
        events_sha256=_sha256_file(events_path),
        artifact_directory=_path_text(job_root),
        checkpoint_directory=_path_text(checkpoint_path),
        events_jsonl=_path_text(events_path),
    )


def build_dense_ablation_report(
    plan_path: str | Path,
    artifact_root: str | Path,
) -> DenseAblationReport:
    """Validate and aggregate the completed 21-job M8 variant matrix."""

    plan_file = Path(plan_path)
    artifact_directory = Path(artifact_root)
    try:
        plan = DenseAblationPlan.model_validate_json(_read_text(plan_file))
    except (DenseAblationReportError, ValueError) as error:
        raise DenseAblationReportError("cannot load the M8 plan") from error
    variants = {item.name: item for item in plan.variants if item.name != "baseline"}
    jobs = [item for item in plan.jobs if item.variant != "baseline"]
    if len(variants) != 7 or len(jobs) != 21:
        raise DenseAblationReportError("the M8 plan does not describe 21 variants")
    if not artifact_directory.is_dir():
        raise DenseAblationReportError("the M8 artifact root does not exist")

    results_list: list[DenseAblationSeedResult] = []
    for job in sorted(jobs, key=lambda item: (item.variant, item.seed)):
        if job.variant == "baseline":
            raise DenseAblationReportError("the M8 plan contains a baseline variant")
        variant_spec = variants.get(job.variant)
        if variant_spec is None:
            raise DenseAblationReportError("the M8 plan contains an unknown variant")
        results_list.append(
            _load_seed_result(
                plan=plan,
                job=job,
                variant_spec=variant_spec,
                artifact_root=artifact_directory,
            )
        )
    results = tuple(results_list)
    if len({item.git_revision for item in results}) != 1:
        raise DenseAblationReportError("M8 source revisions disagree")
    if len({item.git_dirty for item in results}) != 1:
        raise DenseAblationReportError("M8 source cleanliness disagrees")

    summaries_by_variant: dict[str, DenseAblationVariantSummary] = {}
    for name, spec in variants.items():
        selected = [item for item in results if item.variant == name]
        if len(selected) != 3:
            raise DenseAblationReportError(
                f"M8 variant does not have three seeds: {name}"
            )
        if len({item.parameter_count for item in selected}) != 1:
            raise DenseAblationReportError(f"parameter counts disagree for {name}")
        if len({item.optimizer_step for item in selected}) != 1:
            raise DenseAblationReportError(f"step boundaries disagree for {name}")
        if len({item.tokens_consumed for item in selected}) != 1:
            raise DenseAblationReportError(f"token boundaries disagree for {name}")
        memory = [
            float(item.peak_cuda_memory_reserved_bytes)
            for item in selected
            if item.peak_cuda_memory_reserved_bytes is not None
        ]
        summaries_by_variant[name] = DenseAblationVariantSummary(
            variant=name,
            change=spec.change,
            model_variant=spec.model_variant,
            optimizer_variant=spec.optimizer_variant,
            variant_spec_sha256=spec.spec_sha256,
            seeds=tuple(item.seed for item in selected),
            parameter_count=selected[0].parameter_count,
            optimizer_step=selected[0].optimizer_step,
            tokens_consumed=selected[0].tokens_consumed,
            terminal_loss=_metric_summary([item.terminal_loss for item in selected]),
            terminal_gradient_norm=_metric_summary(
                [item.terminal_gradient_norm for item in selected]
            ),
            terminal_tokens_per_second=_metric_summary(
                [item.terminal_tokens_per_second for item in selected]
            ),
            peak_cuda_memory_reserved_bytes=(
                _metric_summary(memory) if len(memory) == len(selected) else None
            ),
            terminal_loss_rank=1,
            throughput_rank=1,
        )

    loss_order = sorted(
        summaries_by_variant.values(),
        key=lambda item: (item.terminal_loss.mean, item.variant),
    )
    throughput_order = sorted(
        summaries_by_variant.values(),
        key=lambda item: (-item.terminal_tokens_per_second.mean, item.variant),
    )
    for rank, item in enumerate(loss_order, start=1):
        summaries_by_variant[item.variant] = item.model_copy(
            update={"terminal_loss_rank": rank}
        )
    for rank, item in enumerate(throughput_order, start=1):
        current = summaries_by_variant[item.variant]
        summaries_by_variant[item.variant] = current.model_copy(
            update={"throughput_rank": rank}
        )
    summaries = tuple(
        summaries_by_variant[name] for name in sorted(summaries_by_variant)
    )
    finalists = tuple(item.variant for item in loss_order[:3])
    fastest = throughput_order[0].variant
    recommended = tuple(
        item.variant for item in loss_order if item.variant in {*finalists, fastest}
    )
    return DenseAblationReport(
        plan_sha256=_sha256_file(plan_file),
        plan_format=plan.format,
        artifact_root=_path_text(artifact_directory),
        source_revision=results[0].git_revision,
        source_dirty=results[0].git_dirty,
        jobs=results,
        variants=summaries,
        finalists_by_terminal_loss=finalists,
        fastest_variant=fastest,
        recommended_variants=recommended,
        selection_rule=(
            "report the three lowest mean terminal losses and the fastest variant; "
            "evaluate the union downstream"
        ),
    )


def write_dense_ablation_report(
    path: str | Path,
    report: DenseAblationReport,
) -> None:
    """Atomically write one canonical M8 report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise DenseAblationReportError("incomplete dense-ablation report exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(report.canonical_bytes() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_dense_ablation_downstream_report(
    ablation_report_path: str | Path,
    evaluation_root: str | Path,
    generation_root: str | Path,
    *,
    seed: int,
) -> DenseAblationDownstreamReport:
    """Bind selected M8 validation and generation records to checkpoints."""

    report_path = Path(ablation_report_path)
    try:
        ablation_report = DenseAblationReport.model_validate_json(
            _read_text(report_path)
        )
    except (DenseAblationReportError, ValueError) as error:
        raise DenseAblationReportError(
            "cannot load the M8 aggregation report"
        ) from error
    if ablation_report.source_dirty:
        raise DenseAblationReportError(
            "M8 aggregation report has dirty source metadata"
        )

    evaluation_directory = Path(evaluation_root)
    generation_directory = Path(generation_root)
    jobs: list[DenseAblationDownstreamRecord] = []
    for variant in ablation_report.recommended_variants:
        matches = [
            item
            for item in ablation_report.jobs
            if item.variant == variant and item.seed == seed
        ]
        if len(matches) != 1:
            raise DenseAblationReportError(
                "M8 aggregation report has no unique selected job: "
                f"{variant}/seed-{seed}"
            )
        job = matches[0]
        slug = variant.replace("_", "-")
        evaluation_path = evaluation_directory / f"m8-{slug}-{seed}-v24.jsonl"
        generation_path = generation_directory / f"m8-{slug}-{seed}.jsonl"
        evaluation = _load_evaluation(evaluation_path)
        generation = _load_generation(generation_path)
        if (
            evaluation.split != "validation"
            or evaluation.batch_count != 24
            or evaluation.sequence_count != 192
            or evaluation.predicted_token_count != 196_416
            or evaluation.model_config_sha256 != job.model_config_sha256
            or evaluation.shard_manifest_sha256 != job.shard_manifest_sha256
            or evaluation.tokenizer_sha256 != job.tokenizer_sha256
            or generation.model_config_sha256 != job.model_config_sha256
            or generation.tokenizer_sha256 != job.tokenizer_sha256
            or generation.result.generated_token_count != 16
            or generation.result.model_forwards != 16
        ):
            raise DenseAblationReportError(
                f"downstream evidence does not bind to {variant}/seed-{seed}"
            )
        jobs.append(
            DenseAblationDownstreamRecord(
                variant=job.variant,
                model_variant=job.model_variant,
                optimizer_variant=job.optimizer_variant,
                seed=job.seed,
                checkpoint_id=job.checkpoint_id,
                checkpoint_manifest_sha256=job.checkpoint_manifest_sha256,
                git_revision=job.git_revision,
                git_dirty=job.git_dirty,
                evaluation_jsonl=_path_text(evaluation_path),
                generation_jsonl=_path_text(generation_path),
                evaluation=evaluation,
                generation=generation,
            )
        )
    return DenseAblationDownstreamReport(
        ablation_report_sha256=_sha256_file(report_path),
        plan_sha256=ablation_report.plan_sha256,
        source_revision=ablation_report.source_revision,
        source_dirty=ablation_report.source_dirty,
        seed=seed,
        records=tuple(jobs),
    )


def write_dense_ablation_downstream_report(
    path: str | Path,
    report: DenseAblationDownstreamReport,
) -> None:
    """Atomically write one canonical downstream report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise DenseAblationReportError("incomplete downstream report exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(report.canonical_bytes() + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
