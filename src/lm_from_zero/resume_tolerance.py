"""Generate compiled-CUDA interrupted-vs-uninterrupted tolerance evidence."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file

from lm_from_zero.training import DenseTrainingConfig, validate_checkpoint
from lm_from_zero.training.metrics import load_optimizer_metrics

SHA256_PATTERN = r"^[0-9a-f]{64}$"
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]


class ResumeToleranceError(RuntimeError):
    """Raised when comparison inputs are incomplete or incompatible."""


class ResumeToleranceThresholds(BaseModel):
    """Numerical acceptance thresholds declared before comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    parameter_atol: Annotated[float, Field(gt=0)] = 1e-5
    parameter_rtol: Annotated[float, Field(gt=0)] = 1e-4
    loss_atol: Annotated[float, Field(gt=0)] = 1e-4


class StepLossDifference(BaseModel):
    """Loss difference at one common optimizer step."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    optimizer_step: Annotated[int, Field(gt=0)]
    uninterrupted_loss: Annotated[float, Field(ge=0)]
    resumed_loss: Annotated[float, Field(ge=0)]
    absolute_difference: Annotated[float, Field(ge=0)]


class DenseResumeToleranceReport(BaseModel):
    """Portable evidence for one compiled bf16 CUDA resume comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-dense-resume-tolerance"] = (
        "lm-from-zero-dense-resume-tolerance"
    )
    format_version: Literal[1] = 1
    recorded_at_utc: datetime
    source_git_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    training_config_sha256: Sha256
    model_config_sha256: Sha256
    shard_manifest_sha256: Sha256
    tokenizer_sha256: Sha256
    uninterrupted_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    uninterrupted_checkpoint_manifest_sha256: Sha256
    uninterrupted_model_artifact_sha256: Sha256
    resumed_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    resumed_checkpoint_manifest_sha256: Sha256
    resumed_model_artifact_sha256: Sha256
    resumed_parent_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    optimizer_steps: Annotated[int, Field(gt=0)]
    final_tokens_consumed: Annotated[int, Field(gt=0)]
    device: Literal["cuda"]
    precision: Literal["bf16"]
    compile_model: Literal[True]
    thresholds: ResumeToleranceThresholds
    compared_tensors: Annotated[int, Field(gt=0)]
    compared_values: Annotated[int, Field(gt=0)]
    exact_equal_values: Annotated[int, Field(ge=0)]
    tolerance_violation_values: Annotated[int, Field(ge=0)]
    tensors_with_tolerance_violations: tuple[str, ...]
    parameter_max_absolute_error: Annotated[float, Field(ge=0)]
    parameter_max_tolerance_ratio: Annotated[float, Field(ge=0)]
    step_loss_differences: tuple[StepLossDifference, ...]
    loss_max_absolute_error: Annotated[float, Field(ge=0)]
    passed: bool

    def canonical_bytes(self) -> bytes:
        """Return deterministic JSON bytes."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


def _canonical_events(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ResumeToleranceError("cannot read training evidence") from error
    if not lines:
        raise ResumeToleranceError("training evidence is empty")
    events: list[dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ResumeToleranceError("training evidence is invalid") from error
        if not isinstance(payload, dict):
            raise ResumeToleranceError("training evidence record is not an object")
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        if line != canonical:
            raise ResumeToleranceError("training evidence is not canonical")
        events.append(payload)
    return events


def _training_contract(
    path: Path,
    *,
    require_resume: bool,
) -> tuple[str, DenseTrainingConfig, int | None]:
    events = _canonical_events(path)
    starts = [event for event in events if event.get("event") == "run_start"]
    resumes = [event for event in events if event.get("event") == "run_resume"]
    expected_resumes = 1 if require_resume else 0
    if len(starts) != 1 or len(resumes) != expected_resumes:
        description = "resumed" if require_resume else "uninterrupted"
        raise ResumeToleranceError(f"{description} run history is invalid")
    hashes = {event.get("training_config_sha256") for event in [*starts, *resumes]}
    if len(hashes) != 1:
        raise ResumeToleranceError("training configuration changed across resume")
    training_hash = hashes.pop()
    if not isinstance(training_hash, str):
        raise ResumeToleranceError("training configuration evidence is missing")
    if re.fullmatch(SHA256_PATTERN, training_hash) is None:
        raise ResumeToleranceError("training configuration hash is invalid")
    configs: list[DenseTrainingConfig] = []
    for event in [*starts, *resumes]:
        config_payload = event.get("training_config")
        if not isinstance(config_payload, dict):
            raise ResumeToleranceError("training configuration evidence is missing")
        try:
            candidate_config = DenseTrainingConfig.model_validate(config_payload)
        except ValueError as error:
            raise ResumeToleranceError(
                "training configuration evidence is invalid"
            ) from error
        if candidate_config.config_hash != training_hash:
            raise ResumeToleranceError("training configuration hash does not match")
        configs.append(candidate_config)
    config = configs[0]
    if any(candidate != config for candidate in configs[1:]):
        raise ResumeToleranceError("training configuration changed across resume")
    if (
        config.device != "cuda"
        or config.precision != "bf16"
        or config.compile_model is not True
        or config.batch.rank != 0
        or config.batch.world_size != 1
    ):
        raise ResumeToleranceError(
            "comparison requires single-GPU compiled bf16 CUDA runs"
        )
    if starts[0].get("optimizer_step") != 0:
        raise ResumeToleranceError("training run does not start at step zero")
    resume_step: int | None = None
    if resumes:
        resume_candidate = resumes[0].get("optimizer_step")
        if (
            not isinstance(resume_candidate, int)
            or isinstance(resume_candidate, bool)
            or resume_candidate <= 0
        ):
            raise ResumeToleranceError("resume optimizer step is invalid")
        resume_step = resume_candidate
    return training_hash, config, resume_step


def _checkpoint_sha256(manifest: Any) -> str:
    return sha256(manifest.canonical_bytes()).hexdigest()


def _compare_parameters(
    uninterrupted: Path,
    resumed: Path,
    thresholds: ResumeToleranceThresholds,
) -> tuple[int, int, int, int, tuple[str, ...], float, float]:
    reference = load_file(str(uninterrupted), device="cpu")
    candidate = load_file(str(resumed), device="cpu")
    if reference.keys() != candidate.keys():
        raise ResumeToleranceError("checkpoint model tensor names disagree")
    compared_values = 0
    exact_values = 0
    violation_values = 0
    violating_tensors: list[str] = []
    maximum_absolute = 0.0
    maximum_ratio = 0.0
    for name in sorted(reference):
        expected = reference[name]
        actual = candidate[name]
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise ResumeToleranceError(f"checkpoint model tensor disagrees: {name}")
        expected_float = expected.to(dtype=torch.float64)
        actual_float = actual.to(dtype=torch.float64)
        difference = (expected_float - actual_float).abs()
        allowed = (
            thresholds.parameter_atol + thresholds.parameter_rtol * expected_float.abs()
        )
        violations = difference > allowed
        values = difference.numel()
        compared_values += values
        exact_values += int(torch.count_nonzero(difference == 0).item())
        tensor_violations = int(torch.count_nonzero(violations).item())
        violation_values += tensor_violations
        if tensor_violations:
            violating_tensors.append(name)
        if values:
            maximum_absolute = max(
                maximum_absolute,
                float(difference.max().item()),
            )
            maximum_ratio = max(
                maximum_ratio,
                float((difference / allowed).max().item()),
            )
    return (
        len(reference),
        compared_values,
        exact_values,
        violation_values,
        tuple(violating_tensors),
        maximum_absolute,
        maximum_ratio,
    )


def build_dense_resume_tolerance_report(
    *,
    uninterrupted_jsonl: str | Path,
    uninterrupted_checkpoint: str | Path,
    resumed_jsonl: str | Path,
    resumed_checkpoint: str | Path,
    thresholds: ResumeToleranceThresholds | None = None,
) -> DenseResumeToleranceReport:
    """Validate two run histories and compare their final model tensors."""

    accepted = ResumeToleranceThresholds() if thresholds is None else thresholds
    uninterrupted_log = Path(uninterrupted_jsonl)
    resumed_log = Path(resumed_jsonl)
    uninterrupted_hash, uninterrupted_config, uninterrupted_resume_step = (
        _training_contract(
            uninterrupted_log,
            require_resume=False,
        )
    )
    resumed_hash, resumed_config, resumed_resume_step = _training_contract(
        resumed_log,
        require_resume=True,
    )
    if uninterrupted_resume_step is not None or resumed_resume_step is None:
        raise ResumeToleranceError("comparison resume history is invalid")
    if uninterrupted_hash != resumed_hash or uninterrupted_config != resumed_config:
        raise ResumeToleranceError("compared training configurations disagree")

    uninterrupted_manifest = validate_checkpoint(uninterrupted_checkpoint)
    resumed_manifest = validate_checkpoint(resumed_checkpoint)
    bindings = (uninterrupted_manifest.binding, resumed_manifest.binding)
    if any(binding.git.dirty for binding in bindings):
        raise ResumeToleranceError("comparison checkpoint has a dirty source")
    if bindings[0].git.revision != bindings[1].git.revision:
        raise ResumeToleranceError("comparison source revisions disagree")
    binding_fields = (
        "model_config_sha256",
        "tokenizer_sha256",
        "shard_manifest_sha256",
        "rank",
        "world_size",
    )
    if any(
        getattr(bindings[0], field) != getattr(bindings[1], field)
        for field in binding_fields
    ):
        raise ResumeToleranceError("comparison checkpoint bindings disagree")
    if (
        bindings[0].rank != 0
        or bindings[0].world_size != 1
        or any(
            not binding.runtime.cuda_available or binding.runtime.cuda_version is None
            for binding in bindings
        )
    ):
        raise ResumeToleranceError("comparison checkpoints lack single-GPU CUDA")
    runtime_fields = (
        "python_version",
        "operating_system",
        "machine",
        "torch_version",
        "cuda_version",
        "cuda_available",
        "cuda_device_names",
        "dependency_versions",
    )
    if any(
        getattr(bindings[0].runtime, field) != getattr(bindings[1].runtime, field)
        for field in runtime_fields
    ):
        raise ResumeToleranceError("comparison checkpoint runtimes disagree")

    uninterrupted_progress = uninterrupted_manifest.progress
    resumed_progress = resumed_manifest.progress
    if (
        uninterrupted_progress.optimizer_step != resumed_progress.optimizer_step
        or uninterrupted_progress.tokens_consumed != resumed_progress.tokens_consumed
    ):
        raise ResumeToleranceError("comparison checkpoint progress disagrees")
    final_step = uninterrupted_progress.optimizer_step
    if final_step <= 0:
        raise ResumeToleranceError("comparison checkpoint has no optimizer steps")
    resumed_parent = resumed_manifest.lineage.parent_checkpoint_id
    if resumed_parent is None:
        raise ResumeToleranceError("resumed checkpoint lacks parent lineage")
    parent_step = int(resumed_parent.removeprefix("step-"))
    if parent_step >= final_step:
        raise ResumeToleranceError("resumed checkpoint parent is not earlier")
    if parent_step != resumed_resume_step:
        raise ResumeToleranceError("resume event and checkpoint lineage disagree")

    uninterrupted_metrics = load_optimizer_metrics(uninterrupted_log)
    resumed_metrics = load_optimizer_metrics(resumed_log)
    expected_steps = list(range(1, final_step + 1))
    if [
        record.optimizer_step for record in uninterrupted_metrics
    ] != expected_steps or [
        record.optimizer_step for record in resumed_metrics
    ] != expected_steps:
        raise ResumeToleranceError("comparison step metrics are incomplete")
    if (
        uninterrupted_metrics[-1].tokens_consumed
        != uninterrupted_progress.tokens_consumed
        or resumed_metrics[-1].tokens_consumed != resumed_progress.tokens_consumed
    ):
        raise ResumeToleranceError("comparison metric tokens disagree")
    loss_differences = tuple(
        StepLossDifference(
            optimizer_step=expected.optimizer_step,
            uninterrupted_loss=expected.loss,
            resumed_loss=actual.loss,
            absolute_difference=abs(expected.loss - actual.loss),
        )
        for expected, actual in zip(
            uninterrupted_metrics,
            resumed_metrics,
            strict=True,
        )
    )
    maximum_loss = max(item.absolute_difference for item in loss_differences)

    uninterrupted_directory = Path(uninterrupted_checkpoint)
    resumed_directory = Path(resumed_checkpoint)
    comparison = _compare_parameters(
        uninterrupted_directory / uninterrupted_manifest.model_artifact.filename,
        resumed_directory / resumed_manifest.model_artifact.filename,
        accepted,
    )
    (
        tensor_count,
        value_count,
        exact_count,
        violation_count,
        violating_tensors,
        maximum_parameter_error,
        maximum_tolerance_ratio,
    ) = comparison
    passed = violation_count == 0 and maximum_loss <= accepted.loss_atol
    return DenseResumeToleranceReport(
        recorded_at_utc=max(
            uninterrupted_manifest.created_at_utc,
            resumed_manifest.created_at_utc,
        ),
        source_git_revision=bindings[0].git.revision,
        training_config_sha256=uninterrupted_hash,
        model_config_sha256=bindings[0].model_config_sha256,
        shard_manifest_sha256=bindings[0].shard_manifest_sha256,
        tokenizer_sha256=bindings[0].tokenizer_sha256,
        uninterrupted_checkpoint_id=(uninterrupted_manifest.lineage.checkpoint_id),
        uninterrupted_checkpoint_manifest_sha256=_checkpoint_sha256(
            uninterrupted_manifest
        ),
        uninterrupted_model_artifact_sha256=(
            uninterrupted_manifest.model_artifact.sha256
        ),
        resumed_checkpoint_id=resumed_manifest.lineage.checkpoint_id,
        resumed_checkpoint_manifest_sha256=_checkpoint_sha256(resumed_manifest),
        resumed_model_artifact_sha256=resumed_manifest.model_artifact.sha256,
        resumed_parent_checkpoint_id=resumed_parent,
        optimizer_steps=final_step,
        final_tokens_consumed=uninterrupted_progress.tokens_consumed,
        device="cuda",
        precision="bf16",
        compile_model=True,
        thresholds=accepted,
        compared_tensors=tensor_count,
        compared_values=value_count,
        exact_equal_values=exact_count,
        tolerance_violation_values=violation_count,
        tensors_with_tolerance_violations=violating_tensors,
        parameter_max_absolute_error=maximum_parameter_error,
        parameter_max_tolerance_ratio=maximum_tolerance_ratio,
        step_loss_differences=loss_differences,
        loss_max_absolute_error=maximum_loss,
        passed=passed,
    )


def write_dense_resume_tolerance_report(
    path: str | Path,
    report: DenseResumeToleranceReport,
) -> None:
    """Atomically write one canonical comparison report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise ResumeToleranceError("incomplete resume-tolerance report exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(report.canonical_bytes())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
