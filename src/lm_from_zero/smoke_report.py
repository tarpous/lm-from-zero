"""Generate compact dense-smoke evidence from validated local artifacts."""

from __future__ import annotations

import json
import os
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from lm_from_zero.diffusion_evaluation import DiffusionEvaluationResult
from lm_from_zero.evaluation import CausalEvaluationResult
from lm_from_zero.export_diffusion_hf import (
    DiffusionHFExportManifest,
    load_diffusion_export_manifest,
)
from lm_from_zero.export_hf import DenseHFExportManifest, load_export_manifest
from lm_from_zero.export_mamba2_hf import (
    Mamba2HFExportManifest,
    load_mamba2_export_manifest,
)
from lm_from_zero.generation import (
    CausalGenerationRecord,
    DiffusionGenerationRecord,
)
from lm_from_zero.training import validate_checkpoint

SHA256_PATTERN = r"^[0-9a-f]{64}$"
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]


class SmokeReportError(RuntimeError):
    """Raised when measured smoke artifacts disagree or are incomplete."""


class TrainingStepEvidence(BaseModel):
    """Measured optimizer-step metrics copied from canonical runner JSONL."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    event: Literal["optimizer_step"] = "optimizer_step"
    recorded_at_utc: datetime
    optimizer_step: Annotated[int, Field(gt=0)]
    loss: Annotated[float, Field(ge=0)]
    learning_rate: Annotated[float, Field(gt=0)]
    gradient_norm: Annotated[float, Field(ge=0)]
    tokens_consumed: Annotated[int, Field(gt=0)]
    elapsed_seconds: Annotated[float, Field(gt=0)]
    tokens_per_second: Annotated[float, Field(gt=0)]
    peak_cuda_memory_allocated_bytes: Annotated[int, Field(gt=0)]
    peak_cuda_memory_reserved_bytes: Annotated[int, Field(gt=0)]


class DenseSmokeReport(BaseModel):
    """Portable, generated evidence for a causal GPU vertical-slice smoke."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal[
        "lm-from-zero-dense-smoke-report",
        "lm-from-zero-mamba2-smoke-report",
    ] = "lm-from-zero-dense-smoke-report"
    format_version: Literal[1] = 1
    architecture: Literal["olmo2", "mamba2"] = "olmo2"
    recorded_at_utc: datetime
    source_git_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    training_config_sha256: Sha256
    model_config_sha256: Sha256
    shard_manifest_sha256: Sha256
    tokenizer_sha256: Sha256
    checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    checkpoint_manifest_sha256: Sha256
    parent_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    device: Literal["cuda"]
    precision: Literal["bf16"]
    compile_model: Literal[True]
    cuda_device_names: tuple[str, ...]
    cuda_version: str
    torch_version: str
    final_tokens_consumed: Annotated[int, Field(gt=0)]
    training_steps: tuple[TrainingStepEvidence, ...]
    validation_loss: Annotated[float, Field(ge=0)]
    validation_perplexity: Annotated[float, Field(gt=0)]
    validation_predicted_tokens_per_second: Annotated[float, Field(gt=0)]
    export_transformers_version: str
    export_fp32_max_abs_error: Annotated[float, Field(ge=0)]
    export_cached_fp32_max_abs_error: Annotated[float, Field(ge=0)] | None = None
    export_requires_trust_remote_code: bool | None = None
    export_artifact_sha256: dict[str, Sha256]
    generation_prompt_token_sha256: Sha256
    generation_model_forwards: Annotated[int, Field(gt=0)]
    generation_token_count: Annotated[int, Field(gt=0)]
    generation_tokens_per_second: Annotated[float, Field(gt=0)]
    generation_stop_reasons: tuple[Literal["eos", "max_new_tokens"], ...]

    def canonical_bytes(self) -> bytes:
        """Return deterministic JSON bytes for the committed report."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


class DiffusionSmokeReport(BaseModel):
    """Portable generated evidence for a diffusion GPU vertical slice."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-diffusion-smoke-report"] = (
        "lm-from-zero-diffusion-smoke-report"
    )
    format_version: Literal[2] = 2
    architecture: Literal["masked_diffusion"] = "masked_diffusion"
    recorded_at_utc: datetime
    source_git_revision: str = Field(pattern=r"^[0-9a-f]{40,64}$")
    training_config_sha256: Sha256
    model_config_sha256: Sha256
    shard_manifest_sha256: Sha256
    tokenizer_sha256: Sha256
    checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    checkpoint_manifest_sha256: Sha256
    parent_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    device: Literal["cuda"]
    precision: Literal["bf16"]
    compile_model: Literal[True]
    cuda_device_names: tuple[str, ...]
    cuda_version: str
    torch_version: str
    final_tokens_consumed: Annotated[int, Field(gt=0)]
    training_steps: tuple[TrainingStepEvidence, ...]
    validation_masked_reconstruction_loss_nats: Annotated[float, Field(ge=0)]
    validation_variational_upper_bound_nats: Annotated[float, Field(ge=0)]
    validation_mean_mask_rate: Annotated[float, Field(gt=0, le=1)]
    validation_masked_tokens_per_second: Annotated[float, Field(gt=0)]
    validation_model_forwards: Annotated[int, Field(gt=0)]
    evaluation_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    evaluation_checkpoint_manifest_sha256: Sha256
    evaluation_device: Literal["cuda"]
    causal_perplexity_applicable: Literal[False] = False
    export_transformers_version: str
    export_fp32_max_abs_error: Annotated[float, Field(ge=0)]
    export_fp32_loss_abs_error: Annotated[float, Field(ge=0)]
    export_deterministic_trajectory_matches: Literal[True]
    export_requires_trust_remote_code: Literal[True]
    export_artifact_sha256: dict[str, Sha256]
    generation_prompt_token_sha256: Sha256
    generation_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    generation_checkpoint_manifest_sha256: Sha256
    generation_device: Literal["cuda"]
    generation_model_forwards: Annotated[int, Field(gt=0)]
    generation_diffusion_steps: Annotated[int, Field(gt=0)]
    generation_response_canvas_length: Annotated[int, Field(gt=0)]
    generation_token_count: Annotated[int, Field(gt=0)]
    generation_tokens_per_second: Annotated[float, Field(gt=0)]
    generation_stop_reasons: tuple[Literal["eos", "canvas_complete"], ...]

    def canonical_bytes(self) -> bytes:
        """Return deterministic JSON bytes for the committed report."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()


def _canonical_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise SmokeReportError(f"cannot read JSONL evidence: {path.name}") from error
    if not lines:
        raise SmokeReportError(f"JSONL evidence is empty: {path.name}")
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise SmokeReportError(f"invalid JSONL evidence: {path.name}") from error
        if not isinstance(payload, dict):
            raise SmokeReportError(f"JSONL record is not an object: {path.name}")
        canonical = json.dumps(
            payload,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        if line != canonical:
            raise SmokeReportError(f"JSONL evidence is not canonical: {path.name}")
        records.append(payload)
    return records


def _single_model_record(
    path: Path,
    model_type: type[BaseModel],
) -> BaseModel:
    records = _canonical_jsonl(path)
    if len(records) != 1:
        raise SmokeReportError(f"expected one JSONL record: {path.name}")
    try:
        return model_type.model_validate(records[0])
    except ValueError as error:
        raise SmokeReportError(f"invalid JSONL schema: {path.name}") from error


def _training_evidence(
    path: str | Path,
) -> tuple[str, tuple[TrainingStepEvidence, ...]]:
    events = _canonical_jsonl(Path(path))
    starts = [record for record in events if record.get("event") == "run_start"]
    resumes = [record for record in events if record.get("event") == "run_resume"]
    if len(starts) != 1 or len(resumes) != 1:
        raise SmokeReportError("smoke evidence requires one start and one resume")
    training_hash = starts[0].get("training_config_sha256")
    if not isinstance(training_hash, str) or training_hash != resumes[0].get(
        "training_config_sha256"
    ):
        raise SmokeReportError("training configuration changed across resume")
    raw_config = starts[0].get("training_config")
    if not isinstance(raw_config, dict):
        raise SmokeReportError("training configuration record is missing")
    if (
        raw_config.get("device") != "cuda"
        or raw_config.get("precision") != "bf16"
        or raw_config.get("compile_model") is not True
    ):
        raise SmokeReportError("smoke did not use compiled bf16 CUDA training")
    raw_steps = [record for record in events if record.get("event") == "optimizer_step"]
    try:
        steps = tuple(TrainingStepEvidence.model_validate(step) for step in raw_steps)
    except ValueError as error:
        raise SmokeReportError("optimizer-step evidence is invalid") from error
    if [step.optimizer_step for step in steps] != list(range(1, len(steps) + 1)):
        raise SmokeReportError("optimizer-step evidence is not contiguous")
    return training_hash, steps


def build_dense_smoke_report(
    *,
    training_jsonl: str | Path,
    checkpoint_directory: str | Path,
    evaluation_jsonl: str | Path,
    export_directory: str | Path,
    generation_jsonl: str | Path,
) -> DenseSmokeReport:
    """Cross-check measured artifacts and return compact portable evidence."""

    return _build_causal_smoke_report(
        training_jsonl=training_jsonl,
        checkpoint_directory=checkpoint_directory,
        evaluation_jsonl=evaluation_jsonl,
        export_directory=export_directory,
        generation_jsonl=generation_jsonl,
        architecture="olmo2",
    )


def build_mamba2_smoke_report(
    *,
    training_jsonl: str | Path,
    checkpoint_directory: str | Path,
    evaluation_jsonl: str | Path,
    export_directory: str | Path,
    generation_jsonl: str | Path,
) -> DenseSmokeReport:
    """Cross-check measured Mamba-2 artifacts and return portable evidence."""

    return _build_causal_smoke_report(
        training_jsonl=training_jsonl,
        checkpoint_directory=checkpoint_directory,
        evaluation_jsonl=evaluation_jsonl,
        export_directory=export_directory,
        generation_jsonl=generation_jsonl,
        architecture="mamba2",
    )


def _build_causal_smoke_report(
    *,
    training_jsonl: str | Path,
    checkpoint_directory: str | Path,
    evaluation_jsonl: str | Path,
    export_directory: str | Path,
    generation_jsonl: str | Path,
    architecture: Literal["olmo2", "mamba2"],
) -> DenseSmokeReport:
    training_hash, steps = _training_evidence(training_jsonl)

    checkpoint = validate_checkpoint(checkpoint_directory)
    if checkpoint.binding.architecture != architecture:
        raise SmokeReportError("checkpoint architecture disagrees with report")
    if checkpoint.binding.git.dirty:
        raise SmokeReportError("smoke checkpoint was created from a dirty worktree")
    checkpoint_hash = sha256(checkpoint.canonical_bytes()).hexdigest()
    if not steps or steps[-1].optimizer_step != checkpoint.progress.optimizer_step:
        raise SmokeReportError("training steps do not reach the final checkpoint")
    if steps[-1].tokens_consumed != checkpoint.progress.tokens_consumed:
        raise SmokeReportError("training tokens do not match the final checkpoint")
    parent_id = checkpoint.lineage.parent_checkpoint_id
    if parent_id is None:
        raise SmokeReportError("smoke checkpoint does not prove resume lineage")

    evaluation = _single_model_record(
        Path(evaluation_jsonl),
        CausalEvaluationResult,
    )
    assert isinstance(evaluation, CausalEvaluationResult)
    exported: DenseHFExportManifest | Mamba2HFExportManifest
    report_format: Literal[
        "lm-from-zero-dense-smoke-report",
        "lm-from-zero-mamba2-smoke-report",
    ]
    if architecture == "olmo2":
        exported = load_export_manifest(Path(export_directory) / "export_manifest.json")
        cached_export_error = None
        requires_trust_remote_code = None
        report_format = "lm-from-zero-dense-smoke-report"
    else:
        exported = load_mamba2_export_manifest(
            Path(export_directory) / "export_manifest.json"
        )
        cached_export_error = exported.cached_fp32_max_abs_error
        requires_trust_remote_code = exported.requires_trust_remote_code
        report_format = "lm-from-zero-mamba2-smoke-report"
    generation = _single_model_record(
        Path(generation_jsonl),
        CausalGenerationRecord,
    )
    assert isinstance(generation, CausalGenerationRecord)

    binding = checkpoint.binding
    common = {
        binding.model_config_sha256,
        evaluation.model_config_sha256,
        exported.model_config_sha256,
        generation.model_config_sha256,
    }
    if len(common) != 1:
        raise SmokeReportError("model configuration hashes disagree")
    tokenizer_hashes = {
        binding.tokenizer_sha256,
        evaluation.tokenizer_sha256,
        exported.tokenizer_sha256,
        generation.tokenizer_sha256,
    }
    if len(tokenizer_hashes) != 1:
        raise SmokeReportError("tokenizer hashes disagree")
    if evaluation.shard_manifest_sha256 != binding.shard_manifest_sha256:
        raise SmokeReportError("evaluation shard hash disagrees with checkpoint")
    if exported.source_checkpoint_id != checkpoint.lineage.checkpoint_id:
        raise SmokeReportError("export checkpoint ID disagrees")
    if exported.source_checkpoint_manifest_sha256 != checkpoint_hash:
        raise SmokeReportError("export checkpoint manifest hash disagrees")
    runtime = binding.runtime
    if not runtime.cuda_available or runtime.cuda_version is None:
        raise SmokeReportError("smoke checkpoint lacks CUDA runtime evidence")

    recorded_at = max(
        checkpoint.created_at_utc,
        evaluation.evaluated_at_utc,
        generation.generated_at_utc,
    )
    return DenseSmokeReport(
        format=report_format,
        architecture=architecture,
        recorded_at_utc=recorded_at,
        source_git_revision=binding.git.revision,
        training_config_sha256=str(training_hash),
        model_config_sha256=binding.model_config_sha256,
        shard_manifest_sha256=binding.shard_manifest_sha256,
        tokenizer_sha256=binding.tokenizer_sha256,
        checkpoint_id=checkpoint.lineage.checkpoint_id,
        checkpoint_manifest_sha256=checkpoint_hash,
        parent_checkpoint_id=parent_id,
        device="cuda",
        precision="bf16",
        compile_model=True,
        cuda_device_names=runtime.cuda_device_names,
        cuda_version=runtime.cuda_version,
        torch_version=runtime.torch_version,
        final_tokens_consumed=checkpoint.progress.tokens_consumed,
        training_steps=steps,
        validation_loss=evaluation.mean_loss,
        validation_perplexity=evaluation.perplexity,
        validation_predicted_tokens_per_second=(evaluation.predicted_tokens_per_second),
        export_transformers_version=exported.transformers_version,
        export_fp32_max_abs_error=exported.fp32_max_abs_error,
        export_cached_fp32_max_abs_error=cached_export_error,
        export_requires_trust_remote_code=requires_trust_remote_code,
        export_artifact_sha256={
            artifact.filename: artifact.sha256 for artifact in exported.artifacts
        },
        generation_prompt_token_sha256=generation.prompt_token_sha256,
        generation_model_forwards=generation.result.model_forwards,
        generation_token_count=generation.result.generated_token_count,
        generation_tokens_per_second=generation.result.tokens_per_second,
        generation_stop_reasons=generation.result.stop_reasons,
    )


def build_diffusion_smoke_report(
    *,
    training_jsonl: str | Path,
    checkpoint_directory: str | Path,
    evaluation_jsonl: str | Path,
    export_directory: str | Path,
    generation_jsonl: str | Path,
) -> DiffusionSmokeReport:
    """Cross-check measured diffusion artifacts and return portable evidence."""

    training_hash, steps = _training_evidence(training_jsonl)
    checkpoint = validate_checkpoint(checkpoint_directory)
    if checkpoint.binding.architecture != "masked_diffusion":
        raise SmokeReportError("checkpoint architecture disagrees with report")
    if checkpoint.binding.git.dirty:
        raise SmokeReportError("smoke checkpoint was created from a dirty worktree")
    checkpoint_hash = sha256(checkpoint.canonical_bytes()).hexdigest()
    if not steps or steps[-1].optimizer_step != checkpoint.progress.optimizer_step:
        raise SmokeReportError("training steps do not reach the final checkpoint")
    if steps[-1].tokens_consumed != checkpoint.progress.tokens_consumed:
        raise SmokeReportError("training tokens do not match the final checkpoint")
    parent_id = checkpoint.lineage.parent_checkpoint_id
    if parent_id is None:
        raise SmokeReportError("smoke checkpoint does not prove resume lineage")

    evaluation = _single_model_record(
        Path(evaluation_jsonl),
        DiffusionEvaluationResult,
    )
    assert isinstance(evaluation, DiffusionEvaluationResult)
    exported: DiffusionHFExportManifest = load_diffusion_export_manifest(
        Path(export_directory) / "export_manifest.json"
    )
    generation = _single_model_record(
        Path(generation_jsonl),
        DiffusionGenerationRecord,
    )
    assert isinstance(generation, DiffusionGenerationRecord)

    binding = checkpoint.binding
    model_hashes = {
        binding.model_config_sha256,
        evaluation.model_config_sha256,
        exported.model_config_sha256,
        generation.model_config_sha256,
    }
    if len(model_hashes) != 1:
        raise SmokeReportError("model configuration hashes disagree")
    tokenizer_hashes = {
        binding.tokenizer_sha256,
        evaluation.tokenizer_sha256,
        exported.tokenizer_sha256,
        generation.tokenizer_sha256,
    }
    if len(tokenizer_hashes) != 1:
        raise SmokeReportError("tokenizer hashes disagree")
    if evaluation.shard_manifest_sha256 != binding.shard_manifest_sha256:
        raise SmokeReportError("evaluation shard hash disagrees with checkpoint")
    if evaluation.source_checkpoint_id != checkpoint.lineage.checkpoint_id:
        raise SmokeReportError("evaluation checkpoint ID disagrees")
    if evaluation.source_checkpoint_manifest_sha256 != checkpoint_hash:
        raise SmokeReportError("evaluation checkpoint manifest hash disagrees")
    if evaluation.device != "cuda":
        raise SmokeReportError("diffusion smoke evaluation did not use CUDA")
    if exported.source_checkpoint_id != checkpoint.lineage.checkpoint_id:
        raise SmokeReportError("export checkpoint ID disagrees")
    if exported.source_checkpoint_manifest_sha256 != checkpoint_hash:
        raise SmokeReportError("export checkpoint manifest hash disagrees")
    if generation.source_checkpoint_id != checkpoint.lineage.checkpoint_id:
        raise SmokeReportError("generation checkpoint ID disagrees")
    if generation.source_checkpoint_manifest_sha256 != checkpoint_hash:
        raise SmokeReportError("generation checkpoint manifest hash disagrees")
    if generation.device != "cuda":
        raise SmokeReportError("diffusion smoke generation did not use CUDA")
    runtime = binding.runtime
    if not runtime.cuda_available or runtime.cuda_version is None:
        raise SmokeReportError("smoke checkpoint lacks CUDA runtime evidence")

    recorded_at = max(
        checkpoint.created_at_utc,
        evaluation.evaluated_at_utc,
        generation.generated_at_utc,
    )
    return DiffusionSmokeReport(
        recorded_at_utc=recorded_at,
        source_git_revision=binding.git.revision,
        training_config_sha256=training_hash,
        model_config_sha256=binding.model_config_sha256,
        shard_manifest_sha256=binding.shard_manifest_sha256,
        tokenizer_sha256=binding.tokenizer_sha256,
        checkpoint_id=checkpoint.lineage.checkpoint_id,
        checkpoint_manifest_sha256=checkpoint_hash,
        parent_checkpoint_id=parent_id,
        device="cuda",
        precision="bf16",
        compile_model=True,
        cuda_device_names=runtime.cuda_device_names,
        cuda_version=runtime.cuda_version,
        torch_version=runtime.torch_version,
        final_tokens_consumed=checkpoint.progress.tokens_consumed,
        training_steps=steps,
        validation_masked_reconstruction_loss_nats=(
            evaluation.masked_reconstruction_loss_nats
        ),
        validation_variational_upper_bound_nats=(
            evaluation.variational_upper_bound_nats
        ),
        validation_mean_mask_rate=evaluation.mean_mask_rate,
        validation_masked_tokens_per_second=(evaluation.masked_tokens_per_second),
        validation_model_forwards=evaluation.model_forwards,
        evaluation_checkpoint_id=evaluation.source_checkpoint_id,
        evaluation_checkpoint_manifest_sha256=(
            evaluation.source_checkpoint_manifest_sha256
        ),
        evaluation_device="cuda",
        export_transformers_version=exported.transformers_version,
        export_fp32_max_abs_error=exported.fp32_max_abs_error,
        export_fp32_loss_abs_error=exported.fp32_loss_abs_error,
        export_deterministic_trajectory_matches=(
            exported.deterministic_trajectory_matches
        ),
        export_requires_trust_remote_code=exported.requires_trust_remote_code,
        export_artifact_sha256={
            artifact.filename: artifact.sha256 for artifact in exported.artifacts
        },
        generation_prompt_token_sha256=generation.prompt_token_sha256,
        generation_checkpoint_id=generation.source_checkpoint_id,
        generation_checkpoint_manifest_sha256=(
            generation.source_checkpoint_manifest_sha256
        ),
        generation_device="cuda",
        generation_model_forwards=generation.result.model_forwards,
        generation_diffusion_steps=generation.result.diffusion_steps,
        generation_response_canvas_length=(generation.result.response_canvas_length),
        generation_token_count=generation.result.generated_token_count,
        generation_tokens_per_second=generation.result.tokens_per_second,
        generation_stop_reasons=generation.result.stop_reasons,
    )


def write_dense_smoke_report(
    path: str | Path,
    report: DenseSmokeReport | DiffusionSmokeReport,
) -> None:
    """Atomically write one canonical report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise SmokeReportError("incomplete smoke report file exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(report.canonical_bytes())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
