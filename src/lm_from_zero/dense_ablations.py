"""CPU-only planning contract for the Milestone-8 dense ablations."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lm_from_zero.architecture_study import (
    STUDY_SEEDS,
    ArchitectureStudyLineagePlan,
    ArchitectureStudyPlan,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
VariantName = Literal[
    "baseline",
    "learned_absolute_positions",
    "layer_norm",
    "gelu",
    "mha",
    "without_qk_norm",
    "tied_embeddings",
    "hybrid_muon",
]
DenseModelVariant = Literal[
    "baseline",
    "learned_absolute_positions",
    "layer_norm",
    "gelu",
    "mha",
    "without_qk_norm",
    "tied_embeddings",
]
OptimizerVariant = Literal["adamw", "hybrid_muon"]
VariantChange = Literal[
    "none",
    "rope_to_learned_absolute_positions",
    "rmsnorm_to_layernorm",
    "swiglu_to_gelu",
    "gqa_to_mha",
    "remove_qk_normalization",
    "tie_input_output_embeddings",
    "adamw_to_hybrid_muon",
]
ExecutionStatus = Literal[
    "reuse_m7_screening",
    "requires_m7_checkpoint_recovery",
    "ready_for_bounded_gpu_smoke",
]


class DenseAblationError(RuntimeError):
    """Raised when an M8 plan would break the controlled-study contract."""


class DenseAblationVariantSpec(BaseModel):
    """One predeclared single-variable change in the dense model study."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: VariantName
    change: VariantChange
    description: str = Field(min_length=1)
    execution_status: ExecutionStatus
    model_variant: DenseModelVariant = "baseline"
    optimizer_variant: OptimizerVariant = "adamw"

    @model_validator(mode="after")
    def validate_status(self) -> Self:
        if self.name == "baseline":
            if self.change != "none":
                raise ValueError("the baseline cannot change a model variable")
            if self.execution_status != "reuse_m7_screening":
                raise ValueError("the baseline must reuse the M7 screening runs")
            if self.model_variant != "baseline" or self.optimizer_variant != "adamw":
                raise ValueError("the baseline must use canonical model and optimizer")
        elif self.execution_status != "ready_for_bounded_gpu_smoke":
            raise ValueError("a new variant cannot reuse a canonical M7 checkpoint")
        if self.name == "hybrid_muon":
            if (
                self.model_variant != "baseline"
                or self.optimizer_variant != "hybrid_muon"
            ):
                raise ValueError("hybrid Muon must keep the baseline model variant")
        elif self.name != "baseline" and (
            self.model_variant != self.name or self.optimizer_variant != "adamw"
        ):
            raise ValueError("dense ablations must change exactly one control")
        return self

    def canonical_json(self) -> str:
        """Return stable bytes for the variant specification hash."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def spec_sha256(self) -> str:
        """Return the immutable hash of the declared one-variable change."""

        return sha256(self.canonical_json().encode()).hexdigest()


class DenseAblationSeedPlan(BaseModel):
    """One seed-specific job binding for one ablation variant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variant: VariantName
    variant_spec_sha256: Sha256
    seed: int
    execution_status: ExecutionStatus
    model_variant: DenseModelVariant = "baseline"
    optimizer_variant: OptimizerVariant = "adamw"
    artifact_directory: str
    checkpoint_directory: str
    jsonl_log: str
    expected_optimizer_step: Annotated[int, Field(gt=0)]
    expected_tokens: Annotated[int, Field(gt=0)]
    model_config_sha256: Sha256 | None = None
    training_config_sha256: Sha256 | None = None
    reused_m7_checkpoint: str | None = None

    @model_validator(mode="after")
    def validate_reuse(self) -> Self:
        if self.execution_status == "reuse_m7_screening":
            if (
                self.model_config_sha256 is None
                or self.training_config_sha256 is None
                or self.reused_m7_checkpoint is None
            ):
                raise ValueError("M7 reuse jobs require all source bindings")
            if self.model_variant != "baseline" or self.optimizer_variant != "adamw":
                raise ValueError("M7 reuse jobs must use canonical controls")
        elif self.execution_status == "requires_m7_checkpoint_recovery":
            if any(
                value is not None
                for value in (
                    self.model_config_sha256,
                    self.training_config_sha256,
                    self.reused_m7_checkpoint,
                )
            ):
                raise ValueError(
                    "checkpoint-recovery jobs cannot claim source bindings"
                )
            if self.model_variant != "baseline" or self.optimizer_variant != "adamw":
                raise ValueError("checkpoint-recovery jobs must use canonical controls")
        elif any(
            value is not None
            for value in (
                self.model_config_sha256,
                self.training_config_sha256,
                self.reused_m7_checkpoint,
            )
        ):
            raise ValueError("new variants cannot claim source bindings")
        if self.variant == "hybrid_muon":
            if (
                self.model_variant != "baseline"
                or self.optimizer_variant != "hybrid_muon"
            ):
                raise ValueError("hybrid Muon jobs must use baseline model controls")
        elif self.variant != "baseline" and (
            self.model_variant != self.variant or self.optimizer_variant != "adamw"
        ):
            raise ValueError("variant job controls do not match the declared variant")
        return self


class DenseAblationPlan(BaseModel):
    """Canonical M8 plan emitted before any dense-ablation execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-dense-ablations-plan"] = (
        "lm-from-zero-dense-ablations-plan"
    )
    format_version: Literal[1] = 1
    source_architecture_study_plan_sha256: Sha256
    shard_manifest_sha256: Sha256
    tokenizer_sha256: Sha256
    sequence_length: Literal[1_024]
    micro_batch_size: Annotated[int, Field(gt=0)]
    gradient_accumulation_steps: Annotated[int, Field(gt=0)]
    dense_reference_tokens: Literal[100_000_000]
    expected_screening_optimizer_step: Annotated[int, Field(gt=0)]
    expected_screening_tokens: Annotated[int, Field(gt=0)]
    seeds: tuple[int, ...]
    variants: tuple[DenseAblationVariantSpec, ...]
    jobs: tuple[DenseAblationSeedPlan, ...]
    execution_ready: Literal[False] = False

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if self.seeds != STUDY_SEEDS:
            raise ValueError("M8 must use the three predeclared study seeds")
        expected_names = {
            "baseline",
            "learned_absolute_positions",
            "layer_norm",
            "gelu",
            "mha",
            "without_qk_norm",
            "tied_embeddings",
            "hybrid_muon",
        }
        realized_names = {variant.name for variant in self.variants}
        if realized_names != expected_names or len(self.variants) != 8:
            raise ValueError(
                "M8 requires one baseline and seven single-change variants"
            )
        expected_jobs = {
            (variant.name, seed) for variant in self.variants for seed in self.seeds
        }
        realized_jobs = {(job.variant, job.seed) for job in self.jobs}
        if realized_jobs != expected_jobs or len(self.jobs) != len(expected_jobs):
            raise ValueError("M8 must contain exactly eight variants by three seeds")
        baseline_jobs = [job for job in self.jobs if job.variant == "baseline"]
        if len(baseline_jobs) != 3 or any(
            job.execution_status
            not in {"reuse_m7_screening", "requires_m7_checkpoint_recovery"}
            for job in baseline_jobs
        ):
            raise ValueError(
                "baseline jobs must reuse or recover M7 screening checkpoints"
            )
        if any(
            job.execution_status == "ready_for_bounded_gpu_smoke"
            and job.reused_m7_checkpoint is not None
            for job in self.jobs
        ):
            raise ValueError("unimplemented variants cannot reuse M7 checkpoints")
        return self

    def canonical_json(self) -> str:
        """Return stable machine-readable planning evidence."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


_VARIANTS: tuple[DenseAblationVariantSpec, ...] = (
    DenseAblationVariantSpec(
        name="baseline",
        change="none",
        description="Canonical OLMo2 dense configuration reused from M7.",
        execution_status="reuse_m7_screening",
    ),
    DenseAblationVariantSpec(
        name="learned_absolute_positions",
        change="rope_to_learned_absolute_positions",
        description="Replace RoPE with learned absolute position embeddings.",
        execution_status="ready_for_bounded_gpu_smoke",
        model_variant="learned_absolute_positions",
    ),
    DenseAblationVariantSpec(
        name="layer_norm",
        change="rmsnorm_to_layernorm",
        description="Replace RMSNorm with LayerNorm while preserving branch layout.",
        execution_status="ready_for_bounded_gpu_smoke",
        model_variant="layer_norm",
    ),
    DenseAblationVariantSpec(
        name="gelu",
        change="swiglu_to_gelu",
        description="Replace SwiGLU with a parameter-matched GELU branch.",
        execution_status="ready_for_bounded_gpu_smoke",
        model_variant="gelu",
    ),
    DenseAblationVariantSpec(
        name="mha",
        change="gqa_to_mha",
        description="Replace grouped-query attention with multi-head attention.",
        execution_status="ready_for_bounded_gpu_smoke",
        model_variant="mha",
    ),
    DenseAblationVariantSpec(
        name="without_qk_norm",
        change="remove_qk_normalization",
        description="Disable query/key normalization and change no other objective.",
        execution_status="ready_for_bounded_gpu_smoke",
        model_variant="without_qk_norm",
    ),
    DenseAblationVariantSpec(
        name="tied_embeddings",
        change="tie_input_output_embeddings",
        description="Tie the input embedding and output projection weights.",
        execution_status="ready_for_bounded_gpu_smoke",
        model_variant="tied_embeddings",
    ),
    DenseAblationVariantSpec(
        name="hybrid_muon",
        change="adamw_to_hybrid_muon",
        description="Use Muon for eligible matrices and AdamW elsewhere.",
        execution_status="ready_for_bounded_gpu_smoke",
        optimizer_variant="hybrid_muon",
    ),
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _dense_lineages(
    plan: ArchitectureStudyPlan,
) -> dict[int, ArchitectureStudyLineagePlan]:
    lineages = [item for item in plan.lineages if item.architecture == "dense"]
    if len(lineages) != len(STUDY_SEEDS):
        raise DenseAblationError("the M7 plan must contain three dense lineages")
    return {item.seed: item for item in lineages}


def create_dense_ablation_plan(
    architecture_study_plan: str | Path,
    *,
    artifact_root: str | Path = "artifacts/dense-ablations",
) -> DenseAblationPlan:
    """Create the non-executable M8 plan from the frozen M7 plan."""

    source_path = Path(architecture_study_plan)
    try:
        source = ArchitectureStudyPlan.model_validate_json(source_path.read_text())
    except (OSError, ValueError) as error:
        raise DenseAblationError(
            "cannot load the M7 architecture-study plan"
        ) from error
    dense = _dense_lineages(source)
    if source.sequence_length != 1_024 or source.micro_batch_size <= 0:
        raise DenseAblationError("M8 requires the M7 1024-token dense batch contract")
    if source.screening_dense_reference_tokens != 100_000_000:
        raise DenseAblationError("M8 reuses only the 100M-token M7 screen")
    expected_steps = {item.screening.optimizer_step for item in dense.values()}
    expected_tokens = {item.screening.tokens_consumed for item in dense.values()}
    if len(expected_steps) != 1 or len(expected_tokens) != 1:
        raise DenseAblationError("M7 dense screening boundaries are not identical")
    expected_step = next(iter(expected_steps))
    expected_token_count = next(iter(expected_tokens))
    root = Path(artifact_root)
    jobs: list[DenseAblationSeedPlan] = []
    for variant in _VARIANTS:
        for seed in STUDY_SEEDS:
            source_lineage = dense[seed]
            job_root = root / variant.name / f"seed-{seed}"
            reused = None
            execution_status = variant.execution_status
            if variant.name == "baseline":
                candidate = Path(source_lineage.screening_checkpoint)
                if not candidate.is_absolute():
                    candidate = Path.cwd() / candidate
                if candidate.is_dir():
                    reused = source_lineage.screening_checkpoint
                else:
                    execution_status = "requires_m7_checkpoint_recovery"
            jobs.append(
                DenseAblationSeedPlan(
                    variant=variant.name,
                    variant_spec_sha256=variant.spec_sha256,
                    seed=seed,
                    execution_status=execution_status,
                    model_variant=variant.model_variant,
                    optimizer_variant=variant.optimizer_variant,
                    artifact_directory=job_root.as_posix(),
                    checkpoint_directory=(job_root / "checkpoints").as_posix(),
                    jsonl_log=(job_root / "events.jsonl").as_posix(),
                    expected_optimizer_step=expected_step,
                    expected_tokens=expected_token_count,
                    model_config_sha256=(
                        source_lineage.model_config_sha256
                        if reused is not None
                        else None
                    ),
                    training_config_sha256=(
                        source_lineage.training_config_sha256
                        if reused is not None
                        else None
                    ),
                    reused_m7_checkpoint=reused,
                )
            )
    return DenseAblationPlan(
        source_architecture_study_plan_sha256=_sha256_file(source_path),
        shard_manifest_sha256=source.shard_manifest_sha256,
        tokenizer_sha256=source.tokenizer_sha256,
        sequence_length=source.sequence_length,
        micro_batch_size=source.micro_batch_size,
        gradient_accumulation_steps=source.gradient_accumulation_steps,
        dense_reference_tokens=100_000_000,
        expected_screening_optimizer_step=expected_step,
        expected_screening_tokens=expected_token_count,
        seeds=STUDY_SEEDS,
        variants=_VARIANTS,
        jobs=tuple(jobs),
    )


def write_dense_ablation_plan(
    path: str | Path,
    plan: DenseAblationPlan,
) -> None:
    """Atomically write one complete M8 plan."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise DenseAblationError("incomplete dense-ablation plan exists")
    try:
        with temporary.open("xb") as handle:
            handle.write((plan.canonical_json() + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
