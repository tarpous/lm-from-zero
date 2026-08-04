"""Frozen offline planning contract for the three-family architecture study."""

from __future__ import annotations

import json
import os
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lm_from_zero.models import (
    Mamba2Config,
    MaskedDiffusionConfig,
    Olmo2Config,
)
from lm_from_zero.sharding import validate_shard_build
from lm_from_zero.training import (
    CausalBatchConfig,
    DenseTrainingConfig,
    DiffusionTrainingConfig,
    Mamba2TrainingConfig,
    OptimizationConfig,
)

Architecture = Literal["dense", "mamba2", "diffusion"]
ModelConfig = Olmo2Config | Mamba2Config | MaskedDiffusionConfig
TrainingConfig = DenseTrainingConfig | Mamba2TrainingConfig | DiffusionTrainingConfig
SHA256_PATTERN = r"^[0-9a-f]{64}$"
STUDY_SEEDS = (1_337, 2_027, 3_407)


class ArchitectureStudyError(RuntimeError):
    """Raised when a study plan cannot preserve its comparison contract."""


class StudyBudgetPlan(BaseModel):
    """One cumulative FLOP boundary in a full-scheduler training lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    dense_reference_tokens: Annotated[int, Field(gt=0)]
    reference_training_flops: Annotated[int, Field(gt=0)]
    optimizer_step: Annotated[int, Field(gt=0)]
    tokens_consumed: Annotated[int, Field(gt=0)]
    estimated_training_flops: Annotated[int, Field(gt=0)]
    training_flop_ratio: Annotated[float, Field(gt=0)]
    estimated_seconds: Annotated[float | None, Field(gt=0)] = None


class ArchitectureStudyLineagePlan(BaseModel):
    """One seed-specific lineage configured with the full-budget scheduler."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture: Architecture
    seed: int
    continues_to_full: bool
    training_config_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    model_config_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    parameter_count: Annotated[int, Field(gt=0)]
    forward_flops_per_token: Annotated[int, Field(gt=0)]
    tokens_per_optimizer_step: Annotated[int, Field(gt=0)]
    scheduler_total_steps: Annotated[int, Field(gt=0)]
    screening: StudyBudgetPlan
    full: StudyBudgetPlan
    checkpoint_directory: str
    jsonl_log: str
    tensorboard_directory: str
    parquet_log: str
    screening_checkpoint: str
    estimated_checkpoint_bytes: Annotated[int, Field(gt=0)]
    estimated_retained_checkpoint_bytes_upper_bound: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_lineage(self) -> Self:
        if self.screening.optimizer_step >= self.full.optimizer_step:
            raise ValueError("screening must stop before the full budget")
        if self.scheduler_total_steps != self.full.optimizer_step:
            raise ValueError("every lineage must use the full-budget scheduler")
        for stage in (self.screening, self.full):
            if not 0.97 <= stage.training_flop_ratio <= 1.03:
                raise ValueError("planned training FLOPs exceed the 3% matching window")
        expected_checkpoint = (
            Path(self.checkpoint_directory)
            / f"step-{self.screening.optimizer_step:012d}"
        ).as_posix()
        if self.screening_checkpoint != expected_checkpoint:
            raise ValueError("screening checkpoint path disagrees with its step")
        if self.continues_to_full != (self.seed == 1_337):
            raise ValueError("only seed 1337 may continue to the full budget")
        return self


class ArchitectureStudyPlan(BaseModel):
    """Canonical nine-lineage contract emitted before any expensive run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-architecture-study-plan"] = (
        "lm-from-zero-architecture-study-plan"
    )
    format_version: Literal[1] = 1
    shard_manifest_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    tokenizer_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    tokenizer_vocab_size: Literal[16_000]
    sequence_length: Literal[1_024]
    micro_batch_size: Annotated[int, Field(gt=0)]
    gradient_accumulation_steps: Annotated[int, Field(gt=0)]
    world_size: Literal[1]
    screening_dense_reference_tokens: Annotated[int, Field(gt=0)]
    full_dense_reference_tokens: Annotated[int, Field(gt=0)]
    lineages: tuple[ArchitectureStudyLineagePlan, ...]

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if self.screening_dense_reference_tokens >= self.full_dense_reference_tokens:
            raise ValueError("screening budget must be smaller than the full budget")
        expected = {
            (architecture, seed)
            for architecture in ("dense", "mamba2", "diffusion")
            for seed in STUDY_SEEDS
        }
        realized = {(item.architecture, item.seed) for item in self.lineages}
        if realized != expected or len(self.lineages) != len(expected):
            raise ValueError(
                "study plan must contain three architectures by three seeds"
            )
        paths = [item.checkpoint_directory for item in self.lineages]
        if len(paths) != len(set(paths)):
            raise ValueError("study checkpoint paths must be unique")
        return self

    def canonical_json(self) -> str:
        """Return stable machine-readable planning evidence."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _matched_budget(
    *,
    dense_reference_tokens: int,
    dense_forward_flops_per_token: int,
    architecture_forward_flops_per_token: int,
    tokens_per_optimizer_step: int,
    estimated_tokens_per_second: float | None,
) -> StudyBudgetPlan:
    reference_flops = 3 * dense_forward_flops_per_token * dense_reference_tokens
    matched_tokens = (
        reference_flops + 3 * architecture_forward_flops_per_token - 1
    ) // (3 * architecture_forward_flops_per_token)
    optimizer_step = (
        matched_tokens + tokens_per_optimizer_step - 1
    ) // tokens_per_optimizer_step
    tokens_consumed = optimizer_step * tokens_per_optimizer_step
    estimated_flops = 3 * architecture_forward_flops_per_token * tokens_consumed
    return StudyBudgetPlan(
        dense_reference_tokens=dense_reference_tokens,
        reference_training_flops=reference_flops,
        optimizer_step=optimizer_step,
        tokens_consumed=tokens_consumed,
        estimated_training_flops=estimated_flops,
        training_flop_ratio=estimated_flops / reference_flops,
        estimated_seconds=(
            None
            if estimated_tokens_per_second is None
            else tokens_consumed / estimated_tokens_per_second
        ),
    )


def create_architecture_study_plan(
    build_manifest: str | Path,
    *,
    artifact_root: str | Path = "artifacts/architecture-study",
    screening_dense_reference_tokens: int = 100_000_000,
    full_dense_reference_tokens: int = 500_000_000,
    micro_batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    dense_tokens_per_second: float | None = None,
    mamba2_tokens_per_second: float | None = None,
    diffusion_tokens_per_second: float | None = None,
) -> ArchitectureStudyPlan:
    """Resolve all nine full-scheduler lineages without allocating a model."""

    if screening_dense_reference_tokens <= 0 or full_dense_reference_tokens <= 0:
        raise ArchitectureStudyError("study token budgets must be positive")
    if screening_dense_reference_tokens >= full_dense_reference_tokens:
        raise ArchitectureStudyError("screening budget must precede the full budget")
    throughputs = {
        "dense": dense_tokens_per_second,
        "mamba2": mamba2_tokens_per_second,
        "diffusion": diffusion_tokens_per_second,
    }
    if any(value is not None and value <= 0 for value in throughputs.values()):
        raise ArchitectureStudyError("estimated token throughput must be positive")

    build_path = Path(build_manifest)
    build = validate_shard_build(build_path)
    if build.tokenizer_vocab_size != 16_000:
        raise ArchitectureStudyError("architecture study requires the 16K tokenizer")

    sequence_length: Literal[1_024] = 1_024
    dense_model = Olmo2Config(tokenizer_hash=build.tokenizer_hash)
    models: dict[Architecture, ModelConfig] = {
        "dense": dense_model,
        "mamba2": Mamba2Config(tokenizer_hash=build.tokenizer_hash),
        "diffusion": MaskedDiffusionConfig(tokenizer_hash=build.tokenizer_hash),
    }
    dense_forward_flops = dense_model.forward_flops(
        sequence_length
    ).total_flops_per_token
    root = Path(artifact_root)
    lineages: list[ArchitectureStudyLineagePlan] = []
    architectures: tuple[Architecture, ...] = ("dense", "mamba2", "diffusion")
    for architecture in architectures:
        model = models[architecture]
        architecture_forward_flops = model.forward_flops(
            sequence_length
        ).total_flops_per_token
        for seed in STUDY_SEEDS:
            batch = CausalBatchConfig(
                sequence_length=sequence_length,
                micro_batch_size=micro_batch_size,
                seed=seed,
            )
            full = _matched_budget(
                dense_reference_tokens=full_dense_reference_tokens,
                dense_forward_flops_per_token=dense_forward_flops,
                architecture_forward_flops_per_token=architecture_forward_flops,
                tokens_per_optimizer_step=(
                    sequence_length * micro_batch_size * gradient_accumulation_steps
                ),
                estimated_tokens_per_second=throughputs[architecture],
            )
            screening = _matched_budget(
                dense_reference_tokens=screening_dense_reference_tokens,
                dense_forward_flops_per_token=dense_forward_flops,
                architecture_forward_flops_per_token=architecture_forward_flops,
                tokens_per_optimizer_step=(
                    sequence_length * micro_batch_size * gradient_accumulation_steps
                ),
                estimated_tokens_per_second=throughputs[architecture],
            )
            optimization = OptimizationConfig(total_steps=full.optimizer_step)
            common = {
                "model": model,
                "batch": batch,
                "optimization": optimization,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "device": "cuda",
                "precision": "bf16",
                "compile_model": True,
                "seed": seed,
            }
            training: TrainingConfig
            if architecture == "dense":
                training = DenseTrainingConfig.model_validate(common)
            elif architecture == "mamba2":
                training = Mamba2TrainingConfig.model_validate(common)
            else:
                training = DiffusionTrainingConfig.model_validate(common)
            lineage_root = root / architecture / f"seed-{seed}"
            checkpoints = lineage_root / "checkpoints"
            checkpoint_bytes = 12 * model.parameter_breakdown().total
            lineages.append(
                ArchitectureStudyLineagePlan(
                    architecture=architecture,
                    seed=seed,
                    continues_to_full=seed == 1_337,
                    training_config_sha256=training.config_hash,
                    model_config_sha256=model.config_hash,
                    parameter_count=model.parameter_breakdown().total,
                    forward_flops_per_token=architecture_forward_flops,
                    tokens_per_optimizer_step=training.tokens_per_optimizer_step,
                    scheduler_total_steps=training.optimization.total_steps,
                    screening=screening,
                    full=full,
                    checkpoint_directory=checkpoints.as_posix(),
                    jsonl_log=(lineage_root / "events.jsonl").as_posix(),
                    tensorboard_directory=(lineage_root / "tensorboard").as_posix(),
                    parquet_log=(lineage_root / "events.parquet").as_posix(),
                    screening_checkpoint=(
                        checkpoints / f"step-{screening.optimizer_step:012d}"
                    ).as_posix(),
                    estimated_checkpoint_bytes=checkpoint_bytes,
                    estimated_retained_checkpoint_bytes_upper_bound=(
                        checkpoint_bytes * (training.keep_latest_checkpoints + 1)
                    ),
                )
            )
    return ArchitectureStudyPlan(
        shard_manifest_sha256=_file_sha256(build_path),
        tokenizer_sha256=build.tokenizer_hash,
        tokenizer_vocab_size=16_000,
        sequence_length=sequence_length,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        world_size=1,
        screening_dense_reference_tokens=screening_dense_reference_tokens,
        full_dense_reference_tokens=full_dense_reference_tokens,
        lineages=tuple(lineages),
    )


def write_architecture_study_plan(
    path: str | Path,
    plan: ArchitectureStudyPlan,
) -> None:
    """Atomically write one canonical plan, replacing only a complete plan."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise ArchitectureStudyError("incomplete architecture-study plan exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(plan.canonical_json().encode())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
