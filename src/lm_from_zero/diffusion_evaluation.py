"""Deterministic architecture-specific evaluation for masked diffusion."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.nn import functional as F

from lm_from_zero.data import Split
from lm_from_zero.models import (
    MaskedDiffusionForMaskedLM,
    base_pretraining_eligible_mask,
    corrupt_for_diffusion,
)
from lm_from_zero.progress import ProgressReporter
from lm_from_zero.training.data import BatchCursor, ShardBatchSource

DeviceKind = Literal["cpu", "cuda"]
Precision = Literal["fp32", "bf16"]
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class DiffusionEvaluationError(RuntimeError):
    """Raised when diffusion evaluation is unsafe or incompatible."""


class DiffusionEvaluationConfig(BaseModel):
    """Bounded deterministic Monte Carlo evaluation policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_batches: Annotated[int, Field(gt=0)]
    corruption_samples_per_batch: Annotated[int, Field(gt=0)] = 1
    seed: int = 1_337
    device: DeviceKind = "cpu"
    precision: Precision = "fp32"


class DiffusionEvaluationResult(BaseModel):
    """Recorded masked reconstruction and variational-bound estimates."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-diffusion-evaluation"] = (
        "lm-from-zero-diffusion-evaluation"
    )
    format_version: Literal[2] = 2
    evaluated_at_utc: datetime
    source_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    source_checkpoint_manifest_sha256: str = Field(pattern=SHA256_PATTERN)
    device: DeviceKind
    split: Split
    model_config_sha256: str
    shard_manifest_sha256: str
    tokenizer_sha256: str
    seed: int
    batch_count: Annotated[int, Field(gt=0)]
    source_sequence_count: Annotated[int, Field(gt=0)]
    corruption_samples_per_batch: Annotated[int, Field(gt=0)]
    evaluated_example_count: Annotated[int, Field(gt=0)]
    model_forwards: Annotated[int, Field(gt=0)]
    eligible_token_count: Annotated[int, Field(gt=0)]
    masked_token_count: Annotated[int, Field(gt=0)]
    mean_mask_rate: Annotated[float, Field(gt=0, le=1)]
    masked_reconstruction_loss_nats: Annotated[float, Field(ge=0)]
    variational_upper_bound_nats: Annotated[float, Field(ge=0)]
    causal_perplexity_applicable: Literal[False] = False
    elapsed_seconds: Annotated[float, Field(gt=0)]
    masked_tokens_per_second: Annotated[float, Field(gt=0)]
    cursor_before: BatchCursor
    cursor_after: BatchCursor

    def canonical_json(self) -> str:
        """Return stable machine-readable evaluation evidence."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


def _validate_binding(
    model: MaskedDiffusionForMaskedLM,
    source: ShardBatchSource,
    config: DiffusionEvaluationConfig,
    cursor: BatchCursor,
) -> None:
    if source.build.tokenizer_hash != model.config.tokenizer_hash:
        raise DiffusionEvaluationError(
            "model tokenizer does not match evaluation shards"
        )
    if source.build.tokenizer_vocab_size != model.config.vocab_size:
        raise DiffusionEvaluationError(
            "model vocabulary does not match evaluation shards"
        )
    if source.config.sequence_length > model.config.max_position_embeddings:
        raise DiffusionEvaluationError(
            "evaluation sequence length exceeds model context"
        )
    if config.device == "cuda" and not torch.cuda.is_available():
        raise DiffusionEvaluationError(
            "CUDA evaluation was requested but CUDA is unavailable"
        )
    remaining_windows = (
        len(source.rank_window_ids(cursor.epoch)) - cursor.next_local_window
    )
    requested_windows = config.max_batches * source.config.micro_batch_size
    if requested_windows > remaining_windows:
        raise DiffusionEvaluationError(
            "evaluation would wrap into a repeated shard epoch"
        )


def evaluate_diffusion(
    model: MaskedDiffusionForMaskedLM,
    source: ShardBatchSource,
    config: DiffusionEvaluationConfig,
    *,
    source_checkpoint_id: str,
    source_checkpoint_manifest_sha256: str,
    cursor: BatchCursor | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> DiffusionEvaluationResult:
    """Evaluate fixed windows with a dedicated seeded corruption generator."""

    before = source.initial_cursor() if cursor is None else cursor
    _validate_binding(model, source, config, before)
    device = torch.device(config.device)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    model.to(device)
    was_training = model.training
    model.eval()
    total_reconstruction_nll = 0.0
    total_variational_bound = 0.0
    eligible_tokens = 0
    masked_tokens = 0
    evaluated_examples = 0
    model_forwards = 0
    current = before
    started = clock()
    progress = ProgressReporter("diffusion evaluation")
    progress.phase(
        "evaluating",
        total=config.max_batches * config.corruption_samples_per_batch,
    )
    completed_work = 0
    completed = False
    try:
        with torch.no_grad():
            for batch_index in range(config.max_batches):
                batch = source.next_batch(current)
                current = batch.cursor_after
                original = batch.input_ids.to(device)
                attention_mask = original != model.config.pad_token_id
                eligible = base_pretraining_eligible_mask(
                    original,
                    attention_mask,
                    pad_token_id=model.config.pad_token_id,
                    bos_token_id=model.config.bos_token_id,
                )
                if torch.any(~eligible.any(dim=1)):
                    raise DiffusionEvaluationError(
                        "every evaluation sequence must contain an eligible token"
                    )
                for sample_index in range(config.corruption_samples_per_batch):
                    corrupted = corrupt_for_diffusion(
                        original,
                        eligible,
                        mask_token_id=model.config.mask_token_id,
                        epsilon=model.config.corruption_epsilon,
                        generator=generator,
                    )
                    with torch.autocast(
                        device_type=config.device,
                        dtype=torch.bfloat16,
                        enabled=config.precision == "bf16",
                    ):
                        output = model(
                            corrupted.input_ids,
                            attention_mask=attention_mask,
                            labels=corrupted.labels,
                            eligible_mask=corrupted.eligible_mask,
                            time=corrupted.time,
                        )
                    loss = cast(Tensor, output.loss)
                    supervised = corrupted.labels != -100
                    count = int(supervised.sum().item())
                    if count <= 0 or not bool(torch.isfinite(loss)):
                        raise DiffusionEvaluationError(
                            "diffusion evaluation produced no finite objective"
                        )
                    reconstruction = F.cross_entropy(
                        output.logits[supervised].float(),
                        corrupted.labels[supervised],
                        reduction="sum",
                    )
                    if not bool(torch.isfinite(reconstruction)):
                        raise DiffusionEvaluationError(
                            "masked reconstruction loss is not finite"
                        )
                    batch_size = original.shape[0]
                    total_reconstruction_nll += float(reconstruction)
                    total_variational_bound += float(loss.float()) * batch_size
                    eligible_tokens += int(eligible.sum().item())
                    masked_tokens += count
                    evaluated_examples += batch_size
                    model_forwards += 1
                    completed_work += 1
                    progress.update(
                        completed_work,
                        fields={
                            "batch": f"{batch_index + 1}/{config.max_batches}",
                            "sample": sample_index + 1,
                            "loss": float(loss.float()),
                            "forwards": model_forwards,
                        },
                    )
        elapsed = clock() - started
        if elapsed <= 0:
            raise DiffusionEvaluationError("evaluation clock did not advance")
        source_sequence_count = config.max_batches * source.config.micro_batch_size
        result = DiffusionEvaluationResult(
            evaluated_at_utc=datetime.now(UTC),
            source_checkpoint_id=source_checkpoint_id,
            source_checkpoint_manifest_sha256=source_checkpoint_manifest_sha256,
            device=config.device,
            split=source.config.split,
            model_config_sha256=model.config.config_hash,
            shard_manifest_sha256=source.build_manifest_sha256,
            tokenizer_sha256=source.build.tokenizer_hash,
            seed=config.seed,
            batch_count=config.max_batches,
            source_sequence_count=source_sequence_count,
            corruption_samples_per_batch=config.corruption_samples_per_batch,
            evaluated_example_count=evaluated_examples,
            model_forwards=model_forwards,
            eligible_token_count=eligible_tokens,
            masked_token_count=masked_tokens,
            mean_mask_rate=masked_tokens / eligible_tokens,
            masked_reconstruction_loss_nats=total_reconstruction_nll / masked_tokens,
            variational_upper_bound_nats=(total_variational_bound / evaluated_examples),
            elapsed_seconds=elapsed,
            masked_tokens_per_second=masked_tokens / elapsed,
            cursor_before=before,
            cursor_after=current,
        )
        completed = True
        return result
    finally:
        model.train(was_training)
        progress.finish("complete" if completed else "failed")


def append_diffusion_evaluation_result(
    path: str | Path,
    result: DiffusionEvaluationResult,
) -> None:
    """Append one canonical evaluation result and flush it durably."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(result.canonical_json())
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
