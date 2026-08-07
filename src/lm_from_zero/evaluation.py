"""Deterministic causal-loss evaluation over validated token shards."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from lm_from_zero.data import Split
from lm_from_zero.models import Mamba2ForCausalLM, Olmo2ForCausalLM
from lm_from_zero.progress import ProgressReporter
from lm_from_zero.training.data import BatchCursor, ShardBatchSource

DeviceKind = Literal["cpu", "cuda"]
Precision = Literal["fp32", "bf16"]
CausalModel = Olmo2ForCausalLM | Mamba2ForCausalLM


class EvaluationError(RuntimeError):
    """Raised when a checkpoint evaluation request is unsafe or incompatible."""


class CausalEvaluationConfig(BaseModel):
    """Bounded deterministic evaluation policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_batches: Annotated[int, Field(gt=0)]
    device: DeviceKind = "cpu"
    precision: Precision = "fp32"


class CausalEvaluationResult(BaseModel):
    """Recorded aggregate causal loss and throughput."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-causal-evaluation"] = "lm-from-zero-causal-evaluation"
    format_version: Literal[1] = 1
    evaluated_at_utc: datetime
    split: Split
    model_config_sha256: str
    shard_manifest_sha256: str
    tokenizer_sha256: str
    batch_count: Annotated[int, Field(gt=0)]
    sequence_count: Annotated[int, Field(gt=0)]
    predicted_token_count: Annotated[int, Field(gt=0)]
    mean_loss: Annotated[float, Field(ge=0)]
    perplexity: Annotated[float, Field(gt=0)]
    elapsed_seconds: Annotated[float, Field(gt=0)]
    predicted_tokens_per_second: Annotated[float, Field(gt=0)]
    cursor_before: BatchCursor
    cursor_after: BatchCursor

    def canonical_json(self) -> str:
        """Return a stable machine-readable evaluation record."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


def _validate_evaluation_binding(
    model: CausalModel,
    source: ShardBatchSource,
    config: CausalEvaluationConfig,
    cursor: BatchCursor,
) -> None:
    if source.build.tokenizer_hash != model.config.tokenizer_hash:
        raise EvaluationError("model tokenizer does not match evaluation shards")
    if source.build.tokenizer_vocab_size != model.config.vocab_size:
        raise EvaluationError("model vocabulary does not match evaluation shards")
    if source.config.sequence_length > model.config.max_position_embeddings:
        raise EvaluationError("evaluation sequence length exceeds model context")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise EvaluationError("CUDA evaluation was requested but CUDA is unavailable")
    remaining_windows = (
        len(source.rank_window_ids(cursor.epoch)) - cursor.next_local_window
    )
    requested_windows = config.max_batches * source.config.micro_batch_size
    if requested_windows > remaining_windows:
        raise EvaluationError("evaluation would wrap into a repeated shard epoch")


def evaluate_causal_loss(
    model: CausalModel,
    source: ShardBatchSource,
    config: CausalEvaluationConfig,
    *,
    cursor: BatchCursor | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> CausalEvaluationResult:
    """Evaluate fixed non-repeating batches and preserve model train/eval mode."""

    before = source.initial_cursor() if cursor is None else cursor
    _validate_evaluation_binding(model, source, config, before)
    device = torch.device(config.device)
    model.to(device)
    was_training = model.training
    model.eval()
    total_negative_log_likelihood = 0.0
    predicted_tokens = 0
    current = before
    started = clock()
    progress = ProgressReporter("causal evaluation")
    progress.phase("evaluating", total=config.max_batches)
    completed = False
    try:
        with torch.no_grad():
            for batch_index in range(config.max_batches):
                batch = source.next_batch(current)
                current = batch.cursor_after
                input_ids = batch.input_ids.to(device)
                labels = batch.labels.to(device)
                with torch.autocast(
                    device_type=config.device,
                    dtype=torch.bfloat16,
                    enabled=config.precision == "bf16",
                ):
                    loss = cast(Tensor, model(input_ids, labels=labels).loss)
                count = int((labels[:, 1:] != -100).sum())
                if count <= 0 or not bool(torch.isfinite(loss)):
                    raise EvaluationError("evaluation batch has no finite loss")
                total_negative_log_likelihood += float(loss.float()) * count
                predicted_tokens += count
                progress.update(
                    batch_index + 1,
                    fields={
                        "loss": float(loss.float()),
                        "predicted_tokens": predicted_tokens,
                    },
                )
        elapsed = clock() - started
        if elapsed <= 0:
            raise EvaluationError("evaluation clock did not advance")
        mean_loss = total_negative_log_likelihood / predicted_tokens
        try:
            perplexity = math.exp(mean_loss)
        except OverflowError as error:
            raise EvaluationError("evaluation perplexity overflowed") from error
        sequence_count = config.max_batches * source.config.micro_batch_size
        result = CausalEvaluationResult(
            evaluated_at_utc=datetime.now(UTC),
            split=source.config.split,
            model_config_sha256=model.config.config_hash,
            shard_manifest_sha256=source.build_manifest_sha256,
            tokenizer_sha256=source.build.tokenizer_hash,
            batch_count=config.max_batches,
            sequence_count=sequence_count,
            predicted_token_count=predicted_tokens,
            mean_loss=mean_loss,
            perplexity=perplexity,
            elapsed_seconds=elapsed,
            predicted_tokens_per_second=predicted_tokens / elapsed,
            cursor_before=before,
            cursor_after=current,
        )
        completed = True
        return result
    finally:
        model.train(was_training)
        progress.finish("complete" if completed else "failed")


def append_evaluation_result(path: str | Path, result: CausalEvaluationResult) -> None:
    """Append one canonical evaluation result and flush it durably."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(result.canonical_json())
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
