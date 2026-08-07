"""Durable one-epoch CUDA supervised fine-tuning for the selected dense model."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import save_file
from tqdm import tqdm  # type: ignore[import-untyped]

from lm_from_zero.models import DenseModelVariant, Olmo2Config, Olmo2ForCausalLM
from lm_from_zero.post_training.chat import Conversation
from lm_from_zero.post_training.dataset import SFTMixManifest, SFTRecord
from lm_from_zero.post_training.sft import (
    SFTBatch,
    SupervisedChatExample,
    assistant_only_causal_loss,
    collate_supervised_chat,
    render_supervised_chat,
)
from lm_from_zero.tokenizer.bpe import ByteBPE
from lm_from_zero.training import (
    OptimizationConfig,
    TrainingMetricSinks,
    capture_rng_state,
    clip_gradients,
    load_checkpoint_model,
    set_learning_rate,
    validate_checkpoint,
)

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
SFTVariant = Literal["hybrid_muon", "mha", "layer_norm", "tied_embeddings"]


class SFTTrainingError(RuntimeError):
    """Raised when a full SFT run cannot safely start or finish."""


class SFTTrainingConfig(BaseModel):
    """Pinned execution policy for the first full SFT run."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-sft-training-config"] = (
        "lm-from-zero-sft-training-config"
    )
    format_version: Literal[1] = 1
    epochs: Literal[1] = 1
    max_length: Annotated[int, Field(gt=1)] = 1_024
    learning_rate: Annotated[float, Field(gt=0)] = 2e-5
    batch_size: Annotated[int, Field(gt=0)] = 8
    bucket_size: Annotated[int, Field(gt=0)] = 256
    seed: int = 1_337
    beta1: Annotated[float, Field(gt=0, lt=1)] = 0.9
    beta2: Annotated[float, Field(gt=0, lt=1)] = 0.95
    epsilon: Annotated[float, Field(gt=0)] = 1e-8
    weight_decay: Annotated[float, Field(ge=0)] = 0.1
    gradient_clip_norm: Annotated[float, Field(gt=0)] = 1.0
    warmup_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.015
    minimum_lr_ratio: Annotated[float, Field(gt=0, le=1)] = 0.1
    checkpoint_every_steps: Annotated[int, Field(gt=0)] = 1_000
    device: Literal["cuda"] = "cuda"
    precision: Literal["bf16"] = "bf16"


class SFTArtifactRecord(BaseModel):
    """Integrity record for one SFT checkpoint file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(min_length=1)
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256


class SFTCheckpointManifest(BaseModel):
    """Atomic, provenance-bound SFT checkpoint manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-sft-checkpoint"] = "lm-from-zero-sft-checkpoint"
    format_version: Literal[1] = 1
    created_at_utc: datetime
    checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    complete: bool
    variant: SFTVariant
    model_variant: DenseModelVariant
    source_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    source_checkpoint_manifest_sha256: Sha256
    dataset_manifest_sha256: Sha256
    dataset_records_sha256: Sha256
    dataset_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40,64}$")]
    tokenizer_sha256: Sha256
    model_config_sha256: Sha256
    training_config_sha256: Sha256
    optimizer_step: Annotated[int, Field(gt=0)]
    examples_consumed: Annotated[int, Field(gt=0)]
    tokens_consumed: Annotated[int, Field(gt=0)]
    model_artifact: SFTArtifactRecord
    recovery_artifact: SFTArtifactRecord


class SFTRunManifest(BaseModel):
    """Canonical final manifest for one completed full SFT run."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-sft-run-manifest"] = "lm-from-zero-sft-run-manifest"
    format_version: Literal[1] = 1
    completed_at_utc: datetime
    variant: SFTVariant
    model_variant: DenseModelVariant
    source_checkpoint_directory: str = (
        "artifacts/dense-ablations-clean-20260807/"
        "hybrid_muon/seed-2027/checkpoints/step-000000012208"
    )
    source_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    source_checkpoint_manifest_sha256: Sha256
    dataset_manifest_sha256: Sha256
    dataset_records_sha256: Sha256
    dataset_revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40,64}$")]
    tokenizer_sha256: Sha256
    model_config_sha256: Sha256
    training_config_sha256: Sha256
    total_optimizer_steps: Annotated[int, Field(gt=0)]
    examples_consumed: Annotated[int, Field(gt=0)]
    tokens_consumed: Annotated[int, Field(gt=0)]
    truncated_examples: Annotated[int, Field(ge=0)]
    final_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    final_checkpoint_manifest_sha256: Sha256
    metrics_jsonl: str = Field(min_length=1)
    metrics_parquet: str = Field(min_length=1)
    training_config: SFTTrainingConfig
    optimization_config: OptimizationConfig

    def canonical_bytes(self) -> bytes:
        """Return stable bytes for the final run manifest."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


class SFTTrainingReport(BaseModel):
    """Generated summary of measured full-run training metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-sft-training-report"] = (
        "lm-from-zero-sft-training-report"
    )
    format_version: Literal[1] = 1
    completed_at_utc: datetime
    run_manifest_sha256: Sha256
    variant: SFTVariant
    model_variant: DenseModelVariant
    optimizer_steps: Annotated[int, Field(gt=0)]
    examples_consumed: Annotated[int, Field(gt=0)]
    tokens_consumed: Annotated[int, Field(gt=0)]
    first_loss: Annotated[float, Field(ge=0)]
    final_loss: Annotated[float, Field(ge=0)]
    final_learning_rate: Annotated[float, Field(gt=0)]
    mean_last_100_loss: Annotated[float, Field(ge=0)]
    elapsed_seconds: Annotated[float, Field(gt=0)]
    final_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    final_checkpoint_manifest_sha256: Sha256

    def canonical_bytes(self) -> bytes:
        """Return stable bytes for the generated report."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


class _BatchItem:
    def __init__(
        self,
        batch: SFTBatch,
        *,
        example_count: int,
        token_count: int,
        truncated_count: int,
    ) -> None:
        self.batch = batch
        self.example_count = example_count
        self.token_count = token_count
        self.truncated_count = truncated_count


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise SFTTrainingError(f"incomplete artifact exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _iter_sft_batches(
    records_path: Path,
    tokenizer: ByteBPE,
    *,
    batch_size: int,
    bucket_size: int,
    max_length: int,
) -> Iterator[_BatchItem]:
    bucket: list[tuple[int, SupervisedChatExample]] = []
    source_order = 0

    def emit_bucket() -> Iterator[_BatchItem]:
        bucket.sort(key=lambda item: (len(item[1].input_ids), item[0]))
        for start in range(0, len(bucket), batch_size):
            selected = [item[1] for item in bucket[start : start + batch_size]]
            batch = collate_supervised_chat(selected)
            yield _BatchItem(
                batch,
                example_count=len(selected),
                token_count=sum(len(example.input_ids) - 1 for example in selected),
                truncated_count=sum(example.truncated for example in selected),
            )

    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = SFTRecord.model_validate_json(line)
            example = render_supervised_chat(
                Conversation(messages=record.messages),
                tokenizer,
                max_length=max_length,
                truncation="left",
            )
            bucket.append((source_order, example))
            source_order += 1
            if len(bucket) >= bucket_size:
                yield from emit_bucket()
                bucket.clear()
    if bucket:
        yield from emit_bucket()


def _artifact_record(path: Path) -> SFTArtifactRecord:
    return SFTArtifactRecord(
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _write_checkpoint(
    checkpoint_root: Path,
    *,
    step: int,
    complete: bool,
    variant: SFTVariant,
    model_variant: DenseModelVariant,
    model: Olmo2ForCausalLM,
    optimizer: torch.optim.Optimizer,
    source_checkpoint_id: str,
    source_checkpoint_manifest_sha256: str,
    dataset_manifest_sha256: str,
    dataset_records_sha256: str,
    dataset_revision: str,
    tokenizer_sha256: str,
    model_config_sha256: str,
    training_config_sha256: str,
    examples_consumed: int,
    tokens_consumed: int,
) -> tuple[Path, SFTCheckpointManifest, str]:
    checkpoint_id = f"step-{step:012d}"
    destination = checkpoint_root / checkpoint_id
    if destination.exists():
        raise SFTTrainingError(f"SFT checkpoint already exists: {destination}")
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{checkpoint_id}-", dir=checkpoint_root))
    try:
        model_path = temporary / "model.safetensors"
        tensors = {
            name: tensor.detach().to(device="cpu").contiguous().clone()
            for name, tensor in model.state_dict().items()
        }
        save_file(
            tensors,
            str(model_path),
            metadata={
                "format": "lm-from-zero-sft-checkpoint",
                "format_version": "1",
            },
        )
        recovery_path = temporary / "recovery.pt"
        torch.save(
            {
                "format": "lm-from-zero-sft-checkpoint",
                "format_version": 1,
                "optimizer_state": optimizer.state_dict(),
                "rng_state": capture_rng_state(),
            },
            recovery_path,
        )
        manifest = SFTCheckpointManifest(
            created_at_utc=datetime.now(UTC),
            checkpoint_id=checkpoint_id,
            complete=complete,
            variant=variant,
            model_variant=model_variant,
            source_checkpoint_id=source_checkpoint_id,
            source_checkpoint_manifest_sha256=source_checkpoint_manifest_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            dataset_records_sha256=dataset_records_sha256,
            dataset_revision=dataset_revision,
            tokenizer_sha256=tokenizer_sha256,
            model_config_sha256=model_config_sha256,
            training_config_sha256=training_config_sha256,
            optimizer_step=step,
            examples_consumed=examples_consumed,
            tokens_consumed=tokens_consumed,
            model_artifact=_artifact_record(model_path),
            recovery_artifact=_artifact_record(recovery_path),
        )
        _atomic_write(
            temporary / "manifest.json",
            json.dumps(
                manifest.model_dump(mode="json"),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n",
        )
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    manifest_path = destination / "manifest.json"
    return destination, manifest, _sha256_file(manifest_path)


def _build_training_config_hash(config: SFTTrainingConfig) -> str:
    encoded = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


def _build_optimization_config(
    config: SFTTrainingConfig,
    *,
    total_steps: int,
) -> OptimizationConfig:
    return OptimizationConfig(
        learning_rate=config.learning_rate,
        beta1=config.beta1,
        beta2=config.beta2,
        epsilon=config.epsilon,
        weight_decay=config.weight_decay,
        gradient_clip_norm=config.gradient_clip_norm,
        total_steps=total_steps,
        warmup_fraction=config.warmup_fraction,
        minimum_lr_ratio=config.minimum_lr_ratio,
    )


def run_sft_training(
    *,
    dataset_manifest_path: str | Path,
    tokenizer_path: str | Path,
    source_checkpoint: str | Path,
    output_directory: str | Path,
    report_path: str | Path,
    variant: SFTVariant = "hybrid_muon",
    model_variant: DenseModelVariant = "baseline",
    config: SFTTrainingConfig | None = None,
) -> SFTTrainingReport:
    """Run one complete deterministic SFT epoch and publish its artifacts."""

    if not torch.cuda.is_available():
        raise SFTTrainingError("CUDA is required for the full SFT run")
    training_config = SFTTrainingConfig() if config is None else config
    manifest_path = Path(dataset_manifest_path)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise SFTTrainingError(f"SFT output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    dataset_manifest = SFTMixManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    records_path = manifest_path.parent / dataset_manifest.records_jsonl
    if _sha256_file(records_path) != dataset_manifest.records_sha256:
        raise SFTTrainingError("SFT records do not match their manifest")
    tokenizer = ByteBPE.load(tokenizer_path)
    checkpoint_path = Path(source_checkpoint)
    checkpoint_manifest = validate_checkpoint(checkpoint_path)
    if checkpoint_manifest.binding.architecture != "olmo2":
        raise SFTTrainingError("full SFT requires an OLMo2 source checkpoint")
    if tokenizer.model_hash != checkpoint_manifest.binding.tokenizer_sha256:
        raise SFTTrainingError("tokenizer does not match the source checkpoint")
    model_config = Olmo2Config.model_validate(
        checkpoint_manifest.binding.resolved_model_config
    )
    if training_config.max_length > model_config.max_position_embeddings:
        raise SFTTrainingError("SFT context exceeds the source model context")
    total_steps = math.ceil(
        dataset_manifest.selected_examples / training_config.batch_size
    )
    optimization_config = _build_optimization_config(
        training_config,
        total_steps=total_steps,
    )
    training_config_hash = _build_training_config_hash(training_config)
    model = Olmo2ForCausalLM(model_config, variant=model_variant)
    load_checkpoint_model(
        checkpoint_path,
        model=model,
        expected_binding=checkpoint_manifest.binding,
    )
    device = torch.device(training_config.device)
    torch.manual_seed(training_config.seed)
    torch.cuda.manual_seed_all(training_config.seed)
    model.to(device)
    model.train()
    from lm_from_zero.training.optimization import build_adamw

    optimizer, _ = build_adamw(model, optimization_config, device_type="cuda")
    metrics_jsonl = output / "metrics.jsonl"
    metrics_parquet = output / "metrics.parquet"
    checkpoints_root = output / "checkpoints"
    source_manifest_hash = _sha256_file(checkpoint_path / "manifest.json")
    sinks = TrainingMetricSinks(
        jsonl_path=metrics_jsonl,
        tensorboard_directory=None,
        parquet_path=metrics_parquet,
        resume_optimizer_step=0,
        durable_every_steps=50,
        durable_every_seconds=5.0,
    )
    started = time.perf_counter()
    optimizer_step = 0
    examples_consumed = 0
    tokens_consumed = 0
    truncated_examples = 0
    losses: list[float] = []
    final_checkpoint: tuple[Path, SFTCheckpointManifest, str] | None = None
    try:
        sinks.append_event(
            {
                "event": "run_start",
                "variant": variant,
                "model_variant": model_variant,
                "dataset_manifest_sha256": _sha256_file(manifest_path),
                "dataset_records_sha256": dataset_manifest.records_sha256,
                "source_checkpoint_id": checkpoint_manifest.lineage.checkpoint_id,
                "training_config": training_config.model_dump(mode="json"),
                "training_config_sha256": training_config_hash,
                "total_optimizer_steps": total_steps,
            }
        )
        progress = tqdm(
            total=dataset_manifest.selected_examples,
            desc=f"SFT {variant}",
            unit="examples",
        )
        for item in _iter_sft_batches(
            records_path,
            tokenizer,
            batch_size=training_config.batch_size,
            bucket_size=training_config.bucket_size,
            max_length=training_config.max_length,
        ):
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            learning_rate = set_learning_rate(
                optimizer,
                optimization_config,
                optimizer_step - 1,
            )
            batch = item.batch
            input_ids = batch.input_ids.to(device)
            labels = batch.labels.to(device)
            attention_mask = batch.attention_mask.to(device)
            torch.cuda.reset_peak_memory_stats(device)
            step_started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_values = model(
                    input_ids,
                    attention_mask=attention_mask,
                )
                loss = assistant_only_causal_loss(output_values.logits, labels)
            if not bool(torch.isfinite(loss)):
                raise SFTTrainingError(f"non-finite SFT loss at step {optimizer_step}")
            loss.backward()  # type: ignore[no-untyped-call]
            gradient_norm = float(
                clip_gradients(model, optimization_config.gradient_clip_norm)
                .detach()
                .cpu()
                .item()
            )
            if not math.isfinite(gradient_norm):
                raise SFTTrainingError(
                    f"non-finite SFT gradient at step {optimizer_step}"
                )
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - step_started
            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            examples_consumed += item.example_count
            tokens_consumed += item.token_count
            truncated_examples += item.truncated_count
            progress.update(item.example_count)
            progress.set_postfix(loss=f"{loss_value:.4f}", lr=f"{learning_rate:.2e}")
            sinks.log_optimizer_step(
                {
                    "optimizer_step": optimizer_step,
                    "loss": loss_value,
                    "learning_rate": learning_rate,
                    "gradient_norm": gradient_norm,
                    "tokens_consumed": tokens_consumed,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": item.token_count / max(elapsed, 1e-9),
                    "peak_cuda_memory_allocated_bytes": torch.cuda.max_memory_allocated(
                        device
                    ),
                    "peak_cuda_memory_reserved_bytes": torch.cuda.max_memory_reserved(
                        device
                    ),
                }
            )
            if (
                optimizer_step % training_config.checkpoint_every_steps == 0
                or optimizer_step == total_steps
            ):
                sinks.snapshot()
                final_checkpoint = _write_checkpoint(
                    checkpoints_root,
                    step=optimizer_step,
                    complete=optimizer_step == total_steps,
                    variant=variant,
                    model_variant=model_variant,
                    model=model,
                    optimizer=optimizer,
                    source_checkpoint_id=checkpoint_manifest.lineage.checkpoint_id,
                    source_checkpoint_manifest_sha256=source_manifest_hash,
                    dataset_manifest_sha256=_sha256_file(manifest_path),
                    dataset_records_sha256=dataset_manifest.records_sha256,
                    dataset_revision=dataset_manifest.dataset_revision,
                    tokenizer_sha256=tokenizer.model_hash,
                    model_config_sha256=model_config.config_hash,
                    training_config_sha256=training_config_hash,
                    examples_consumed=examples_consumed,
                    tokens_consumed=tokens_consumed,
                )
            del output_values, loss, input_ids, labels, attention_mask
        progress.close()
        if examples_consumed != dataset_manifest.selected_examples:
            raise SFTTrainingError("SFT epoch did not consume the complete manifest")
        sinks.append_event(
            {
                "event": "run_complete",
                "optimizer_step": optimizer_step,
                "examples_consumed": examples_consumed,
                "tokens_consumed": tokens_consumed,
                "truncated_examples": truncated_examples,
            }
        )
        sinks.close()
    except Exception:
        sinks.abort()
        raise
    finally:
        del optimizer, model
        torch.cuda.empty_cache()
    if final_checkpoint is None:
        raise SFTTrainingError("SFT run finished without a final checkpoint")
    _final_checkpoint_path, final_checkpoint_manifest, final_checkpoint_hash = (
        final_checkpoint
    )
    run_manifest = SFTRunManifest(
        completed_at_utc=datetime.now(UTC),
        variant=variant,
        model_variant=model_variant,
        source_checkpoint_directory=str(checkpoint_path),
        source_checkpoint_id=checkpoint_manifest.lineage.checkpoint_id,
        source_checkpoint_manifest_sha256=source_manifest_hash,
        dataset_manifest_sha256=_sha256_file(manifest_path),
        dataset_records_sha256=dataset_manifest.records_sha256,
        dataset_revision=dataset_manifest.dataset_revision,
        tokenizer_sha256=tokenizer.model_hash,
        model_config_sha256=model_config.config_hash,
        training_config_sha256=training_config_hash,
        total_optimizer_steps=optimizer_step,
        examples_consumed=examples_consumed,
        tokens_consumed=tokens_consumed,
        truncated_examples=truncated_examples,
        final_checkpoint_id=final_checkpoint_manifest.checkpoint_id,
        final_checkpoint_manifest_sha256=final_checkpoint_hash,
        metrics_jsonl=str(metrics_jsonl.name),
        metrics_parquet=str(metrics_parquet.name),
        training_config=training_config,
        optimization_config=optimization_config,
    )
    run_manifest_path = output / "manifest.json"
    _atomic_write(run_manifest_path, run_manifest.canonical_bytes() + b"\n")
    report = SFTTrainingReport(
        completed_at_utc=run_manifest.completed_at_utc,
        run_manifest_sha256=_sha256_file(run_manifest_path),
        variant=variant,
        model_variant=model_variant,
        optimizer_steps=optimizer_step,
        examples_consumed=examples_consumed,
        tokens_consumed=tokens_consumed,
        first_loss=losses[0],
        final_loss=losses[-1],
        final_learning_rate=optimization_config.learning_rate_at(optimizer_step - 1),
        mean_last_100_loss=sum(losses[-100:]) / min(100, len(losses)),
        elapsed_seconds=time.perf_counter() - started,
        final_checkpoint_id=final_checkpoint_manifest.checkpoint_id,
        final_checkpoint_manifest_sha256=final_checkpoint_hash,
    )
    _atomic_write(Path(report_path), report.canonical_bytes() + b"\n")
    return report
