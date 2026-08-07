"""Durable one-epoch DPO training from a validated SFT checkpoint."""

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
from typing import Annotated, Literal, TextIO

import torch
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import save_file
from torch import Tensor
from tqdm import tqdm  # type: ignore[import-untyped]

from lm_from_zero.models import DenseModelVariant, Olmo2Config, Olmo2ForCausalLM
from lm_from_zero.post_training.chat import DEFAULT_CHAT_TEMPLATE
from lm_from_zero.post_training.dpo import (
    DPOConfig,
    DPOObjectiveOutput,
    PreferencePairExample,
    PreferenceSequence,
    ReferenceLogProbCacheIdentity,
    dpo_objective,
    masked_sequence_logprob,
    render_preference_pair,
)
from lm_from_zero.post_training.preference_dataset import (
    PreferenceMixManifest,
    PreferenceRecord,
)
from lm_from_zero.post_training.sft_train import (
    SFTCheckpointManifest,
    SFTRunManifest,
)
from lm_from_zero.tokenizer.bpe import SPECIAL_TOKEN_IDS, ByteBPE
from lm_from_zero.training import (
    OptimizationConfig,
    TrainingMetricSinks,
    capture_rng_state,
    clip_gradients,
    validate_checkpoint,
)
from lm_from_zero.training.optimization import build_adamw, set_learning_rate

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40,64}$")]


class DPOTrainingError(RuntimeError):
    """Raised when a full DPO run cannot safely start or finish."""


class DPOTrainingConfig(BaseModel):
    """Pinned execution policy for the full one-epoch DPO run."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-dpo-training-config"] = (
        "lm-from-zero-dpo-training-config"
    )
    format_version: Literal[1] = 1
    epochs: Literal[1] = 1
    max_length: Annotated[int, Field(gt=1)] = 1_024
    learning_rate: Annotated[float, Field(gt=0)] = 5e-7
    beta: Annotated[float, Field(gt=0)] = 0.1
    batch_size: Annotated[int, Field(gt=0)] = 2
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


class DPOArtifactRecord(BaseModel):
    """Integrity record for one DPO checkpoint file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(min_length=1)
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256


class DPOCheckpointManifest(BaseModel):
    """Atomic, provenance-bound DPO checkpoint manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-dpo-checkpoint"] = "lm-from-zero-dpo-checkpoint"
    format_version: Literal[1] = 1
    created_at_utc: datetime
    checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    complete: bool
    model_variant: DenseModelVariant
    source_sft_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    source_sft_checkpoint_manifest_sha256: Sha256
    dataset_manifest_sha256: Sha256
    dataset_records_sha256: Sha256
    dataset_revision: Revision
    tokenizer_sha256: Sha256
    model_config_sha256: Sha256
    dpo_config_sha256: Sha256
    training_config_sha256: Sha256
    reference_cache_key: Sha256
    optimizer_step: Annotated[int, Field(gt=0)]
    pairs_consumed: Annotated[int, Field(gt=0)]
    chosen_response_tokens: Annotated[int, Field(gt=0)]
    rejected_response_tokens: Annotated[int, Field(gt=0)]
    truncated_pairs: Annotated[int, Field(ge=0)]
    model_artifact: DPOArtifactRecord
    recovery_artifact: DPOArtifactRecord


class DPORunManifest(BaseModel):
    """Canonical final manifest for one completed DPO run."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-dpo-run-manifest"] = "lm-from-zero-dpo-run-manifest"
    format_version: Literal[1] = 1
    completed_at_utc: datetime
    model_variant: DenseModelVariant
    source_sft_checkpoint_directory: str = Field(min_length=1)
    source_sft_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    source_sft_checkpoint_manifest_sha256: Sha256
    dataset_manifest_sha256: Sha256
    dataset_records_sha256: Sha256
    dataset_revision: Revision
    tokenizer_sha256: Sha256
    model_config_sha256: Sha256
    dpo_config_sha256: Sha256
    training_config_sha256: Sha256
    total_optimizer_steps: Annotated[int, Field(gt=0)]
    pairs_consumed: Annotated[int, Field(gt=0)]
    chosen_response_tokens: Annotated[int, Field(gt=0)]
    rejected_response_tokens: Annotated[int, Field(gt=0)]
    truncated_pairs: Annotated[int, Field(ge=0)]
    reference_cache_filename: str = Field(min_length=1)
    reference_cache_sha256: Sha256
    reference_cache_identity: ReferenceLogProbCacheIdentity
    final_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    final_checkpoint_manifest_sha256: Sha256
    metrics_jsonl: str = Field(min_length=1)
    metrics_parquet: str = Field(min_length=1)
    dpo_metrics_jsonl: str = Field(min_length=1)
    dpo_config: DPOConfig
    training_config: DPOTrainingConfig
    optimization_config: OptimizationConfig

    def canonical_bytes(self) -> bytes:
        """Return stable bytes for the final run manifest."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


class DPOTrainingReport(BaseModel):
    """Generated summary of measured full-run DPO metrics."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-dpo-training-report"] = (
        "lm-from-zero-dpo-training-report"
    )
    format_version: Literal[1] = 1
    completed_at_utc: datetime
    run_manifest_sha256: Sha256
    model_variant: DenseModelVariant
    optimizer_steps: Annotated[int, Field(gt=0)]
    pairs_consumed: Annotated[int, Field(gt=0)]
    chosen_response_tokens: Annotated[int, Field(gt=0)]
    rejected_response_tokens: Annotated[int, Field(gt=0)]
    truncated_pairs: Annotated[int, Field(ge=0)]
    truncation_rate: Annotated[float, Field(ge=0, le=1)]
    first_loss: Annotated[float, Field(ge=0)]
    final_loss: Annotated[float, Field(ge=0)]
    mean_last_100_loss: Annotated[float, Field(ge=0)]
    first_reward_margin: float
    final_reward_margin: float
    final_chosen_reward: float
    final_rejected_reward: float
    final_preference_accuracy: Annotated[float, Field(ge=0, le=1)]
    mean_chosen_response_tokens: Annotated[float, Field(gt=0)]
    mean_rejected_response_tokens: Annotated[float, Field(gt=0)]
    final_learning_rate: Annotated[float, Field(gt=0)]
    reference_cache_key: Sha256
    reference_cache_sha256: Sha256
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


class _PreferenceBatch:
    def __init__(
        self,
        *,
        indices: tuple[int, ...],
        chosen_input_ids: Tensor,
        chosen_response_mask: Tensor,
        chosen_attention_mask: Tensor,
        rejected_input_ids: Tensor,
        rejected_response_mask: Tensor,
        rejected_attention_mask: Tensor,
        chosen_lengths: tuple[int, ...],
        rejected_lengths: tuple[int, ...],
        chosen_response_tokens: tuple[int, ...],
        rejected_response_tokens: tuple[int, ...],
        chosen_truncated: tuple[bool, ...],
        rejected_truncated: tuple[bool, ...],
    ) -> None:
        self.indices = indices
        self.chosen_input_ids = chosen_input_ids
        self.chosen_response_mask = chosen_response_mask
        self.chosen_attention_mask = chosen_attention_mask
        self.rejected_input_ids = rejected_input_ids
        self.rejected_response_mask = rejected_response_mask
        self.rejected_attention_mask = rejected_attention_mask
        self.chosen_lengths = chosen_lengths
        self.rejected_lengths = rejected_lengths
        self.chosen_response_tokens = chosen_response_tokens
        self.rejected_response_tokens = rejected_response_tokens
        self.chosen_truncated = chosen_truncated
        self.rejected_truncated = rejected_truncated


class _RenderedPreference:
    def __init__(self, index: int, rendered: PreferencePairExample) -> None:
        self.index = index
        self.rendered = rendered


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(model: BaseModel) -> str:
    encoded = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256(encoded).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise DPOTrainingError(f"incomplete artifact exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _artifact_record(path: Path) -> DPOArtifactRecord:
    return DPOArtifactRecord(
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _pad_sequences(
    sequences: tuple[PreferenceSequence, ...],
    *,
    target_length: int,
) -> tuple[Tensor, Tensor, Tensor, tuple[int, ...]]:
    if not sequences:
        raise DPOTrainingError("DPO batch cannot be empty")
    lengths = tuple(len(sequence.input_ids) for sequence in sequences)
    if target_length < max(lengths):
        raise DPOTrainingError("DPO padding length is shorter than a sequence")
    input_ids = torch.full(
        (len(sequences), target_length),
        SPECIAL_TOKEN_IDS["<|pad|>"],
        dtype=torch.long,
    )
    response_mask = torch.zeros((len(sequences), target_length), dtype=torch.bool)
    attention_mask = torch.zeros((len(sequences), target_length), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        length = len(sequence.input_ids)
        input_ids[row, :length] = torch.tensor(sequence.input_ids, dtype=torch.long)
        response_mask[row, :length] = torch.tensor(
            sequence.response_mask, dtype=torch.bool
        )
        attention_mask[row, :length] = True
    return input_ids, response_mask, attention_mask, lengths


def _make_batch(
    items: list[_RenderedPreference],
) -> _PreferenceBatch:
    chosen = tuple(item.rendered for item in items)
    rejected = tuple(item.rendered for item in items)
    target_length = max(
        max(len(sequence.chosen.input_ids) for sequence in chosen),
        max(len(sequence.rejected.input_ids) for sequence in rejected),
    )
    chosen_batch = _pad_sequences(
        tuple(sequence.chosen for sequence in chosen), target_length=target_length
    )
    rejected_batch = _pad_sequences(
        tuple(sequence.rejected for sequence in rejected), target_length=target_length
    )
    return _PreferenceBatch(
        indices=tuple(item.index for item in items),
        chosen_input_ids=chosen_batch[0],
        chosen_response_mask=chosen_batch[1],
        chosen_attention_mask=chosen_batch[2],
        rejected_input_ids=rejected_batch[0],
        rejected_response_mask=rejected_batch[1],
        rejected_attention_mask=rejected_batch[2],
        chosen_lengths=chosen_batch[3],
        rejected_lengths=rejected_batch[3],
        chosen_response_tokens=tuple(
            sequence.chosen.response_token_count for sequence in chosen
        ),
        rejected_response_tokens=tuple(
            sequence.rejected.response_token_count for sequence in rejected
        ),
        chosen_truncated=tuple(sequence.chosen.truncated for sequence in chosen),
        rejected_truncated=tuple(sequence.rejected.truncated for sequence in rejected),
    )


def _iter_preference_batches(
    records_path: Path,
    tokenizer: ByteBPE,
    *,
    expected_records: int,
    batch_size: int,
    bucket_size: int,
    max_length: int,
) -> Iterator[_PreferenceBatch]:
    """Render records in deterministic length buckets for low padding waste."""

    if batch_size <= 0 or bucket_size <= 0:
        raise DPOTrainingError("DPO batch and bucket sizes must be positive")
    bucket: list[_RenderedPreference] = []
    record_index = 0

    def emit_bucket() -> Iterator[_PreferenceBatch]:
        bucket.sort(
            key=lambda item: (
                max(
                    len(item.rendered.chosen.input_ids),
                    len(item.rendered.rejected.input_ids),
                ),
                item.index,
            )
        )
        for start in range(0, len(bucket), batch_size):
            yield _make_batch(bucket[start : start + batch_size])

    with records_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = PreferenceRecord.model_validate_json(line)
            rendered = render_preference_pair(
                record.pair,
                tokenizer,
                max_length=max_length,
                truncation="left",
            )
            bucket.append(
                _RenderedPreference(
                    record_index,
                    rendered,
                )
            )
            record_index += 1
            if len(bucket) >= bucket_size:
                yield from emit_bucket()
                bucket.clear()
    if bucket:
        yield from emit_bucket()
    if record_index != expected_records:
        raise DPOTrainingError(
            f"preference record count {record_index} disagrees with manifest "
            f"count {expected_records}"
        )


def _logprob_pair(
    model: Olmo2ForCausalLM,
    batch: _PreferenceBatch,
    *,
    device: torch.device,
    reference: bool,
) -> tuple[Tensor, Tensor]:
    input_ids = torch.cat([batch.chosen_input_ids, batch.rejected_input_ids], dim=0).to(
        device
    )
    response_mask = torch.cat(
        [batch.chosen_response_mask, batch.rejected_response_mask], dim=0
    ).to(device)
    attention_mask = torch.cat(
        [batch.chosen_attention_mask, batch.rejected_attention_mask], dim=0
    ).to(device)
    context = torch.no_grad() if reference else torch.enable_grad()
    with context, torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = model(input_ids, attention_mask=attention_mask)
        logps = masked_sequence_logprob(output.logits, input_ids, response_mask)
    batch_size = batch.chosen_input_ids.shape[0]
    return logps[:batch_size], logps[batch_size:]


def _write_reference_cache(
    path: Path,
    *,
    chosen_logps: Tensor,
    rejected_logps: Tensor,
    identity: ReferenceLogProbCacheIdentity,
) -> str:
    if chosen_logps.ndim != 1 or rejected_logps.ndim != 1:
        raise DPOTrainingError("reference cache tensors must be one-dimensional")
    if chosen_logps.shape != rejected_logps.shape or chosen_logps.numel() == 0:
        raise DPOTrainingError("reference cache tensors must have equal nonzero shapes")
    if not bool(torch.isfinite(chosen_logps).all()) or not bool(
        torch.isfinite(rejected_logps).all()
    ):
        raise DPOTrainingError("reference cache contains non-finite log-probabilities")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise DPOTrainingError(f"incomplete reference cache exists: {temporary}")
    try:
        save_file(
            {
                "chosen_logps": chosen_logps.detach().float().cpu().contiguous(),
                "rejected_logps": rejected_logps.detach().float().cpu().contiguous(),
            },
            str(temporary),
            metadata={
                "format": "lm-from-zero-reference-logprob-cache",
                "format_version": "1",
                "cache_key": identity.cache_key,
                "identity": identity.canonical_json(),
                "pair_count": str(chosen_logps.numel()),
            },
        )
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return _sha256_file(path)


def _gradient_norm(model: Olmo2ForCausalLM) -> float:
    squared_norms = [
        parameter.grad.detach().float().norm(2).square()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not squared_norms:
        raise DPOTrainingError("the DPO update produced no gradients")
    return float(torch.stack(squared_norms).sum().sqrt().item())


def _objective_metrics(output: DPOObjectiveOutput) -> tuple[float, float, float, float]:
    accuracy = float((output.logits > 0).float().mean().item())
    return (
        float(output.chosen_rewards.mean().detach().cpu().item()),
        float(output.rejected_rewards.mean().detach().cpu().item()),
        float(output.reward_margins.mean().detach().cpu().item()),
        accuracy,
    )


def _append_dpo_metric(
    handle: TextIO, payload: dict[str, object], *, durable: bool
) -> None:
    """Append one canonical DPO-specific metric without widening shared metrics."""

    record = dict(payload)
    record["recorded_at_utc"] = datetime.now(UTC).isoformat()
    encoded = json.dumps(
        record,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    handle.write(encoded + "\n")
    if durable:
        handle.flush()
        os.fsync(handle.fileno())


def _write_checkpoint(
    checkpoint_root: Path,
    *,
    step: int,
    complete: bool,
    model: Olmo2ForCausalLM,
    optimizer: torch.optim.Optimizer,
    model_variant: DenseModelVariant,
    source_sft_checkpoint_id: str,
    source_sft_checkpoint_manifest_sha256: str,
    dataset_manifest_sha256: str,
    dataset_records_sha256: str,
    dataset_revision: str,
    tokenizer_sha256: str,
    model_config_sha256: str,
    dpo_config_sha256: str,
    training_config_sha256: str,
    reference_cache_key: str,
    pairs_consumed: int,
    chosen_response_tokens: int,
    rejected_response_tokens: int,
    truncated_pairs: int,
) -> tuple[Path, DPOCheckpointManifest, str]:
    checkpoint_id = f"step-{step:012d}"
    destination = checkpoint_root / checkpoint_id
    if destination.exists():
        raise DPOTrainingError(f"DPO checkpoint already exists: {destination}")
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
                "format": "lm-from-zero-dpo-checkpoint",
                "format_version": "1",
            },
        )
        recovery_path = temporary / "recovery.pt"
        torch.save(
            {
                "format": "lm-from-zero-dpo-checkpoint",
                "format_version": 1,
                "optimizer_state": optimizer.state_dict(),
                "rng_state": capture_rng_state(),
            },
            recovery_path,
        )
        manifest = DPOCheckpointManifest(
            created_at_utc=datetime.now(UTC),
            checkpoint_id=checkpoint_id,
            complete=complete,
            model_variant=model_variant,
            source_sft_checkpoint_id=source_sft_checkpoint_id,
            source_sft_checkpoint_manifest_sha256=source_sft_checkpoint_manifest_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            dataset_records_sha256=dataset_records_sha256,
            dataset_revision=dataset_revision,
            tokenizer_sha256=tokenizer_sha256,
            model_config_sha256=model_config_sha256,
            dpo_config_sha256=dpo_config_sha256,
            training_config_sha256=training_config_sha256,
            reference_cache_key=reference_cache_key,
            optimizer_step=step,
            pairs_consumed=pairs_consumed,
            chosen_response_tokens=chosen_response_tokens,
            rejected_response_tokens=rejected_response_tokens,
            truncated_pairs=truncated_pairs,
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


def _load_sft_run(
    checkpoint_path: Path,
    tokenizer: ByteBPE,
    *,
    model: Olmo2ForCausalLM,
) -> tuple[SFTCheckpointManifest, SFTRunManifest, str, object]:
    """Validate the final SFT checkpoint and its pretraining lineage."""

    checkpoint_manifest_path = checkpoint_path / "manifest.json"
    checkpoint_hash = _sha256_file(checkpoint_manifest_path)
    try:
        sft_checkpoint = SFTCheckpointManifest.model_validate_json(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
        run_manifest_path = checkpoint_path.parent.parent / "manifest.json"
        run_manifest = SFTRunManifest.model_validate_json(
            run_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise DPOTrainingError("SFT checkpoint or run manifest is invalid") from error
    if (
        not sft_checkpoint.complete
        or sft_checkpoint.checkpoint_id != run_manifest.final_checkpoint_id
    ):
        raise DPOTrainingError("DPO requires the complete final SFT checkpoint")
    if checkpoint_hash != run_manifest.final_checkpoint_manifest_sha256:
        raise DPOTrainingError("SFT checkpoint hash does not match its run manifest")
    for artifact in (sft_checkpoint.model_artifact, sft_checkpoint.recovery_artifact):
        artifact_path = checkpoint_path / artifact.filename
        if not artifact_path.is_file():
            raise DPOTrainingError(
                f"SFT checkpoint artifact is missing: {artifact.filename}"
            )
        if artifact_path.stat().st_size != artifact.size_bytes:
            raise DPOTrainingError(
                f"SFT checkpoint artifact size mismatch: {artifact.filename}"
            )
        if _sha256_file(artifact_path) != artifact.sha256:
            raise DPOTrainingError(
                f"SFT checkpoint artifact hash mismatch: {artifact.filename}"
            )
    if tokenizer.model_hash != sft_checkpoint.tokenizer_sha256:
        raise DPOTrainingError("tokenizer does not match the SFT checkpoint")
    source_checkpoint = Path(run_manifest.source_checkpoint_directory)
    if not source_checkpoint.is_absolute():
        source_checkpoint = Path.cwd() / source_checkpoint
    source_manifest = validate_checkpoint(source_checkpoint)
    if _sha256_file(source_checkpoint / "manifest.json") != (
        run_manifest.source_checkpoint_manifest_sha256
    ):
        raise DPOTrainingError("SFT source checkpoint does not match its run manifest")
    if source_manifest.binding.architecture != "olmo2":
        raise DPOTrainingError("DPO requires an OLMo2 SFT source checkpoint")
    if source_manifest.binding.resolved_model_config != model.config.model_dump(
        mode="json"
    ):
        raise DPOTrainingError(
            "SFT source model configuration does not match the model"
        )
    try:
        from safetensors.torch import load_file

        model.load_state_dict(
            load_file(
                str(checkpoint_path / sft_checkpoint.model_artifact.filename),
                device="cpu",
            ),
            strict=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise DPOTrainingError(
            "SFT checkpoint model weights could not be loaded"
        ) from error
    return sft_checkpoint, run_manifest, checkpoint_hash, source_manifest


def _prepare_reference_cache(
    *,
    records_path: Path,
    manifest: PreferenceMixManifest,
    tokenizer: ByteBPE,
    model: Olmo2ForCausalLM,
    device: torch.device,
    config: DPOTrainingConfig,
    identity: ReferenceLogProbCacheIdentity,
    cache_path: Path,
) -> tuple[str, int, int, int, int]:
    chosen_cache = torch.empty(manifest.selected_pairs, dtype=torch.float32)
    rejected_cache = torch.empty(manifest.selected_pairs, dtype=torch.float32)
    chosen_response_tokens = 0
    rejected_response_tokens = 0
    truncated_pairs = 0
    batches = _iter_preference_batches(
        records_path,
        tokenizer,
        expected_records=manifest.selected_pairs,
        batch_size=config.batch_size,
        bucket_size=config.bucket_size,
        max_length=config.max_length,
    )
    model.eval()
    progress = tqdm(
        total=manifest.selected_pairs, desc="DPO reference cache", unit="pairs"
    )
    try:
        for batch in batches:
            reference_chosen, reference_rejected = _logprob_pair(
                model,
                batch,
                device=device,
                reference=True,
            )
            for row, index in enumerate(batch.indices):
                chosen_cache[index] = reference_chosen[row].detach().float().cpu()
                rejected_cache[index] = reference_rejected[row].detach().float().cpu()
            chosen_response_tokens += sum(batch.chosen_response_tokens)
            rejected_response_tokens += sum(batch.rejected_response_tokens)
            truncated_pairs += sum(
                chosen or rejected
                for chosen, rejected in zip(
                    batch.chosen_truncated,
                    batch.rejected_truncated,
                    strict=True,
                )
            )
            progress.update(len(batch.indices))
    finally:
        progress.close()
    cache_hash = _write_reference_cache(
        cache_path,
        chosen_logps=chosen_cache,
        rejected_logps=rejected_cache,
        identity=identity,
    )
    return (
        cache_hash,
        chosen_response_tokens,
        rejected_response_tokens,
        truncated_pairs,
        manifest.selected_pairs,
    )


def run_dpo_training(
    *,
    dataset_manifest_path: str | Path,
    tokenizer_path: str | Path,
    source_checkpoint: str | Path,
    output_directory: str | Path,
    report_path: str | Path,
    config: DPOTrainingConfig | None = None,
) -> DPOTrainingReport:
    """Run one complete deterministic DPO epoch from the final SFT checkpoint."""

    if not torch.cuda.is_available():
        raise DPOTrainingError("CUDA is required for the full DPO run")
    training_config = DPOTrainingConfig() if config is None else config
    manifest_path = Path(dataset_manifest_path)
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise DPOTrainingError(f"DPO output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    try:
        manifest = PreferenceMixManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise DPOTrainingError("preference manifest is invalid") from error
    records_path = manifest_path.parent / manifest.records_jsonl
    if _sha256_file(records_path) != manifest.records_sha256:
        raise DPOTrainingError("preference records do not match their manifest hash")
    tokenizer = ByteBPE.load(tokenizer_path)
    if tokenizer.model_hash != manifest.tokenizer_hash:
        raise DPOTrainingError("tokenizer does not match the preference manifest")
    if manifest.chat_template_hash != DEFAULT_CHAT_TEMPLATE.template_hash:
        raise DPOTrainingError(
            "preference chat template does not match the project template"
        )
    if training_config.max_length > manifest.max_length:
        raise DPOTrainingError("DPO context exceeds the prepared preference context")

    source_checkpoint_path = Path(source_checkpoint)
    source_manifest = validate_checkpoint(source_checkpoint_path)
    if source_manifest.binding.architecture != "olmo2":
        raise DPOTrainingError("full DPO requires an OLMo2 source checkpoint")
    model_config = Olmo2Config.model_validate(
        source_manifest.binding.resolved_model_config
    )
    if training_config.max_length > model_config.max_position_embeddings:
        raise DPOTrainingError("DPO context exceeds the source model context")
    model_variant: DenseModelVariant
    checkpoint_manifest_path = source_checkpoint_path / "manifest.json"
    checkpoint_hash = _sha256_file(checkpoint_manifest_path)
    try:
        sft_run_manifest = SFTRunManifest.model_validate_json(
            (source_checkpoint_path.parent.parent / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError) as error:
        raise DPOTrainingError("SFT run manifest is invalid") from error
    model_variant = sft_run_manifest.model_variant
    model = Olmo2ForCausalLM(model_config, variant=model_variant)
    sft_checkpoint, _validated_sft_run_manifest, checkpoint_hash, _ = _load_sft_run(
        source_checkpoint_path,
        tokenizer,
        model=model,
    )
    dpo_config = DPOConfig(
        beta=training_config.beta,
        learning_rate=training_config.learning_rate,
        max_length=training_config.max_length,
        pair_count=manifest.selected_pairs,
    )
    dpo_config_hash = _canonical_hash(dpo_config)
    training_config_hash = _canonical_hash(training_config)
    total_steps = math.ceil(manifest.selected_pairs / training_config.batch_size)
    optimization_config = OptimizationConfig(
        learning_rate=training_config.learning_rate,
        beta1=training_config.beta1,
        beta2=training_config.beta2,
        epsilon=training_config.epsilon,
        weight_decay=training_config.weight_decay,
        gradient_clip_norm=training_config.gradient_clip_norm,
        total_steps=total_steps,
        warmup_fraction=training_config.warmup_fraction,
        minimum_lr_ratio=training_config.minimum_lr_ratio,
    )
    identity = ReferenceLogProbCacheIdentity(
        model_hash=model_config.config_hash,
        tokenizer_hash=tokenizer.model_hash,
        checkpoint_hash=checkpoint_hash,
        template_hash=manifest.chat_template_hash,
        max_length=training_config.max_length,
    )
    cache_path = output / "reference-logprobs.safetensors"
    device = torch.device(training_config.device)
    torch.manual_seed(training_config.seed)
    torch.cuda.manual_seed_all(training_config.seed)
    model.to(device)
    reference_model = Olmo2ForCausalLM(model_config, variant=model_variant)
    _load_sft_run(source_checkpoint_path, tokenizer, model=reference_model)
    reference_model.to(device)
    try:
        (
            cache_hash,
            chosen_response_tokens,
            rejected_response_tokens,
            truncated_pairs,
            pairs_cached,
        ) = _prepare_reference_cache(
            records_path=records_path,
            manifest=manifest,
            tokenizer=tokenizer,
            model=reference_model,
            device=device,
            config=training_config,
            identity=identity,
            cache_path=cache_path,
        )
    finally:
        del reference_model
        torch.cuda.empty_cache()
    if pairs_cached != manifest.selected_pairs:
        raise DPOTrainingError(
            "reference cache did not cover the complete preference mix"
        )
    from safetensors.torch import load_file

    cache_tensors = load_file(str(cache_path), device="cpu")
    try:
        chosen_cache = cache_tensors["chosen_logps"]
        rejected_cache = cache_tensors["rejected_logps"]
    except KeyError as error:
        raise DPOTrainingError(
            "reference cache is missing log-probability tensors"
        ) from error

    optimizer, _ = build_adamw(model, optimization_config, device_type="cuda")
    model.train()
    metrics_jsonl = output / "metrics.jsonl"
    metrics_parquet = output / "metrics.parquet"
    dpo_metrics_jsonl = output / "dpo_metrics.jsonl"
    checkpoints_root = output / "checkpoints"
    sinks = TrainingMetricSinks(
        jsonl_path=metrics_jsonl,
        tensorboard_directory=None,
        parquet_path=metrics_parquet,
        resume_optimizer_step=0,
        durable_every_steps=50,
        durable_every_seconds=5.0,
    )
    dpo_metrics_handle = dpo_metrics_jsonl.open("a", encoding="utf-8", newline="\n")
    started = time.perf_counter()
    optimizer_step = 0
    pairs_consumed = 0
    chosen_tokens_consumed = 0
    rejected_tokens_consumed = 0
    losses: list[float] = []
    reward_margins: list[float] = []
    last_metrics: dict[str, float] = {}
    final_checkpoint: tuple[Path, DPOCheckpointManifest, str] | None = None
    dataset_manifest_hash = _sha256_file(manifest_path)
    try:
        sinks.append_event(
            {
                "event": "run_start",
                "model_variant": model_variant,
                "source_sft_checkpoint_id": sft_checkpoint.checkpoint_id,
                "dataset_manifest_sha256": dataset_manifest_hash,
                "dataset_records_sha256": manifest.records_sha256,
                "training_config": training_config.model_dump(mode="json"),
                "training_config_sha256": training_config_hash,
                "dpo_config": dpo_config.model_dump(mode="json"),
                "dpo_config_sha256": dpo_config_hash,
                "reference_cache_key": identity.cache_key,
                "total_optimizer_steps": total_steps,
            }
        )
        progress = tqdm(total=manifest.selected_pairs, desc="DPO train", unit="pairs")
        try:
            batches = _iter_preference_batches(
                records_path,
                tokenizer,
                expected_records=manifest.selected_pairs,
                batch_size=training_config.batch_size,
                bucket_size=training_config.bucket_size,
                max_length=training_config.max_length,
            )
            for batch in batches:
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                learning_rate = set_learning_rate(
                    optimizer,
                    optimization_config,
                    optimizer_step - 1,
                )
                step_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(device)
                policy_chosen, policy_rejected = _logprob_pair(
                    model,
                    batch,
                    device=device,
                    reference=False,
                )
                reference_chosen = chosen_cache[list(batch.indices)].to(device)
                reference_rejected = rejected_cache[list(batch.indices)].to(device)
                objective = dpo_objective(
                    policy_chosen,
                    policy_rejected,
                    reference_chosen,
                    reference_rejected,
                    beta=dpo_config.beta,
                )
                loss = objective.loss
                if not bool(torch.isfinite(loss)):
                    raise DPOTrainingError(
                        f"non-finite DPO loss at step {optimizer_step}"
                    )
                loss.backward()  # type: ignore[no-untyped-call]
                gradient_norm = float(
                    clip_gradients(model, optimization_config.gradient_clip_norm)
                    .detach()
                    .cpu()
                    .item()
                )
                if not math.isfinite(gradient_norm):
                    raise DPOTrainingError(
                        f"non-finite DPO gradient at step {optimizer_step}"
                    )
                optimizer.step()
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - step_started
                loss_value = float(loss.detach().cpu().item())
                chosen_reward, rejected_reward, margin, accuracy = _objective_metrics(
                    objective
                )
                losses.append(loss_value)
                reward_margins.append(margin)
                batch_pairs = len(batch.indices)
                pairs_consumed += batch_pairs
                batch_chosen_tokens = sum(batch.chosen_response_tokens)
                batch_rejected_tokens = sum(batch.rejected_response_tokens)
                chosen_tokens_consumed += batch_chosen_tokens
                rejected_tokens_consumed += batch_rejected_tokens
                batch_truncated = sum(
                    chosen or rejected
                    for chosen, rejected in zip(
                        batch.chosen_truncated,
                        batch.rejected_truncated,
                        strict=True,
                    )
                )
                progress.update(batch_pairs)
                progress.set_postfix(loss=f"{loss_value:.4f}", margin=f"{margin:.3f}")
                sinks.log_optimizer_step(
                    {
                        "optimizer_step": optimizer_step,
                        "loss": loss_value,
                        "learning_rate": learning_rate,
                        "gradient_norm": gradient_norm,
                        "tokens_consumed": chosen_tokens_consumed
                        + rejected_tokens_consumed,
                        "elapsed_seconds": elapsed,
                        "tokens_per_second": (
                            batch_chosen_tokens + batch_rejected_tokens
                        )
                        / max(elapsed, 1e-9),
                        "peak_cuda_memory_allocated_bytes": (
                            torch.cuda.max_memory_allocated(device)
                        ),
                        "peak_cuda_memory_reserved_bytes": (
                            torch.cuda.max_memory_reserved(device)
                        ),
                    }
                )
                last_metrics = {
                    "chosen_reward_mean": chosen_reward,
                    "rejected_reward_mean": rejected_reward,
                    "reward_margin_mean": margin,
                    "preference_accuracy": accuracy,
                    "gradient_norm": gradient_norm,
                    "chosen_response_tokens": batch_chosen_tokens / batch_pairs,
                    "rejected_response_tokens": batch_rejected_tokens / batch_pairs,
                    "truncated_pairs": float(batch_truncated),
                }
                _append_dpo_metric(
                    dpo_metrics_handle,
                    {
                        "event": "dpo_optimizer_step",
                        "optimizer_step": optimizer_step,
                        "loss": loss_value,
                        "learning_rate": learning_rate,
                        **last_metrics,
                        "elapsed_seconds": elapsed,
                    },
                    durable=optimizer_step % 50 == 0,
                )
                if (
                    optimizer_step % training_config.checkpoint_every_steps == 0
                    or optimizer_step == total_steps
                ):
                    sinks.durable_sync()
                    dpo_metrics_handle.flush()
                    os.fsync(dpo_metrics_handle.fileno())
                    final_checkpoint = _write_checkpoint(
                        checkpoints_root,
                        step=optimizer_step,
                        complete=optimizer_step == total_steps,
                        model=model,
                        optimizer=optimizer,
                        model_variant=model_variant,
                        source_sft_checkpoint_id=sft_checkpoint.checkpoint_id,
                        source_sft_checkpoint_manifest_sha256=checkpoint_hash,
                        dataset_manifest_sha256=dataset_manifest_hash,
                        dataset_records_sha256=manifest.records_sha256,
                        dataset_revision=manifest.dataset_revision,
                        tokenizer_sha256=tokenizer.model_hash,
                        model_config_sha256=model_config.config_hash,
                        dpo_config_sha256=dpo_config_hash,
                        training_config_sha256=training_config_hash,
                        reference_cache_key=identity.cache_key,
                        pairs_consumed=pairs_consumed,
                        chosen_response_tokens=chosen_response_tokens,
                        rejected_response_tokens=rejected_response_tokens,
                        truncated_pairs=truncated_pairs,
                    )
                del policy_chosen, policy_rejected, objective, loss
        finally:
            progress.close()
        if pairs_consumed != manifest.selected_pairs:
            raise DPOTrainingError("DPO epoch did not consume the complete manifest")
        sinks.append_event(
            {
                "event": "run_complete",
                "optimizer_step": optimizer_step,
                "pairs_consumed": pairs_consumed,
                "chosen_response_tokens": chosen_response_tokens,
                "rejected_response_tokens": rejected_response_tokens,
                "truncated_pairs": truncated_pairs,
            }
        )
        sinks.close()
        dpo_metrics_handle.flush()
        os.fsync(dpo_metrics_handle.fileno())
        dpo_metrics_handle.close()
    except Exception:
        sinks.abort()
        dpo_metrics_handle.close()
        raise
    finally:
        del optimizer, model
        torch.cuda.empty_cache()
    if final_checkpoint is None:
        raise DPOTrainingError("DPO run finished without a final checkpoint")
    _final_checkpoint_path, final_checkpoint_manifest, final_checkpoint_hash = (
        final_checkpoint
    )
    run_manifest = DPORunManifest(
        completed_at_utc=datetime.now(UTC),
        model_variant=model_variant,
        source_sft_checkpoint_directory=str(source_checkpoint_path),
        source_sft_checkpoint_id=sft_checkpoint.checkpoint_id,
        source_sft_checkpoint_manifest_sha256=checkpoint_hash,
        dataset_manifest_sha256=dataset_manifest_hash,
        dataset_records_sha256=manifest.records_sha256,
        dataset_revision=manifest.dataset_revision,
        tokenizer_sha256=tokenizer.model_hash,
        model_config_sha256=model_config.config_hash,
        dpo_config_sha256=dpo_config_hash,
        training_config_sha256=training_config_hash,
        total_optimizer_steps=optimizer_step,
        pairs_consumed=pairs_consumed,
        chosen_response_tokens=chosen_response_tokens,
        rejected_response_tokens=rejected_response_tokens,
        truncated_pairs=truncated_pairs,
        reference_cache_filename=cache_path.name,
        reference_cache_sha256=cache_hash,
        reference_cache_identity=identity,
        final_checkpoint_id=final_checkpoint_manifest.checkpoint_id,
        final_checkpoint_manifest_sha256=final_checkpoint_hash,
        metrics_jsonl=metrics_jsonl.name,
        metrics_parquet=metrics_parquet.name,
        dpo_metrics_jsonl=dpo_metrics_jsonl.name,
        dpo_config=dpo_config,
        training_config=training_config,
        optimization_config=optimization_config,
    )
    run_manifest_path = output / "manifest.json"
    _atomic_write(run_manifest_path, run_manifest.canonical_bytes() + b"\n")
    if not losses or not last_metrics:
        raise DPOTrainingError("DPO run produced no metrics")
    report = DPOTrainingReport(
        completed_at_utc=run_manifest.completed_at_utc,
        run_manifest_sha256=_sha256_file(run_manifest_path),
        model_variant=model_variant,
        optimizer_steps=optimizer_step,
        pairs_consumed=pairs_consumed,
        chosen_response_tokens=chosen_response_tokens,
        rejected_response_tokens=rejected_response_tokens,
        truncated_pairs=truncated_pairs,
        truncation_rate=truncated_pairs / pairs_consumed,
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_last_100_loss=sum(losses[-100:]) / min(100, len(losses)),
        first_reward_margin=reward_margins[0],
        final_reward_margin=reward_margins[-1],
        final_chosen_reward=last_metrics["chosen_reward_mean"],
        final_rejected_reward=last_metrics["rejected_reward_mean"],
        final_preference_accuracy=last_metrics["preference_accuracy"],
        mean_chosen_response_tokens=chosen_response_tokens / pairs_consumed,
        mean_rejected_response_tokens=rejected_response_tokens / pairs_consumed,
        final_learning_rate=optimization_config.learning_rate_at(optimizer_step - 1),
        reference_cache_key=identity.cache_key,
        reference_cache_sha256=cache_hash,
        elapsed_seconds=time.perf_counter() - started,
        final_checkpoint_id=final_checkpoint_manifest.checkpoint_id,
        final_checkpoint_manifest_sha256=final_checkpoint_hash,
    )
    _atomic_write(Path(report_path), report.canonical_bytes() + b"\n")
    return report
