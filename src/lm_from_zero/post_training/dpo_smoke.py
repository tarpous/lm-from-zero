"""Run a bounded CUDA DPO smoke against the completed SFT checkpoint."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file
from torch import Tensor

from lm_from_zero.models import Olmo2Config, Olmo2ForCausalLM
from lm_from_zero.post_training.dpo import (
    DPOConfig,
    DPOObjectiveOutput,
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
from lm_from_zero.post_training.sft_train import SFTCheckpointManifest, SFTRunManifest
from lm_from_zero.tokenizer.bpe import SPECIAL_TOKEN_IDS, ByteBPE
from lm_from_zero.training import validate_checkpoint

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40,64}$")]


class DPOSmokeError(RuntimeError):
    """Raised when the bounded DPO smoke cannot prove a valid update."""


class DPOSmokeStep(BaseModel):
    """Measured metrics for one bounded DPO optimizer update."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    optimizer_step: Annotated[int, Field(gt=0)]
    loss: Annotated[float, Field(ge=0)]
    chosen_reward_mean: float
    rejected_reward_mean: float
    reward_margin_mean: float
    preference_accuracy: Annotated[float, Field(ge=0, le=1)]
    gradient_norm: Annotated[float, Field(ge=0)]
    elapsed_seconds: Annotated[float, Field(gt=0)]
    peak_cuda_memory_allocated_bytes: Annotated[int, Field(gt=0)]
    peak_cuda_memory_reserved_bytes: Annotated[int, Field(gt=0)]
    finite: Literal[True] = True


class DPOSmokeReport(BaseModel):
    """Portable generated evidence for the bounded CUDA DPO smoke."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-dpo-smoke-report"] = "lm-from-zero-dpo-smoke-report"
    format_version: Literal[1] = 1
    recorded_at_utc: datetime
    dataset_manifest_sha256: Sha256
    dataset_records_sha256: Sha256
    dataset_revision: Revision
    selection_seed: int
    target_pairs: Annotated[int, Field(gt=0)]
    chat_template_sha256: Sha256
    dpo_config: DPOConfig
    checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    checkpoint_manifest_sha256: Sha256
    model_config_sha256: Sha256
    tokenizer_sha256: Sha256
    reference_cache_identity: ReferenceLogProbCacheIdentity
    device: Literal["cuda"] = "cuda"
    precision: Literal["bf16"] = "bf16"
    cuda_device_names: tuple[str, ...]
    cuda_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    batch_size: Annotated[int, Field(gt=0)]
    max_length: Annotated[int, Field(gt=1)]
    chosen_input_lengths: tuple[Annotated[int, Field(gt=1)], ...]
    rejected_input_lengths: tuple[Annotated[int, Field(gt=1)], ...]
    loss_before: Annotated[float, Field(ge=0)]
    loss_after: Annotated[float, Field(ge=0)]
    optimizer_steps: tuple[DPOSmokeStep, ...]

    def canonical_bytes(self) -> bytes:
        """Return stable JSON bytes for the generated report."""

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
        chosen_input_ids: Tensor,
        chosen_response_mask: Tensor,
        chosen_attention_mask: Tensor,
        rejected_input_ids: Tensor,
        rejected_response_mask: Tensor,
        rejected_attention_mask: Tensor,
        chosen_lengths: tuple[int, ...],
        rejected_lengths: tuple[int, ...],
    ) -> None:
        self.chosen_input_ids = chosen_input_ids
        self.chosen_response_mask = chosen_response_mask
        self.chosen_attention_mask = chosen_attention_mask
        self.rejected_input_ids = rejected_input_ids
        self.rejected_response_mask = rejected_response_mask
        self.rejected_attention_mask = rejected_attention_mask
        self.chosen_lengths = chosen_lengths
        self.rejected_lengths = rejected_lengths


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_sft_checkpoint(
    checkpoint_path: Path,
    *,
    model: Olmo2ForCausalLM,
) -> SFTCheckpointManifest:
    """Validate and load the separate SFT checkpoint contract."""

    manifest_path = checkpoint_path / "manifest.json"
    try:
        manifest = SFTCheckpointManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise DPOSmokeError("SFT checkpoint manifest is invalid") from error
    if not manifest.complete:
        raise DPOSmokeError("DPO smoke requires a complete SFT checkpoint")
    for artifact in (manifest.model_artifact, manifest.recovery_artifact):
        path = checkpoint_path / artifact.filename
        if not path.is_file():
            raise DPOSmokeError(
                f"SFT checkpoint artifact is missing: {artifact.filename}"
            )
        if path.stat().st_size != artifact.size_bytes:
            raise DPOSmokeError(
                f"SFT checkpoint artifact size mismatch: {artifact.filename}"
            )
        if _sha256_file(path) != artifact.sha256:
            raise DPOSmokeError(
                f"SFT checkpoint artifact hash mismatch: {artifact.filename}"
            )
    try:
        tensors = load_file(
            str(checkpoint_path / manifest.model_artifact.filename), device="cpu"
        )
        model.load_state_dict(tensors, strict=True)
    except (OSError, RuntimeError, ValueError) as error:
        raise DPOSmokeError(
            "SFT checkpoint model weights could not be loaded"
        ) from error
    return manifest


def _pad_sequences(
    sequences: tuple[PreferenceSequence, ...],
    *,
    target_length: int,
) -> tuple[Tensor, Tensor, Tensor, tuple[int, ...]]:
    if not sequences:
        raise DPOSmokeError("DPO smoke batch cannot be empty")
    lengths = tuple(len(sequence.input_ids) for sequence in sequences)
    if target_length < max(lengths):
        raise DPOSmokeError("DPO smoke padding length is shorter than a sequence")
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


def _load_batch(
    records_path: Path,
    tokenizer: ByteBPE,
    *,
    batch_size: int,
    max_length: int,
) -> _PreferenceBatch:
    chosen: list[PreferenceSequence] = []
    rejected: list[PreferenceSequence] = []
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
            chosen.append(rendered.chosen)
            rejected.append(rendered.rejected)
            if len(chosen) == batch_size:
                break
    if len(chosen) != batch_size:
        raise DPOSmokeError("the preference manifest does not contain enough records")
    chosen_sequences = tuple(chosen)
    rejected_sequences = tuple(rejected)
    target_length = max(
        max(len(sequence.input_ids) for sequence in chosen_sequences),
        max(len(sequence.input_ids) for sequence in rejected_sequences),
    )
    chosen_batch = _pad_sequences(chosen_sequences, target_length=target_length)
    rejected_batch = _pad_sequences(rejected_sequences, target_length=target_length)
    return _PreferenceBatch(
        chosen_input_ids=chosen_batch[0],
        chosen_response_mask=chosen_batch[1],
        chosen_attention_mask=chosen_batch[2],
        rejected_input_ids=rejected_batch[0],
        rejected_response_mask=rejected_batch[1],
        rejected_attention_mask=rejected_batch[2],
        chosen_lengths=chosen_batch[3],
        rejected_lengths=rejected_batch[3],
    )


def _gradient_norm(model: Olmo2ForCausalLM) -> float:
    squared_norms = [
        parameter.grad.detach().float().norm(2).square()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not squared_norms:
        raise DPOSmokeError("the DPO update produced no gradients")
    return float(torch.stack(squared_norms).sum().sqrt().item())


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


def _objective_metrics(output: DPOObjectiveOutput) -> tuple[float, float, float, float]:
    accuracy = float((output.logits > 0).float().mean().item())
    return (
        float(output.chosen_rewards.mean().detach().cpu().item()),
        float(output.rejected_rewards.mean().detach().cpu().item()),
        float(output.reward_margins.mean().detach().cpu().item()),
        accuracy,
    )


def run_dpo_smoke(
    *,
    dataset_manifest_path: str | Path,
    tokenizer_path: str | Path,
    source_checkpoint: str | Path,
    output_path: str | Path,
    batch_size: int = 2,
    max_length: int = 512,
    steps: int = 2,
) -> DPOSmokeReport:
    """Run a finite DPO update sequence with a frozen SFT reference model."""

    if not torch.cuda.is_available():
        raise DPOSmokeError("CUDA is required for the bounded DPO smoke")
    if batch_size <= 0 or max_length <= 1 or steps <= 0:
        raise DPOSmokeError("DPO smoke dimensions must be positive")

    manifest_path = Path(dataset_manifest_path)
    manifest = PreferenceMixManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    records_path = manifest_path.parent / manifest.records_jsonl
    if _sha256_file(records_path) != manifest.records_sha256:
        raise DPOSmokeError("preference records do not match their manifest hash")
    tokenizer = ByteBPE.load(tokenizer_path)
    if tokenizer.model_hash != manifest.tokenizer_hash:
        raise DPOSmokeError("tokenizer does not match the preference manifest")
    checkpoint_path = Path(source_checkpoint)
    checkpoint_manifest_hash = _sha256_file(checkpoint_path / "manifest.json")
    run_manifest_path = checkpoint_path.parent.parent / "manifest.json"
    try:
        run_manifest = SFTRunManifest.model_validate_json(
            run_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise DPOSmokeError("SFT run manifest is invalid") from error
    source_checkpoint = Path(run_manifest.source_checkpoint_directory)
    if not source_checkpoint.is_absolute():
        source_checkpoint = Path.cwd() / source_checkpoint
    source_manifest = validate_checkpoint(source_checkpoint)
    if _sha256_file(source_checkpoint / "manifest.json") != (
        run_manifest.source_checkpoint_manifest_sha256
    ):
        raise DPOSmokeError("SFT source checkpoint does not match the run manifest")
    if tokenizer.model_hash != run_manifest.tokenizer_sha256:
        raise DPOSmokeError("tokenizer does not match the SFT run manifest")
    model_config = Olmo2Config.model_validate(
        source_manifest.binding.resolved_model_config
    )
    if model_config.config_hash != run_manifest.model_config_sha256:
        raise DPOSmokeError("SFT model configuration does not match its run manifest")
    if max_length > model_config.max_position_embeddings:
        raise DPOSmokeError("DPO smoke length exceeds checkpoint context")

    batch = _load_batch(
        records_path,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )
    device = torch.device("cuda")
    torch.manual_seed(1_337)
    torch.cuda.manual_seed_all(1_337)
    policy = Olmo2ForCausalLM(model_config, variant=run_manifest.model_variant)
    reference_model = Olmo2ForCausalLM(model_config, variant=run_manifest.model_variant)
    sft_checkpoint_manifest = _load_sft_checkpoint(
        checkpoint_path,
        model=policy,
    )
    _load_sft_checkpoint(checkpoint_path, model=reference_model)
    if sft_checkpoint_manifest.model_config_sha256 != run_manifest.model_config_sha256:
        raise DPOSmokeError("SFT checkpoint model configuration does not match its run")
    if sft_checkpoint_manifest.tokenizer_sha256 != tokenizer.model_hash:
        raise DPOSmokeError("SFT checkpoint tokenizer does not match the tokenizer")
    policy.to(device)
    reference_model.to(device)
    policy.train()
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    dpo_config = DPOConfig(max_length=max_length)
    reference_chosen, reference_rejected = _logprob_pair(
        reference_model,
        batch,
        device=device,
        reference=True,
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=dpo_config.learning_rate)
    torch.cuda.reset_peak_memory_stats(device)
    steps_out: list[DPOSmokeStep] = []
    for optimizer_step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        policy_chosen, policy_rejected = _logprob_pair(
            policy,
            batch,
            device=device,
            reference=False,
        )
        output = dpo_objective(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=dpo_config.beta,
        )
        if not bool(torch.isfinite(output.loss)):
            raise DPOSmokeError("DPO produced a non-finite loss")
        output.loss.backward()  # type: ignore[no-untyped-call]
        gradient_norm = _gradient_norm(policy)
        if not torch.isfinite(torch.tensor(gradient_norm, device=device)):
            raise DPOSmokeError("DPO produced a non-finite gradient norm")
        optimizer.step()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        chosen_reward, rejected_reward, margin, accuracy = _objective_metrics(output)
        steps_out.append(
            DPOSmokeStep(
                optimizer_step=optimizer_step,
                loss=float(output.loss.detach().cpu().item()),
                chosen_reward_mean=chosen_reward,
                rejected_reward_mean=rejected_reward,
                reward_margin_mean=margin,
                preference_accuracy=accuracy,
                gradient_norm=gradient_norm,
                elapsed_seconds=elapsed,
                peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(
                    device
                ),
                peak_cuda_memory_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        )

    reference_identity = ReferenceLogProbCacheIdentity(
        model_hash=sft_checkpoint_manifest.model_config_sha256,
        tokenizer_hash=tokenizer.model_hash,
        checkpoint_hash=checkpoint_manifest_hash,
        template_hash=manifest.chat_template_hash,
        max_length=max_length,
    )
    report = DPOSmokeReport(
        recorded_at_utc=datetime.now(UTC),
        dataset_manifest_sha256=_sha256_file(manifest_path),
        dataset_records_sha256=manifest.records_sha256,
        dataset_revision=manifest.dataset_revision,
        selection_seed=manifest.selection_seed,
        target_pairs=manifest.target_pairs,
        chat_template_sha256=manifest.chat_template_hash,
        dpo_config=dpo_config,
        checkpoint_id=sft_checkpoint_manifest.checkpoint_id,
        checkpoint_manifest_sha256=checkpoint_manifest_hash,
        model_config_sha256=sft_checkpoint_manifest.model_config_sha256,
        tokenizer_sha256=tokenizer.model_hash,
        reference_cache_identity=reference_identity,
        cuda_device_names=tuple(
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ),
        cuda_version=torch.version.cuda or "unknown",
        torch_version=torch.__version__,
        batch_size=batch_size,
        max_length=max_length,
        chosen_input_lengths=batch.chosen_lengths,
        rejected_input_lengths=batch.rejected_lengths,
        loss_before=steps_out[0].loss,
        loss_after=steps_out[-1].loss,
        optimizer_steps=tuple(steps_out),
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(report.canonical_bytes() + b"\n")
    temporary.replace(destination)
    del optimizer, policy, reference_model
    torch.cuda.empty_cache()
    return report
