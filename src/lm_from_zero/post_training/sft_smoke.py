"""Run a bounded CUDA assistant-only SFT smoke against selected checkpoints."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from lm_from_zero.models import DenseModelVariant, Olmo2Config, Olmo2ForCausalLM
from lm_from_zero.post_training.chat import Conversation
from lm_from_zero.post_training.dataset import SFTMixManifest, SFTRecord
from lm_from_zero.post_training.sft import (
    SFTBatch,
    SFTConfig,
    assistant_only_causal_loss,
    collate_supervised_chat,
    render_supervised_chat,
)
from lm_from_zero.tokenizer.bpe import ByteBPE
from lm_from_zero.training import load_checkpoint_model, validate_checkpoint

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40,64}$")]
SFTSmokeVariant = Literal["hybrid_muon", "mha", "layer_norm", "tied_embeddings"]


class SFTSmokeError(RuntimeError):
    """Raised when the bounded SFT smoke cannot prove a valid update."""


class SFTSmokeStep(BaseModel):
    """Measured metrics for one bounded SFT optimizer update."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    optimizer_step: Annotated[int, Field(gt=0)]
    loss: Annotated[float, Field(ge=0)]
    gradient_norm: Annotated[float, Field(ge=0)]
    elapsed_seconds: Annotated[float, Field(gt=0)]
    peak_cuda_memory_allocated_bytes: Annotated[int, Field(gt=0)]
    peak_cuda_memory_reserved_bytes: Annotated[int, Field(gt=0)]
    finite: Literal[True] = True


class SFTSmokeVariantResult(BaseModel):
    """Results for one selected pretrained checkpoint and dense variant."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    variant: SFTSmokeVariant
    model_variant: DenseModelVariant
    checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    checkpoint_manifest_sha256: Sha256
    model_config_sha256: Sha256
    tokenizer_sha256: Sha256
    batch_size: Annotated[int, Field(gt=0)]
    max_length: Annotated[int, Field(gt=1)]
    input_lengths: tuple[Annotated[int, Field(gt=1)], ...]
    assistant_token_counts: tuple[Annotated[int, Field(gt=0)], ...]
    loss_before: Annotated[float, Field(ge=0)]
    loss_after: Annotated[float, Field(ge=0)]
    optimizer_steps: tuple[SFTSmokeStep, ...]


class SFTSmokeReport(BaseModel):
    """Portable generated evidence for the bounded CUDA SFT smoke."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-sft-smoke-report"] = "lm-from-zero-sft-smoke-report"
    format_version: Literal[1] = 1
    recorded_at_utc: datetime
    dataset_manifest_sha256: Sha256
    dataset_records_sha256: Sha256
    dataset_revision: Revision
    selection_seed: int
    target_examples: Annotated[int, Field(gt=0)]
    chat_template_sha256: Sha256
    sft_config: SFTConfig
    device: Literal["cuda"] = "cuda"
    precision: Literal["bf16"] = "bf16"
    cuda_device_names: tuple[str, ...]
    cuda_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    variants: tuple[SFTSmokeVariantResult, ...]

    def canonical_bytes(self) -> bytes:
        """Return stable JSON bytes for the generated report."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_batch(
    records_path: Path,
    tokenizer: ByteBPE,
    *,
    batch_size: int,
    max_length: int,
) -> tuple[SFTBatch, tuple[int, ...], tuple[int, ...], str]:
    examples = []
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
            examples.append(example)
            if len(examples) == batch_size:
                break
    if len(examples) != batch_size:
        raise SFTSmokeError("the SFT manifest does not contain enough records")
    batch = collate_supervised_chat(examples)
    return (
        batch,
        batch.lengths,
        batch.assistant_token_counts,
        examples[0].template_hash,
    )


def _gradient_norm(model: Olmo2ForCausalLM) -> float:
    parameters = model.parameters()
    squared_norms = [
        parameter.grad.detach().float().norm(2).square()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squared_norms:
        raise SFTSmokeError("the SFT update produced no gradients")
    return float(torch.stack(squared_norms).sum().sqrt().item())


def _run_variant(
    *,
    variant: SFTSmokeVariant,
    model_variant: DenseModelVariant,
    checkpoint: Path,
    batch: SFTBatch,
    input_lengths: tuple[int, ...],
    assistant_token_counts: tuple[int, ...],
    max_length: int,
    learning_rate: float,
    steps: int,
    device: torch.device,
) -> SFTSmokeVariantResult:
    checkpoint_manifest = validate_checkpoint(checkpoint)
    if checkpoint_manifest.binding.architecture != "olmo2":
        raise SFTSmokeError("SFT smoke requires an OLMo2 checkpoint")
    model_config = Olmo2Config.model_validate(
        checkpoint_manifest.binding.resolved_model_config
    )
    if max_length > model_config.max_position_embeddings:
        raise SFTSmokeError("SFT smoke length exceeds checkpoint context")
    model = Olmo2ForCausalLM(model_config, variant=model_variant)
    load_checkpoint_model(
        checkpoint,
        model=model,
        expected_binding=checkpoint_manifest.binding,
    )
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    cuda_device = torch.device(device)
    batch_input = batch.input_ids.to(cuda_device)
    batch_labels = batch.labels.to(cuda_device)
    batch_mask = batch.attention_mask.to(cuda_device)
    steps_out: list[SFTSmokeStep] = []
    torch.cuda.reset_peak_memory_stats(cuda_device)
    for optimizer_step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = model(batch_input, attention_mask=batch_mask)
            loss = assistant_only_causal_loss(output.logits, batch_labels)
        if not bool(torch.isfinite(loss)):
            raise SFTSmokeError(f"{variant} produced a non-finite loss")
        loss.backward()  # type: ignore[no-untyped-call]
        gradient_norm = _gradient_norm(model)
        if not torch.isfinite(torch.tensor(gradient_norm, device=cuda_device)):
            raise SFTSmokeError(f"{variant} produced a non-finite gradient norm")
        optimizer.step()
        torch.cuda.synchronize(cuda_device)
        elapsed = time.perf_counter() - start
        steps_out.append(
            SFTSmokeStep(
                optimizer_step=optimizer_step,
                loss=float(loss.detach().cpu().item()),
                gradient_norm=gradient_norm,
                elapsed_seconds=elapsed,
                peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(
                    cuda_device
                ),
                peak_cuda_memory_reserved_bytes=torch.cuda.max_memory_reserved(
                    cuda_device
                ),
            )
        )
    result = SFTSmokeVariantResult(
        variant=variant,
        model_variant=model_variant,
        checkpoint_id=checkpoint_manifest.lineage.checkpoint_id,
        checkpoint_manifest_sha256=_sha256_file(checkpoint / "manifest.json"),
        model_config_sha256=checkpoint_manifest.binding.model_config_sha256,
        tokenizer_sha256=checkpoint_manifest.binding.tokenizer_sha256,
        batch_size=len(input_lengths),
        max_length=max_length,
        input_lengths=input_lengths,
        assistant_token_counts=assistant_token_counts,
        loss_before=steps_out[0].loss,
        loss_after=steps_out[-1].loss,
        optimizer_steps=tuple(steps_out),
    )
    del optimizer, model
    torch.cuda.empty_cache()
    return result


def run_sft_smoke(
    *,
    dataset_manifest_path: str | Path,
    tokenizer_path: str | Path,
    checkpoints: dict[SFTSmokeVariant, Path],
    model_variants: dict[SFTSmokeVariant, DenseModelVariant],
    output_path: str | Path,
    batch_size: int = 2,
    max_length: int = 512,
    steps: int = 2,
) -> SFTSmokeReport:
    """Run a finite CUDA update for each selected M8 dense checkpoint."""

    import torch

    if not torch.cuda.is_available():
        raise SFTSmokeError("CUDA is required for the bounded SFT smoke")
    if batch_size <= 0 or max_length <= 1 or steps <= 0:
        raise SFTSmokeError("SFT smoke dimensions must be positive")
    manifest_path = Path(dataset_manifest_path)
    manifest = SFTMixManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    records_path = manifest_path.parent / manifest.records_jsonl
    if _sha256_file(records_path) != manifest.records_sha256:
        raise SFTSmokeError("SFT records do not match their manifest hash")
    tokenizer = ByteBPE.load(tokenizer_path)
    batch, input_lengths, assistant_token_counts, template_hash = _load_batch(
        records_path,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
    )
    if not template_hash:
        raise SFTSmokeError("chat template hash is empty")
    device = torch.device("cuda")
    sft_config = SFTConfig()
    results = tuple(
        _run_variant(
            variant=variant,
            model_variant=model_variants[variant],
            checkpoint=checkpoints[variant],
            batch=batch,
            input_lengths=input_lengths,
            assistant_token_counts=assistant_token_counts,
            max_length=max_length,
            learning_rate=sft_config.learning_rate,
            steps=steps,
            device=device,
        )
        for variant in ("hybrid_muon", "mha", "layer_norm", "tied_embeddings")
    )
    report = SFTSmokeReport(
        recorded_at_utc=datetime.now(UTC),
        dataset_manifest_sha256=_sha256_file(manifest_path),
        dataset_records_sha256=manifest.records_sha256,
        dataset_revision=manifest.dataset_revision,
        selection_seed=manifest.selection_seed,
        target_examples=manifest.target_examples,
        chat_template_sha256=template_hash,
        sft_config=sft_config,
        cuda_device_names=tuple(
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ),
        cuda_version=torch.version.cuda or "unknown",
        torch_version=torch.__version__,
        variants=results,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(report.canonical_bytes() + b"\n")
    temporary.replace(destination)
    return report
