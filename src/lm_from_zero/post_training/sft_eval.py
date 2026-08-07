"""Evaluate a completed SFT checkpoint with deterministic native chat generation."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file

from lm_from_zero.generation import CausalGenerationConfig, generate_causal
from lm_from_zero.models import Olmo2Config, Olmo2ForCausalLM
from lm_from_zero.post_training.chat import ChatMessage, Conversation
from lm_from_zero.post_training.sft_train import (
    SFTCheckpointManifest,
    SFTRunManifest,
)
from lm_from_zero.tokenizer.bpe import SPECIAL_TOKEN_IDS, ByteBPE
from lm_from_zero.training import validate_checkpoint

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class SFTEvaluationError(RuntimeError):
    """Raised when a completed SFT checkpoint cannot be evaluated safely."""


class SFTGenerationCase(BaseModel):
    """One deterministic chat prompt and its generated continuation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str = Field(min_length=1)
    prompt_token_count: Annotated[int, Field(gt=0)]
    generated_text: str
    generated_token_count: Annotated[int, Field(ge=0)]
    stop_reason: Literal["eos", "max_new_tokens"]


class SFTGenerationReport(BaseModel):
    """Generated behavior evidence for one completed SFT checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-sft-generation-report"] = (
        "lm-from-zero-sft-generation-report"
    )
    format_version: Literal[1] = 1
    generated_at_utc: datetime
    run_manifest_sha256: Sha256
    checkpoint_manifest_sha256: Sha256
    checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    model_config_sha256: Sha256
    tokenizer_sha256: Sha256
    device: Literal["cuda"] = "cuda"
    strategy: Literal["greedy"] = "greedy"
    max_new_tokens: Annotated[int, Field(gt=0)]
    seed: int
    cases: tuple[SFTGenerationCase, ...]
    model_forwards: Annotated[int, Field(gt=0)]
    generated_token_count: Annotated[int, Field(ge=0)]
    elapsed_seconds: Annotated[float, Field(gt=0)]
    tokens_per_second: Annotated[float, Field(ge=0)]

    def canonical_bytes(self) -> bytes:
        """Return stable bytes for the generated report."""

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


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    if temporary.exists():
        raise SFTEvaluationError(f"incomplete evaluation artifact exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _chat_prefix(tokenizer: ByteBPE, prompt: str) -> list[int]:
    conversation = Conversation(
        messages=(
            ChatMessage(role="user", content=prompt),
            ChatMessage(role="assistant", content=""),
        )
    )
    user = conversation.messages[0]
    return [
        SPECIAL_TOKEN_IDS["<|bos|>"],
        SPECIAL_TOKEN_IDS["<|user|>"],
        *tokenizer.encode(user.content),
        SPECIAL_TOKEN_IDS["<|end|>"],
        SPECIAL_TOKEN_IDS["<|assistant|>"],
    ]


def run_sft_generation(
    *,
    run_directory: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    prompts: tuple[str, ...],
    max_new_tokens: int = 64,
    seed: int = 1_337,
) -> SFTGenerationReport:
    """Generate deterministic continuations from a completed SFT run."""

    if not torch.cuda.is_available():
        raise SFTEvaluationError("CUDA is required for SFT generation evaluation")
    if not prompts:
        raise SFTEvaluationError("SFT evaluation requires at least one prompt")
    run_root = Path(run_directory)
    run_manifest_path = run_root / "manifest.json"
    run_manifest = SFTRunManifest.model_validate_json(
        run_manifest_path.read_text(encoding="utf-8")
    )
    checkpoint = run_root / "checkpoints" / run_manifest.final_checkpoint_id
    checkpoint_manifest_path = checkpoint / "manifest.json"
    checkpoint_manifest = SFTCheckpointManifest.model_validate_json(
        checkpoint_manifest_path.read_text(encoding="utf-8")
    )
    if not checkpoint_manifest.complete:
        raise SFTEvaluationError("SFT evaluation requires a complete checkpoint")
    tokenizer = ByteBPE.load(tokenizer_path)
    if tokenizer.model_hash != checkpoint_manifest.tokenizer_sha256:
        raise SFTEvaluationError("tokenizer does not match the SFT checkpoint")
    source_checkpoint = run_root.parent.parent.parent.parent / Path(
        run_manifest.source_checkpoint_directory
    )
    source_checkpoint_manifest = validate_checkpoint(source_checkpoint)
    if _sha256_file(source_checkpoint / "manifest.json") != (
        checkpoint_manifest.source_checkpoint_manifest_sha256
    ):
        raise SFTEvaluationError("source checkpoint manifest hash does not match")
    model_config = Olmo2Config.model_validate(
        source_checkpoint_manifest.binding.resolved_model_config
    )
    model = Olmo2ForCausalLM(model_config, variant=checkpoint_manifest.model_variant)
    model.load_state_dict(
        load_file(str(checkpoint / checkpoint_manifest.model_artifact.filename))
    )
    model.to(torch.device("cuda"))
    model.eval()
    prompt_ids = [_chat_prefix(tokenizer, prompt) for prompt in prompts]
    generation_config = CausalGenerationConfig(
        strategy="greedy",
        max_new_tokens=max_new_tokens,
        seed=seed,
    )
    result = generate_causal(model, prompt_ids, generation_config)
    cases = tuple(
        SFTGenerationCase(
            prompt=prompt,
            prompt_token_count=len(prompt_ids[index]),
            generated_text=tokenizer.decode(
                result.generated_token_ids[index],
                render_special=True,
                errors="replace",
            ),
            generated_token_count=len(result.generated_token_ids[index]),
            stop_reason=result.stop_reasons[index],
        )
        for index, prompt in enumerate(prompts)
    )
    report = SFTGenerationReport(
        generated_at_utc=datetime.now(UTC),
        run_manifest_sha256=_sha256_file(run_manifest_path),
        checkpoint_manifest_sha256=_sha256_file(checkpoint_manifest_path),
        checkpoint_id=checkpoint_manifest.checkpoint_id,
        model_config_sha256=checkpoint_manifest.model_config_sha256,
        tokenizer_sha256=checkpoint_manifest.tokenizer_sha256,
        max_new_tokens=max_new_tokens,
        seed=seed,
        cases=cases,
        model_forwards=result.model_forwards,
        generated_token_count=result.generated_token_count,
        elapsed_seconds=result.elapsed_seconds,
        tokens_per_second=result.tokens_per_second,
    )
    _atomic_write(Path(output_path), report.canonical_bytes() + b"\n")
    del model
    torch.cuda.empty_cache()
    return report
