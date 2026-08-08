"""Held-out preference and behavior evaluation for completed SFT and DPO runs."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

import torch
import torch.nn.functional as functional
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file
from torch import Tensor
from tqdm import tqdm  # type: ignore[import-untyped]

from lm_from_zero.generation import CausalGenerationConfig, generate_causal
from lm_from_zero.models import DenseModelVariant, Olmo2Config, Olmo2ForCausalLM
from lm_from_zero.post_training.chat import DEFAULT_CHAT_TEMPLATE
from lm_from_zero.post_training.dpo import dpo_objective, masked_sequence_logprob
from lm_from_zero.post_training.dpo_train import (
    DPOCheckpointManifest,
    DPORunManifest,
    _iter_preference_batches,
    _PreferenceBatch,
)
from lm_from_zero.post_training.preference_dataset import PreferenceHoldoutManifest
from lm_from_zero.post_training.sft_train import (
    SFTCheckpointManifest,
    SFTRunManifest,
)
from lm_from_zero.tokenizer.bpe import SPECIAL_TOKEN_IDS, ByteBPE
from lm_from_zero.training import validate_checkpoint

Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
DEFAULT_DPO_BEHAVIOR_PROMPTS: tuple[str, ...] = (
    "Explain why the sky is blue in one sentence.",
    "Write a short bedtime story about a red kite.",
    "List three steps for making tea.",
)


class PreferenceEvaluationError(RuntimeError):
    """Raised when held-out preference evaluation cannot be run safely."""


class PreferenceEvaluationConfig(BaseModel):
    """Pinned execution contract for held-out SFT-versus-DPO evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-preference-evaluation-config"] = (
        "lm-from-zero-preference-evaluation-config"
    )
    format_version: Literal[1] = 1
    max_length: Annotated[int, Field(gt=1)] = 1_024
    batch_size: Annotated[int, Field(gt=0)] = 2
    bucket_size: Annotated[int, Field(gt=0)] = 256
    max_pairs: Annotated[int | None, Field(gt=0)] = None
    beta: Annotated[float, Field(gt=0)] = 0.1
    generation_max_new_tokens: Annotated[int, Field(gt=0)] = 64
    seed: int = 1_337
    device: Literal["cuda"] = "cuda"
    precision: Literal["bf16"] = "bf16"

    def canonical_json(self) -> str:
        """Return the stable configuration encoding."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def config_sha256(self) -> str:
        """Return the SHA-256 binding for this evaluation configuration."""

        return sha256(self.canonical_json().encode()).hexdigest()


class PreferenceModelSummary(BaseModel):
    """Aggregate held-out preference scores for one checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    role: Literal["sft_reference", "dpo_policy"]
    checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    checkpoint_manifest_sha256: Sha256
    model_config_sha256: Sha256
    pair_count: Annotated[int, Field(gt=0)]
    preference_accuracy: Annotated[float, Field(ge=0, le=1)]
    mean_chosen_sequence_logprob: float
    mean_rejected_sequence_logprob: float
    mean_preference_margin: float
    mean_chosen_response_tokens: Annotated[float, Field(gt=0)]
    mean_rejected_response_tokens: Annotated[float, Field(gt=0)]
    truncated_pairs: Annotated[int, Field(ge=0)]
    truncation_rate: Annotated[float, Field(ge=0, le=1)]


class BehaviorCompletion(BaseModel):
    """One deterministic native completion from an evaluated checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_text: str
    generated_token_count: Annotated[int, Field(ge=0)]
    stop_reason: Literal["eos", "max_new_tokens"]


class PreferenceBehaviorCase(BaseModel):
    """Side-by-side deterministic behavior evidence for one fixed prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str = Field(min_length=1)
    prompt_token_count: Annotated[int, Field(gt=0)]
    sft_reference: BehaviorCompletion
    dpo_policy: BehaviorCompletion


class BehaviorGenerationSummary(BaseModel):
    """Measured native generation work for a behavior panel."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    model_forwards: Annotated[int, Field(gt=0)]
    generated_token_count: Annotated[int, Field(ge=0)]
    elapsed_seconds: Annotated[float, Field(gt=0)]
    tokens_per_second: Annotated[float, Field(ge=0)]


class DPOHeldoutEvaluationReport(BaseModel):
    """Generated, provenance-bound SFT-versus-DPO held-out report."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-dpo-heldout-evaluation-report"] = (
        "lm-from-zero-dpo-heldout-evaluation-report"
    )
    format_version: Literal[1] = 1
    evaluated_at_utc: datetime
    holdout_manifest_sha256: Sha256
    holdout_records_sha256: Sha256
    training_manifest_sha256: Sha256
    evaluation_mode: Literal["smoke", "full"]
    evaluated_pairs: Annotated[int, Field(gt=0)]
    evaluation_config: PreferenceEvaluationConfig
    evaluation_config_sha256: Sha256
    sft_reference: PreferenceModelSummary
    dpo_policy: PreferenceModelSummary
    mean_dpo_loss: Annotated[float, Field(ge=0)]
    dpo_preference_accuracy: Annotated[float, Field(ge=0, le=1)]
    mean_reward_margin: float
    mean_chosen_reward: float
    mean_rejected_reward: float
    mean_policy_forward_kl_per_response_token: Annotated[float, Field(ge=0)]
    behavior_prompt_sha256: Sha256
    behavior_cases: tuple[PreferenceBehaviorCase, ...]
    sft_generation: BehaviorGenerationSummary
    dpo_generation: BehaviorGenerationSummary
    elapsed_seconds: Annotated[float, Field(gt=0)]
    pairs_per_second: Annotated[float, Field(gt=0)]
    peak_cuda_memory_allocated_bytes: Annotated[int, Field(ge=0)]
    peak_cuda_memory_reserved_bytes: Annotated[int, Field(ge=0)]

    def canonical_bytes(self) -> bytes:
        """Return stable bytes for atomic report publication."""

        return (
            json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )


@dataclass(frozen=True, slots=True)
class _LoadedModel:
    model: Olmo2ForCausalLM
    checkpoint_id: str
    checkpoint_manifest_sha256: str
    model_config_sha256: str


@dataclass(frozen=True, slots=True)
class DPOPolicyForInference:
    """A validated final DPO policy suitable for local inference or export."""

    model: Olmo2ForCausalLM
    checkpoint_id: str
    checkpoint_manifest_sha256: str
    model_config_sha256: str
    model_variant: DenseModelVariant


@dataclass(slots=True)
class _ScoreAccumulator:
    role: Literal["sft_reference", "dpo_policy"]
    checkpoint_id: str
    checkpoint_manifest_sha256: str
    model_config_sha256: str
    pair_count: int = 0
    preferred_count: int = 0
    chosen_logprob_total: float = 0.0
    rejected_logprob_total: float = 0.0
    chosen_response_tokens: int = 0
    rejected_response_tokens: int = 0
    truncated_pairs: int = 0

    def append(
        self,
        chosen_logprobs: Tensor,
        rejected_logprobs: Tensor,
        batch: _PreferenceBatch,
    ) -> None:
        """Accumulate one validated evaluation batch."""

        batch_size = len(batch.indices)
        if chosen_logprobs.shape != (batch_size,) or rejected_logprobs.shape != (
            batch_size,
        ):
            raise PreferenceEvaluationError(
                "preference score shape disagrees with batch"
            )
        self.pair_count += batch_size
        self.preferred_count += int((chosen_logprobs > rejected_logprobs).sum().item())
        self.chosen_logprob_total += float(chosen_logprobs.sum().item())
        self.rejected_logprob_total += float(rejected_logprobs.sum().item())
        self.chosen_response_tokens += sum(batch.chosen_response_tokens)
        self.rejected_response_tokens += sum(batch.rejected_response_tokens)
        self.truncated_pairs += sum(
            chosen or rejected
            for chosen, rejected in zip(
                batch.chosen_truncated,
                batch.rejected_truncated,
                strict=True,
            )
        )

    def summary(self) -> PreferenceModelSummary:
        """Return the immutable aggregate summary."""

        if self.pair_count <= 0:
            raise PreferenceEvaluationError("held-out evaluation produced no pairs")
        return PreferenceModelSummary(
            role=self.role,
            checkpoint_id=self.checkpoint_id,
            checkpoint_manifest_sha256=self.checkpoint_manifest_sha256,
            model_config_sha256=self.model_config_sha256,
            pair_count=self.pair_count,
            preference_accuracy=self.preferred_count / self.pair_count,
            mean_chosen_sequence_logprob=self.chosen_logprob_total / self.pair_count,
            mean_rejected_sequence_logprob=self.rejected_logprob_total
            / self.pair_count,
            mean_preference_margin=(
                self.chosen_logprob_total - self.rejected_logprob_total
            )
            / self.pair_count,
            mean_chosen_response_tokens=self.chosen_response_tokens / self.pair_count,
            mean_rejected_response_tokens=self.rejected_response_tokens
            / self.pair_count,
            truncated_pairs=self.truncated_pairs,
            truncation_rate=self.truncated_pairs / self.pair_count,
        )


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
        raise PreferenceEvaluationError(
            f"incomplete evaluation artifact exists: {temporary}"
        )
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _resolve_project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _validate_artifact(
    checkpoint: Path,
    *,
    filename: str,
    size_bytes: int,
    expected_sha256: str,
) -> None:
    artifact = checkpoint / filename
    if not artifact.is_file():
        raise PreferenceEvaluationError(f"checkpoint artifact is missing: {filename}")
    if artifact.stat().st_size != size_bytes:
        raise PreferenceEvaluationError(
            f"checkpoint artifact size mismatch: {filename}"
        )
    if _sha256_file(artifact) != expected_sha256:
        raise PreferenceEvaluationError(
            f"checkpoint artifact hash mismatch: {filename}"
        )


def _load_holdout(
    manifest_path: Path,
    tokenizer: ByteBPE,
    config: PreferenceEvaluationConfig,
) -> PreferenceHoldoutManifest:
    try:
        manifest = PreferenceHoldoutManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise PreferenceEvaluationError(
            "held-out preference manifest is invalid"
        ) from error
    records_path = manifest_path.parent / manifest.records_jsonl
    if _sha256_file(records_path) != manifest.records_sha256:
        raise PreferenceEvaluationError(
            "held-out preference records do not match manifest"
        )
    if tokenizer.model_hash != manifest.tokenizer_hash:
        raise PreferenceEvaluationError("tokenizer does not match held-out manifest")
    if manifest.chat_template_hash != DEFAULT_CHAT_TEMPLATE.template_hash:
        raise PreferenceEvaluationError("held-out chat template is unsupported")
    if config.max_length != manifest.max_length:
        raise PreferenceEvaluationError(
            "evaluation length does not match held-out manifest"
        )
    if config.max_pairs is not None and config.max_pairs > manifest.selected_pairs:
        raise PreferenceEvaluationError("evaluation pair bound exceeds held-out split")
    return manifest


def _load_sft_reference(run_root: Path, tokenizer: ByteBPE) -> _LoadedModel:
    run_manifest_path = run_root / "manifest.json"
    try:
        run_manifest = SFTRunManifest.model_validate_json(
            run_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise PreferenceEvaluationError("SFT run manifest is invalid") from error
    checkpoint = run_root / "checkpoints" / run_manifest.final_checkpoint_id
    checkpoint_manifest_path = checkpoint / "manifest.json"
    checkpoint_hash = _sha256_file(checkpoint_manifest_path)
    try:
        checkpoint_manifest = SFTCheckpointManifest.model_validate_json(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise PreferenceEvaluationError("SFT checkpoint manifest is invalid") from error
    if (
        not checkpoint_manifest.complete
        or checkpoint_manifest.checkpoint_id != run_manifest.final_checkpoint_id
        or checkpoint_hash != run_manifest.final_checkpoint_manifest_sha256
    ):
        raise PreferenceEvaluationError("SFT final checkpoint does not match its run")
    for artifact in (
        checkpoint_manifest.model_artifact,
        checkpoint_manifest.recovery_artifact,
    ):
        _validate_artifact(
            checkpoint,
            filename=artifact.filename,
            size_bytes=artifact.size_bytes,
            expected_sha256=artifact.sha256,
        )
    if tokenizer.model_hash != checkpoint_manifest.tokenizer_sha256:
        raise PreferenceEvaluationError("tokenizer does not match SFT checkpoint")
    source_checkpoint = _resolve_project_path(run_manifest.source_checkpoint_directory)
    source_manifest = validate_checkpoint(source_checkpoint)
    if _sha256_file(source_checkpoint / "manifest.json") != (
        checkpoint_manifest.source_checkpoint_manifest_sha256
    ):
        raise PreferenceEvaluationError("SFT source checkpoint does not match run")
    if source_manifest.binding.architecture != "olmo2":
        raise PreferenceEvaluationError("held-out DPO evaluation requires an OLMo2 SFT")
    model_config = Olmo2Config.model_validate(
        source_manifest.binding.resolved_model_config
    )
    if model_config.config_hash != checkpoint_manifest.model_config_sha256:
        raise PreferenceEvaluationError("SFT model configuration hash does not match")
    model = Olmo2ForCausalLM(model_config, variant=checkpoint_manifest.model_variant)
    try:
        model.load_state_dict(
            load_file(
                str(checkpoint / checkpoint_manifest.model_artifact.filename),
                device="cpu",
            ),
            strict=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise PreferenceEvaluationError(
            "SFT checkpoint model weights cannot be loaded"
        ) from error
    return _LoadedModel(
        model=model,
        checkpoint_id=checkpoint_manifest.checkpoint_id,
        checkpoint_manifest_sha256=checkpoint_hash,
        model_config_sha256=checkpoint_manifest.model_config_sha256,
    )


def _load_dpo_policy(
    run_root: Path,
    tokenizer: ByteBPE,
    reference: _LoadedModel,
) -> tuple[_LoadedModel, DPORunManifest]:
    run_manifest_path = run_root / "manifest.json"
    try:
        run_manifest = DPORunManifest.model_validate_json(
            run_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise PreferenceEvaluationError("DPO run manifest is invalid") from error
    checkpoint = run_root / "checkpoints" / run_manifest.final_checkpoint_id
    checkpoint_manifest_path = checkpoint / "manifest.json"
    checkpoint_hash = _sha256_file(checkpoint_manifest_path)
    try:
        checkpoint_manifest = DPOCheckpointManifest.model_validate_json(
            checkpoint_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise PreferenceEvaluationError("DPO checkpoint manifest is invalid") from error
    if (
        not checkpoint_manifest.complete
        or checkpoint_manifest.checkpoint_id != run_manifest.final_checkpoint_id
        or checkpoint_hash != run_manifest.final_checkpoint_manifest_sha256
    ):
        raise PreferenceEvaluationError("DPO final checkpoint does not match its run")
    if (
        run_manifest.source_sft_checkpoint_id != reference.checkpoint_id
        or run_manifest.source_sft_checkpoint_manifest_sha256
        != reference.checkpoint_manifest_sha256
        or checkpoint_manifest.source_sft_checkpoint_id != reference.checkpoint_id
        or checkpoint_manifest.source_sft_checkpoint_manifest_sha256
        != reference.checkpoint_manifest_sha256
    ):
        raise PreferenceEvaluationError(
            "DPO policy is not bound to the supplied SFT run"
        )
    if (
        tokenizer.model_hash != checkpoint_manifest.tokenizer_sha256
        or checkpoint_manifest.model_config_sha256 != reference.model_config_sha256
        or run_manifest.model_config_sha256 != reference.model_config_sha256
    ):
        raise PreferenceEvaluationError(
            "DPO policy model or tokenizer does not match SFT"
        )
    for artifact in (
        checkpoint_manifest.model_artifact,
        checkpoint_manifest.recovery_artifact,
    ):
        _validate_artifact(
            checkpoint,
            filename=artifact.filename,
            size_bytes=artifact.size_bytes,
            expected_sha256=artifact.sha256,
        )
    model = Olmo2ForCausalLM(
        reference.model.config,
        variant=checkpoint_manifest.model_variant,
    )
    try:
        model.load_state_dict(
            load_file(
                str(checkpoint / checkpoint_manifest.model_artifact.filename),
                device="cpu",
            ),
            strict=True,
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise PreferenceEvaluationError(
            "DPO checkpoint model weights cannot be loaded"
        ) from error
    return (
        _LoadedModel(
            model=model,
            checkpoint_id=checkpoint_manifest.checkpoint_id,
            checkpoint_manifest_sha256=checkpoint_hash,
            model_config_sha256=checkpoint_manifest.model_config_sha256,
        ),
        run_manifest,
    )


def load_final_dpo_policy(
    checkpoint_path: str | Path,
    tokenizer: ByteBPE,
) -> DPOPolicyForInference:
    """Validate a final DPO policy and every bound SFT/pretraining artifact."""

    checkpoint = Path(checkpoint_path)
    if checkpoint.parent.name != "checkpoints":
        raise PreferenceEvaluationError(
            "DPO inference requires a checkpoint under its run checkpoints"
        )
    run_root = checkpoint.parent.parent
    try:
        run_manifest = DPORunManifest.model_validate_json(
            (run_root / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise PreferenceEvaluationError("DPO run manifest is invalid") from error
    source_sft_checkpoint = _resolve_project_path(
        run_manifest.source_sft_checkpoint_directory
    )
    reference = _load_sft_reference(source_sft_checkpoint.parent.parent, tokenizer)
    policy, validated_run_manifest = _load_dpo_policy(run_root, tokenizer, reference)
    expected_checkpoint = (
        run_root / "checkpoints" / validated_run_manifest.final_checkpoint_id
    )
    if checkpoint.resolve() != expected_checkpoint.resolve():
        raise PreferenceEvaluationError(
            "DPO inference requires the run's complete final checkpoint"
        )
    return DPOPolicyForInference(
        model=policy.model,
        checkpoint_id=policy.checkpoint_id,
        checkpoint_manifest_sha256=policy.checkpoint_manifest_sha256,
        model_config_sha256=policy.model_config_sha256,
        model_variant=validated_run_manifest.model_variant,
    )


def _sequence_logprobs_and_policy_kl(
    policy_logits: Tensor,
    reference_logits: Tensor,
    input_ids: Tensor,
    response_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return sequence scores and exact forward KL over response-token positions."""

    if policy_logits.shape != reference_logits.shape:
        raise PreferenceEvaluationError(
            "policy and reference logits have different shapes"
        )
    policy_logprobs = masked_sequence_logprob(policy_logits, input_ids, response_mask)
    reference_logprobs = masked_sequence_logprob(
        reference_logits,
        input_ids,
        response_mask,
    )
    shifted_mask = response_mask[:, 1:]
    policy_log_distribution = functional.log_softmax(
        policy_logits[:, :-1].float(), dim=-1
    )
    reference_log_distribution = functional.log_softmax(
        reference_logits[:, :-1].float(),
        dim=-1,
    )
    token_kl = (
        policy_log_distribution.exp()
        * (policy_log_distribution - reference_log_distribution)
    ).sum(dim=-1)
    sequence_kl = (token_kl * shifted_mask).sum(dim=1)
    response_token_counts = shifted_mask.sum(dim=1)
    return policy_logprobs, reference_logprobs, sequence_kl, response_token_counts


def _score_batch(
    reference: Olmo2ForCausalLM,
    policy: Olmo2ForCausalLM,
    batch: _PreferenceBatch,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    input_ids = torch.cat([batch.chosen_input_ids, batch.rejected_input_ids], dim=0).to(
        device
    )
    response_mask = torch.cat(
        [batch.chosen_response_mask, batch.rejected_response_mask],
        dim=0,
    ).to(device)
    attention_mask = torch.cat(
        [batch.chosen_attention_mask, batch.rejected_attention_mask],
        dim=0,
    ).to(device)
    with (
        torch.inference_mode(),
        torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    ):
        reference_logits = reference(input_ids, attention_mask=attention_mask).logits
        policy_logits = policy(input_ids, attention_mask=attention_mask).logits
        policy_logprobs, reference_logprobs, sequence_kl, response_token_counts = (
            _sequence_logprobs_and_policy_kl(
                policy_logits,
                reference_logits,
                input_ids,
                response_mask,
            )
        )
    batch_size = len(batch.indices)
    return (
        reference_logprobs[:batch_size],
        reference_logprobs[batch_size:],
        policy_logprobs[:batch_size],
        policy_logprobs[batch_size:],
        sequence_kl,
        response_token_counts,
    )


def _chat_prefix(tokenizer: ByteBPE, prompt: str) -> list[int]:
    return [
        SPECIAL_TOKEN_IDS["<|bos|>"],
        SPECIAL_TOKEN_IDS["<|user|>"],
        *tokenizer.encode(prompt),
        SPECIAL_TOKEN_IDS["<|end|>"],
        SPECIAL_TOKEN_IDS["<|assistant|>"],
    ]


def _generation_summary(result: object) -> BehaviorGenerationSummary:
    from lm_from_zero.generation import CausalGenerationResult

    if not isinstance(result, CausalGenerationResult):
        raise PreferenceEvaluationError("native generation returned an invalid result")
    return BehaviorGenerationSummary(
        model_forwards=result.model_forwards,
        generated_token_count=result.generated_token_count,
        elapsed_seconds=result.elapsed_seconds,
        tokens_per_second=result.tokens_per_second,
    )


def _behavior_panel(
    reference: Olmo2ForCausalLM,
    policy: Olmo2ForCausalLM,
    tokenizer: ByteBPE,
    config: PreferenceEvaluationConfig,
) -> tuple[
    tuple[PreferenceBehaviorCase, ...],
    BehaviorGenerationSummary,
    BehaviorGenerationSummary,
]:
    prompt_ids = [
        _chat_prefix(tokenizer, prompt) for prompt in DEFAULT_DPO_BEHAVIOR_PROMPTS
    ]
    generation_config = CausalGenerationConfig(
        strategy="greedy",
        max_new_tokens=config.generation_max_new_tokens,
        seed=config.seed,
    )
    reference_result = generate_causal(reference, prompt_ids, generation_config)
    policy_result = generate_causal(policy, prompt_ids, generation_config)
    cases = tuple(
        PreferenceBehaviorCase(
            prompt=prompt,
            prompt_token_count=len(prompt_ids[index]),
            sft_reference=BehaviorCompletion(
                generated_text=tokenizer.decode(
                    reference_result.generated_token_ids[index],
                    render_special=True,
                    errors="replace",
                ),
                generated_token_count=len(reference_result.generated_token_ids[index]),
                stop_reason=reference_result.stop_reasons[index],
            ),
            dpo_policy=BehaviorCompletion(
                generated_text=tokenizer.decode(
                    policy_result.generated_token_ids[index],
                    render_special=True,
                    errors="replace",
                ),
                generated_token_count=len(policy_result.generated_token_ids[index]),
                stop_reason=policy_result.stop_reasons[index],
            ),
        )
        for index, prompt in enumerate(DEFAULT_DPO_BEHAVIOR_PROMPTS)
    )
    return (
        cases,
        _generation_summary(reference_result),
        _generation_summary(policy_result),
    )


def run_dpo_heldout_evaluation(
    *,
    holdout_manifest_path: str | Path,
    tokenizer_path: str | Path,
    sft_run_directory: str | Path,
    dpo_run_directory: str | Path,
    output_path: str | Path,
    config: PreferenceEvaluationConfig | None = None,
) -> DPOHeldoutEvaluationReport:
    """Evaluate the final SFT reference and DPO policy on a disjoint preference set."""

    if not torch.cuda.is_available():
        raise PreferenceEvaluationError("CUDA is required for held-out DPO evaluation")
    if config is None:
        config = PreferenceEvaluationConfig()
    holdout_path = Path(holdout_manifest_path)
    tokenizer_file = Path(tokenizer_path)
    if not holdout_path.is_file():
        raise PreferenceEvaluationError("held-out preference manifest does not exist")
    if not tokenizer_file.is_file():
        raise PreferenceEvaluationError("preference tokenizer does not exist")
    try:
        tokenizer = ByteBPE.load(tokenizer_file)
    except (OSError, ValueError) as error:
        raise PreferenceEvaluationError("preference tokenizer is invalid") from error
    holdout = _load_holdout(holdout_path, tokenizer, config)
    reference = _load_sft_reference(Path(sft_run_directory), tokenizer)
    policy, dpo_run = _load_dpo_policy(Path(dpo_run_directory), tokenizer, reference)
    if dpo_run.dataset_manifest_sha256 != holdout.train_manifest_sha256:
        raise PreferenceEvaluationError(
            "held-out split does not match DPO training mix"
        )
    if dpo_run.dataset_records_sha256 != holdout.train_records_sha256:
        raise PreferenceEvaluationError(
            "held-out split does not match DPO training records"
        )
    if config.beta != dpo_run.dpo_config.beta:
        raise PreferenceEvaluationError("evaluation beta does not match DPO training")
    if config.max_length != dpo_run.dpo_config.max_length:
        raise PreferenceEvaluationError("evaluation length does not match DPO training")
    evaluated_pairs = config.max_pairs or holdout.selected_pairs
    evaluation_mode: Literal["smoke", "full"] = (
        "smoke" if config.max_pairs is not None else "full"
    )

    device = torch.device(config.device)
    reference.model.to(device)
    policy.model.to(device)
    reference.model.eval()
    policy.model.eval()
    reference_scores = _ScoreAccumulator(
        role="sft_reference",
        checkpoint_id=reference.checkpoint_id,
        checkpoint_manifest_sha256=reference.checkpoint_manifest_sha256,
        model_config_sha256=reference.model_config_sha256,
    )
    policy_scores = _ScoreAccumulator(
        role="dpo_policy",
        checkpoint_id=policy.checkpoint_id,
        checkpoint_manifest_sha256=policy.checkpoint_manifest_sha256,
        model_config_sha256=policy.model_config_sha256,
    )
    dpo_loss_total = 0.0
    dpo_preferred_count = 0
    reward_margin_total = 0.0
    chosen_reward_total = 0.0
    rejected_reward_total = 0.0
    policy_kl_total = 0.0
    response_token_total = 0
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    records_path = holdout_path.parent / holdout.records_jsonl
    progress = tqdm(total=evaluated_pairs, desc="DPO held-out evaluation", unit="pairs")
    try:
        batches = _iter_preference_batches(
            records_path,
            tokenizer,
            expected_records=evaluated_pairs,
            batch_size=config.batch_size,
            bucket_size=config.bucket_size,
            max_length=config.max_length,
            max_records=config.max_pairs,
        )
        for batch in batches:
            (
                reference_chosen,
                reference_rejected,
                policy_chosen,
                policy_rejected,
                sequence_kl,
                response_token_counts,
            ) = _score_batch(reference.model, policy.model, batch, device)
            reference_scores.append(reference_chosen, reference_rejected, batch)
            policy_scores.append(policy_chosen, policy_rejected, batch)
            objective = dpo_objective(
                policy_chosen,
                policy_rejected,
                reference_chosen,
                reference_rejected,
                beta=config.beta,
            )
            batch_size = len(batch.indices)
            dpo_loss_total += float(objective.loss.item()) * batch_size
            dpo_preferred_count += int((objective.logits > 0).sum().item())
            reward_margin_total += float(objective.reward_margins.sum().item())
            chosen_reward_total += float(objective.chosen_rewards.sum().item())
            rejected_reward_total += float(objective.rejected_rewards.sum().item())
            policy_kl_total += float(sequence_kl.sum().item())
            response_token_total += int(response_token_counts.sum().item())
            progress.update(batch_size)
    finally:
        progress.close()
    elapsed_seconds = time.perf_counter() - started
    if response_token_total <= 0:
        raise PreferenceEvaluationError(
            "held-out evaluation produced no response tokens"
        )
    reference_summary = reference_scores.summary()
    policy_summary = policy_scores.summary()
    if reference_summary.pair_count != evaluated_pairs:
        raise PreferenceEvaluationError("held-out evaluation did not score every pair")
    behavior_cases, reference_generation, policy_generation = _behavior_panel(
        reference.model,
        policy.model,
        tokenizer,
        config,
    )
    report = DPOHeldoutEvaluationReport(
        evaluated_at_utc=datetime.now(UTC),
        holdout_manifest_sha256=_sha256_file(holdout_path),
        holdout_records_sha256=holdout.records_sha256,
        training_manifest_sha256=holdout.train_manifest_sha256,
        evaluation_mode=evaluation_mode,
        evaluated_pairs=evaluated_pairs,
        evaluation_config=config,
        evaluation_config_sha256=config.config_sha256,
        sft_reference=reference_summary,
        dpo_policy=policy_summary,
        mean_dpo_loss=dpo_loss_total / evaluated_pairs,
        dpo_preference_accuracy=dpo_preferred_count / evaluated_pairs,
        mean_reward_margin=reward_margin_total / evaluated_pairs,
        mean_chosen_reward=chosen_reward_total / evaluated_pairs,
        mean_rejected_reward=rejected_reward_total / evaluated_pairs,
        mean_policy_forward_kl_per_response_token=policy_kl_total
        / response_token_total,
        behavior_prompt_sha256=sha256(
            json.dumps(
                DEFAULT_DPO_BEHAVIOR_PROMPTS,
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        behavior_cases=behavior_cases,
        sft_generation=reference_generation,
        dpo_generation=policy_generation,
        elapsed_seconds=elapsed_seconds,
        pairs_per_second=evaluated_pairs / elapsed_seconds,
        peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_cuda_memory_reserved_bytes=torch.cuda.max_memory_reserved(device),
    )
    _atomic_write(Path(output_path), report.canonical_bytes())
    del policy, reference
    torch.cuda.empty_cache()
    return report
