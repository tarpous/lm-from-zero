"""Native iterative denoising for the project-owned masked diffusion model."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, Literal, Self, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from lm_from_zero.models import MaskedDiffusionForMaskedLM

DEFAULT_SUPPRESSED_TOKEN_IDS = (0, 1, 3, 4, 5, 6, 7)
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class DiffusionGenerationError(RuntimeError):
    """Raised when iterative denoising cannot satisfy its request contract."""


class DiffusionGenerationConfig(BaseModel):
    """Validated canvas, reveal, sampling, and remasking policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal["greedy", "sample"] = "greedy"
    response_length: Annotated[int, Field(gt=0)] = 64
    diffusion_steps: Annotated[int | None, Field(gt=0)] = None
    reveal_schedule: Literal["linear", "cosine"] = "linear"
    temperature: Annotated[float, Field(gt=0)] = 1.0
    remask_strategy: Literal["none", "low_confidence"] = "none"
    remask_fraction: Annotated[float, Field(ge=0, lt=1)] = 0.0
    seed: int = 1337
    allow_raw_special_tokens: bool = False

    @model_validator(mode="after")
    def validate_strategy_options(self) -> Self:
        if self.strategy == "greedy" and self.temperature != 1.0:
            raise ValueError("temperature requires strategy='sample'")
        if self.remask_strategy == "none" and self.remask_fraction != 0:
            raise ValueError("remask_fraction requires low_confidence remasking")
        if self.remask_strategy == "low_confidence" and self.remask_fraction == 0:
            raise ValueError("low_confidence remasking requires a positive fraction")
        return self

    @property
    def resolved_steps(self) -> int:
        """Default to one denoising step per requested response token."""

        return (
            self.response_length
            if self.diffusion_steps is None
            else self.diffusion_steps
        )


class DiffusionGenerationEvent(BaseModel):
    """One streamed reveal/remask transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: Annotated[int, Field(gt=0)]
    model_forwards: Annotated[int, Field(gt=0)]
    revealed_token_ids: tuple[tuple[int | None, ...], ...]
    remasked_positions: tuple[tuple[int, ...], ...]
    remaining_masks: tuple[Annotated[int, Field(ge=0)], ...]


class DiffusionGenerationResult(BaseModel):
    """Final response tokens and measured iterative-denoising work."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    prompt_token_counts: tuple[Annotated[int, Field(gt=0)], ...]
    response_canvas_length: Annotated[int, Field(gt=0)]
    generated_token_ids: tuple[tuple[int, ...], ...]
    stop_reasons: tuple[Literal["eos", "canvas_complete"], ...]
    diffusion_steps: Annotated[int, Field(gt=0)]
    model_forwards: Annotated[int, Field(gt=0)]
    generated_token_count: Annotated[int, Field(ge=0)]
    elapsed_seconds: Annotated[float, Field(gt=0)]
    tokens_per_second: Annotated[float, Field(ge=0)]

    @model_validator(mode="after")
    def validate_batch(self) -> Self:
        batch_size = len(self.prompt_token_counts)
        if (
            len(self.generated_token_ids) != batch_size
            or len(self.stop_reasons) != batch_size
        ):
            raise ValueError("generation result batch fields have different lengths")
        if sum(map(len, self.generated_token_ids)) != self.generated_token_count:
            raise ValueError("generated token count does not match the sequences")
        if self.model_forwards != self.diffusion_steps:
            raise ValueError("reference sampler requires one model forward per step")
        return self

    def canonical_json(self) -> str:
        """Return deterministic JSON for CLI and append-only records."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


class DiffusionGenerationRecord(BaseModel):
    """Canonical provenance wrapper for one denoising request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-diffusion-generation"] = (
        "lm-from-zero-diffusion-generation"
    )
    format_version: Literal[1] = 1
    generated_at_utc: datetime
    model_config_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    tokenizer_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    prompt_token_sha256: Annotated[str, Field(pattern=SHA256_PATTERN)]
    result: DiffusionGenerationResult

    def canonical_json(self) -> str:
        """Return deterministic JSON for append-only evidence."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


def create_diffusion_generation_record(
    result: DiffusionGenerationResult,
    prompts: Sequence[Sequence[int]],
    *,
    model_config_sha256: str,
    tokenizer_sha256: str,
) -> DiffusionGenerationRecord:
    """Bind denoising output to model, tokenizer, and prompt tokens."""

    prompt_payload = json.dumps(
        [list(prompt) for prompt in prompts],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode()
    return DiffusionGenerationRecord(
        generated_at_utc=datetime.now(UTC),
        model_config_sha256=model_config_sha256,
        tokenizer_sha256=tokenizer_sha256,
        prompt_token_sha256=sha256(prompt_payload).hexdigest(),
        result=result,
    )


def append_diffusion_generation_record(
    path: str | Path,
    record: DiffusionGenerationRecord,
) -> None:
    """Append and fsync one canonical denoising evidence record."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (record.canonical_json() + "\n").encode()
    descriptor = os.open(
        destination,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_request(
    prompts: Sequence[Sequence[int]],
    model: MaskedDiffusionForMaskedLM,
    config: DiffusionGenerationConfig,
) -> None:
    if not prompts:
        raise DiffusionGenerationError("at least one prompt is required")
    if any(not prompt for prompt in prompts):
        raise DiffusionGenerationError("generation prompts cannot be empty")
    if any(
        token_id < 0 or token_id >= model.config.vocab_size
        for prompt in prompts
        for token_id in prompt
    ):
        raise DiffusionGenerationError("a prompt token is outside the model vocabulary")
    if any(
        len(prompt) + config.response_length > model.config.max_position_embeddings
        for prompt in prompts
    ):
        raise DiffusionGenerationError(
            "prompt plus response canvas exceeds model context"
        )


def _initial_canvas(
    prompts: Sequence[Sequence[int]],
    *,
    response_length: int,
    pad_token_id: int,
    mask_token_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    maximum_length = max(len(prompt) + response_length for prompt in prompts)
    canvas = torch.full(
        (len(prompts), maximum_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(canvas, dtype=torch.bool)
    response_mask = torch.zeros_like(canvas, dtype=torch.bool)
    for row, prompt in enumerate(prompts):
        prompt_length = len(prompt)
        canvas[row, :prompt_length] = torch.tensor(
            prompt,
            dtype=torch.long,
            device=device,
        )
        canvas[row, prompt_length : prompt_length + response_length] = mask_token_id
        attention_mask[row, : prompt_length + response_length] = True
        response_mask[row, prompt_length : prompt_length + response_length] = True
    return canvas, attention_mask, response_mask


def _scheduled_reveals(config: DiffusionGenerationConfig, step: int) -> int:
    progress = step / config.resolved_steps
    fraction = (
        progress
        if config.reveal_schedule == "linear"
        else 1.0 - math.cos(math.pi * progress / 2.0)
    )
    if step == config.resolved_steps:
        return config.response_length
    return min(
        config.response_length, max(1, math.ceil(config.response_length * fraction))
    )


def _proposal_scores(
    logits: Tensor,
    config: DiffusionGenerationConfig,
) -> Tensor:
    scores = logits.float().clone()
    suppressed = (
        (7,) if config.allow_raw_special_tokens else DEFAULT_SUPPRESSED_TOKEN_IDS
    )
    scores[..., list(suppressed)] = -torch.inf
    scores /= config.temperature
    if torch.any(torch.isneginf(scores).all(dim=-1)):
        raise DiffusionGenerationError("sampling filters removed every token")
    return scores


def _propose_tokens(
    logits: Tensor,
    config: DiffusionGenerationConfig,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    scores = _proposal_scores(logits, config)
    probabilities = torch.softmax(scores, dim=-1)
    if config.strategy == "greedy":
        confidence, selected = probabilities.max(dim=-1)
        return selected, confidence
    selected = torch.multinomial(
        probabilities.view(-1, probabilities.shape[-1]),
        num_samples=1,
        generator=generator,
    ).view(probabilities.shape[:-1])
    confidence = probabilities.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
    return selected, confidence


def _lowest_confidence_positions(
    response_positions: Tensor,
    confidence: Tensor,
    count: int,
) -> list[int]:
    candidates = torch.nonzero(response_positions, as_tuple=False).flatten().tolist()
    ordered = sorted(candidates, key=lambda index: (float(confidence[index]), index))
    return ordered[:count]


def _highest_confidence_positions(
    masked_positions: Tensor,
    confidence: Tensor,
    count: int,
) -> list[int]:
    candidates = torch.nonzero(masked_positions, as_tuple=False).flatten().tolist()
    ordered = sorted(candidates, key=lambda index: (-float(confidence[index]), index))
    return ordered[:count]


def generate_diffusion(
    model: MaskedDiffusionForMaskedLM,
    prompts: Sequence[Sequence[int]],
    config: DiffusionGenerationConfig,
    *,
    on_step: Callable[[DiffusionGenerationEvent], None] | None = None,
) -> DiffusionGenerationResult:
    """Iteratively fill fixed response canvases and stream reveal transitions."""

    _validate_request(prompts, model, config)
    device = next(model.parameters()).device
    canvas, attention_mask, response_mask = _initial_canvas(
        prompts,
        response_length=config.response_length,
        pad_token_id=model.config.pad_token_id,
        mask_token_id=model.config.mask_token_id,
        device=device,
    )
    prompt_snapshot = canvas.masked_select(~response_mask & attention_mask).clone()
    reveal_confidence = torch.full(
        canvas.shape,
        torch.inf,
        dtype=torch.float32,
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(config.seed)
    was_training = model.training
    model_forwards = 0
    started = perf_counter()
    try:
        model.eval()
        with torch.no_grad():
            for step in range(1, config.resolved_steps + 1):
                remasked_by_row: list[tuple[int, ...]] = []
                if (
                    config.remask_strategy == "low_confidence"
                    and step > 1
                    and step < config.resolved_steps
                ):
                    for row in range(canvas.shape[0]):
                        revealed = response_mask[row] & (
                            canvas[row] != model.config.mask_token_id
                        )
                        count = math.floor(
                            int(revealed.sum().item()) * config.remask_fraction
                        )
                        positions = _lowest_confidence_positions(
                            revealed,
                            reveal_confidence[row],
                            count,
                        )
                        if positions:
                            canvas[row, positions] = model.config.mask_token_id
                            reveal_confidence[row, positions] = torch.inf
                        remasked_by_row.append(tuple(positions))
                else:
                    remasked_by_row = [tuple() for _ in prompts]

                output = cast(Any, model)(
                    canvas,
                    attention_mask=attention_mask,
                )
                model_forwards += 1
                selected, confidence = _propose_tokens(
                    output.logits,
                    config,
                    generator,
                )
                target_revealed = _scheduled_reveals(config, step)
                revealed_events: list[tuple[int | None, ...]] = []
                remaining: list[int] = []
                for row in range(canvas.shape[0]):
                    currently_revealed = response_mask[row] & (
                        canvas[row] != model.config.mask_token_id
                    )
                    reveal_count = max(
                        target_revealed - int(currently_revealed.sum().item()),
                        0,
                    )
                    masked = response_mask[row] & (
                        canvas[row] == model.config.mask_token_id
                    )
                    positions = _highest_confidence_positions(
                        masked,
                        confidence[row],
                        reveal_count,
                    )
                    event_tokens: list[int | None] = [None] * config.response_length
                    prompt_length = len(prompts[row])
                    for position in positions:
                        token_id = int(selected[row, position].item())
                        canvas[row, position] = token_id
                        reveal_confidence[row, position] = confidence[row, position]
                        event_tokens[position - prompt_length] = token_id
                    revealed_events.append(tuple(event_tokens))
                    remaining.append(
                        int(
                            (
                                response_mask[row]
                                & (canvas[row] == model.config.mask_token_id)
                            )
                            .sum()
                            .item()
                        )
                    )
                if on_step is not None:
                    on_step(
                        DiffusionGenerationEvent(
                            step=step,
                            model_forwards=model_forwards,
                            revealed_token_ids=tuple(revealed_events),
                            remasked_positions=tuple(remasked_by_row),
                            remaining_masks=tuple(remaining),
                        )
                    )
    finally:
        model.train(was_training)

    if torch.any(response_mask & (canvas == model.config.mask_token_id)):
        raise DiffusionGenerationError("denoising terminated with masked tokens")
    if not torch.equal(
        canvas.masked_select(~response_mask & attention_mask),
        prompt_snapshot,
    ):
        raise DiffusionGenerationError("denoising mutated prompt tokens")

    generated: list[tuple[int, ...]] = []
    reasons: list[Literal["eos", "canvas_complete"]] = []
    for row, prompt in enumerate(prompts):
        start = len(prompt)
        response = canvas[row, start : start + config.response_length].tolist()
        if model.config.eos_token_id in response:
            eos_index = response.index(model.config.eos_token_id)
            generated.append(tuple(response[: eos_index + 1]))
            reasons.append("eos")
        else:
            generated.append(tuple(response))
            reasons.append("canvas_complete")

    elapsed = perf_counter() - started
    generated_count = sum(map(len, generated))
    return DiffusionGenerationResult(
        prompt_token_counts=tuple(map(len, prompts)),
        response_canvas_length=config.response_length,
        generated_token_ids=tuple(generated),
        stop_reasons=tuple(reasons),
        diffusion_steps=config.resolved_steps,
        model_forwards=model_forwards,
        generated_token_count=generated_count,
        elapsed_seconds=elapsed,
        tokens_per_second=generated_count / elapsed,
    )
