"""Native batched autoregressive generation with dense KV caching."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Annotated, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from lm_from_zero.models import Olmo2ForCausalLM

DEFAULT_SUPPRESSED_TOKEN_IDS = (0, 3, 4, 5, 7)


class GenerationError(RuntimeError):
    """Raised when native generation cannot satisfy its request contract."""


class CausalGenerationConfig(BaseModel):
    """Validated decoding policy for project-owned causal models."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: Literal["greedy", "sample"] = "greedy"
    max_new_tokens: Annotated[int, Field(gt=0)] = 64
    temperature: Annotated[float, Field(gt=0)] = 1.0
    top_k: Annotated[int | None, Field(gt=0)] = None
    top_p: Annotated[float | None, Field(gt=0, le=1)] = None
    seed: int = 1337
    allow_raw_special_tokens: bool = False

    @model_validator(mode="after")
    def validate_strategy(self) -> Self:
        sampling_options = (
            self.temperature != 1.0 or self.top_k is not None or self.top_p is not None
        )
        if self.strategy == "greedy" and sampling_options:
            raise ValueError("temperature, top_k, and top_p require strategy='sample'")
        return self


class CausalGenerationEvent(BaseModel):
    """One immediately emitted batched decoding step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    step: Annotated[int, Field(gt=0)]
    token_ids: tuple[int | None, ...]
    finished: tuple[bool, ...]


class CausalGenerationResult(BaseModel):
    """Final generated tokens, stop reasons, and measured native work."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    prompt_token_counts: tuple[Annotated[int, Field(gt=0)], ...]
    generated_token_ids: tuple[tuple[int, ...], ...]
    stop_reasons: tuple[Literal["eos", "max_new_tokens"], ...]
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
        return self

    def canonical_json(self) -> str:
        """Return deterministic JSON for CLI and append-only records."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )


def _validate_prompts(
    prompts: Sequence[Sequence[int]],
    model: Olmo2ForCausalLM,
    config: CausalGenerationConfig,
) -> None:
    if not prompts:
        raise GenerationError("at least one prompt is required")
    if any(not prompt for prompt in prompts):
        raise GenerationError("generation prompts cannot be empty")
    if any(
        token_id < 0 or token_id >= model.config.vocab_size
        for prompt in prompts
        for token_id in prompt
    ):
        raise GenerationError("a prompt token is outside the model vocabulary")
    maximum_prompt = max(map(len, prompts))
    if maximum_prompt + config.max_new_tokens > model.config.max_position_embeddings:
        raise GenerationError("prompt plus requested generation exceeds model context")
    if config.top_k is not None and config.top_k > model.config.vocab_size:
        raise GenerationError("top_k cannot exceed the model vocabulary size")


def _left_padded_batch(
    prompts: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    maximum_length = max(map(len, prompts))
    input_ids = torch.full(
        (len(prompts), maximum_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, prompt in enumerate(prompts):
        prompt_tensor = torch.tensor(prompt, dtype=torch.long, device=device)
        input_ids[row, -len(prompt) :] = prompt_tensor
        attention_mask[row, -len(prompt) :] = 1
    return input_ids, attention_mask


def _filtered_logits(
    logits: Tensor,
    config: CausalGenerationConfig,
) -> Tensor:
    scores = logits.float().clone()
    if not config.allow_raw_special_tokens:
        scores[:, list(DEFAULT_SUPPRESSED_TOKEN_IDS)] = -torch.inf
    scores /= config.temperature
    if config.top_k is not None:
        threshold = torch.topk(scores, config.top_k, dim=-1).values[:, -1:]
        scores.masked_fill_(scores < threshold, -torch.inf)
    if config.top_p is not None and config.top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        probabilities = torch.softmax(sorted_scores, dim=-1)
        cumulative = probabilities.cumsum(dim=-1)
        remove = cumulative - probabilities >= config.top_p
        sorted_scores.masked_fill_(remove, -torch.inf)
        scores = torch.full_like(scores, -torch.inf).scatter(
            dim=-1,
            index=sorted_indices,
            src=sorted_scores,
        )
    if torch.any(torch.isneginf(scores).all(dim=-1)):
        raise GenerationError("sampling filters removed every token")
    return scores


def _select_next_tokens(
    logits: Tensor,
    config: CausalGenerationConfig,
    generator: torch.Generator,
) -> Tensor:
    scores = _filtered_logits(logits, config)
    if config.strategy == "greedy":
        return scores.argmax(dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    ).squeeze(-1)


def generate_causal(
    model: Olmo2ForCausalLM,
    prompts: Sequence[Sequence[int]],
    config: CausalGenerationConfig,
    *,
    on_token: Callable[[CausalGenerationEvent], None] | None = None,
) -> CausalGenerationResult:
    """Generate batched continuations and emit each step through ``on_token``."""

    _validate_prompts(prompts, model, config)
    device = next(model.parameters()).device
    input_ids, attention_mask = _left_padded_batch(
        prompts,
        pad_token_id=model.config.pad_token_id,
        device=device,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    generated: list[list[int]] = [[] for _ in prompts]
    finished = torch.zeros(len(prompts), dtype=torch.bool, device=device)
    model_forwards = 0
    cache = None
    was_training = model.training
    started = perf_counter()
    try:
        model.eval()
        with torch.no_grad():
            for step in range(1, config.max_new_tokens + 1):
                output = model(
                    input_ids,
                    attention_mask=attention_mask,
                    cache=cache,
                    use_cache=True,
                )
                model_forwards += 1
                if output.cache is None:
                    raise GenerationError("causal model did not return a KV cache")
                cache = output.cache
                active = ~finished
                selected = _select_next_tokens(
                    output.logits[:, -1, :],
                    config,
                    generator,
                )
                event_tokens: list[int | None] = []
                for row, is_active in enumerate(active.tolist()):
                    if not is_active:
                        event_tokens.append(None)
                        continue
                    token_id = int(selected[row].item())
                    generated[row].append(token_id)
                    event_tokens.append(token_id)
                    if token_id == model.config.eos_token_id:
                        finished[row] = True
                event = CausalGenerationEvent(
                    step=step,
                    token_ids=tuple(event_tokens),
                    finished=tuple(finished.tolist()),
                )
                if on_token is not None:
                    on_token(event)
                if bool(torch.all(finished)) or step == config.max_new_tokens:
                    break
                input_ids = torch.where(
                    active,
                    selected,
                    torch.full_like(selected, model.config.pad_token_id),
                ).unsqueeze(1)
                attention_mask = torch.cat(
                    (attention_mask, active.to(dtype=torch.long).unsqueeze(1)),
                    dim=1,
                )
    finally:
        model.train(was_training)

    elapsed = perf_counter() - started
    generated_count = sum(map(len, generated))
    reasons: tuple[Literal["eos", "max_new_tokens"], ...] = tuple(
        "eos" if done else "max_new_tokens" for done in finished.tolist()
    )
    return CausalGenerationResult(
        prompt_token_counts=tuple(map(len, prompts)),
        generated_token_ids=tuple(tuple(sequence) for sequence in generated),
        stop_reasons=reasons,
        model_forwards=model_forwards,
        generated_token_count=generated_count,
        elapsed_seconds=elapsed,
        tokens_per_second=generated_count / elapsed,
    )
