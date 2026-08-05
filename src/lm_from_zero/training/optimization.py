"""Deterministic optimizer grouping and warmup-cosine scheduling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn


class OptimizationConfig(BaseModel):
    """Pinned AdamW, clipping, and learning-rate policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    learning_rate: Annotated[float, Field(gt=0)] = 1e-3
    beta1: Annotated[float, Field(gt=0, lt=1)] = 0.9
    beta2: Annotated[float, Field(gt=0, lt=1)] = 0.95
    epsilon: Annotated[float, Field(gt=0)] = 1e-8
    weight_decay: Annotated[float, Field(ge=0)] = 0.1
    gradient_clip_norm: Annotated[float, Field(gt=0)] = 1.0
    total_steps: Annotated[int, Field(gt=1)]
    warmup_fraction: Annotated[float, Field(gt=0, lt=1)] = 0.015
    minimum_lr_ratio: Annotated[float, Field(gt=0, le=1)] = 0.1

    @model_validator(mode="after")
    def validate_warmup(self) -> Self:
        if self.warmup_steps >= self.total_steps:
            raise ValueError("warmup must leave at least one cosine-decay step")
        return self

    @property
    def warmup_steps(self) -> int:
        """Return the integral warmup length, rounded upward."""

        return max(1, math.ceil(self.total_steps * self.warmup_fraction))

    def lr_multiplier(self, step: int) -> float:
        """Return the learning-rate multiplier for a zero-based optimizer step."""

        if step < 0:
            raise ValueError("step cannot be negative")
        if step < self.warmup_steps:
            return (step + 1) / self.warmup_steps
        decay_steps = self.total_steps - self.warmup_steps
        progress = min(1.0, (step - self.warmup_steps) / max(1, decay_steps - 1))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.minimum_lr_ratio + (1.0 - self.minimum_lr_ratio) * cosine

    def learning_rate_at(self, step: int) -> float:
        """Return the absolute learning rate for a zero-based step."""

        return self.learning_rate * self.lr_multiplier(step)


AdamWBackend = Literal["auto", "foreach", "fused"]


@dataclass(frozen=True, slots=True)
class ParameterPartition:
    """Auditable decay/no-decay assignment for every trainable parameter."""

    decay_names: tuple[str, ...]
    no_decay_names: tuple[str, ...]
    decay_parameters: tuple[nn.Parameter, ...]
    no_decay_parameters: tuple[nn.Parameter, ...]


def partition_parameters(model: nn.Module) -> ParameterPartition:
    """Put matrices in decay except embeddings; vectors and biases do not decay."""

    modules = dict(model.named_modules())
    decay_names: list[str] = []
    no_decay_names: list[str] = []
    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        module_name, _, _ = name.rpartition(".")
        module = modules[module_name]
        no_decay = parameter.ndim < 2 or isinstance(module, nn.Embedding)
        if no_decay:
            no_decay_names.append(name)
            no_decay_parameters.append(parameter)
        else:
            decay_names.append(name)
            decay_parameters.append(parameter)

    assigned_ids = [
        id(parameter) for parameter in (*decay_parameters, *no_decay_parameters)
    ]
    trainable_ids = [
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    ]
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("a trainable parameter was assigned more than once")
    if set(assigned_ids) != set(trainable_ids):
        raise ValueError("optimizer partition does not cover every trainable parameter")
    return ParameterPartition(
        decay_names=tuple(decay_names),
        no_decay_names=tuple(no_decay_names),
        decay_parameters=tuple(decay_parameters),
        no_decay_parameters=tuple(no_decay_parameters),
    )


def build_adamw(
    model: nn.Module,
    config: OptimizationConfig,
    *,
    backend: AdamWBackend = "auto",
    device_type: Literal["cpu", "cuda"] = "cpu",
) -> tuple[torch.optim.AdamW, ParameterPartition]:
    """Build pinned AdamW groups and return their auditable partition."""

    if backend == "fused" and device_type != "cuda":
        raise ValueError("fused AdamW requires CUDA")

    partition = partition_parameters(model)
    groups: list[dict[str, Any]] = [
        {
            "params": partition.decay_parameters,
            "weight_decay": config.weight_decay,
            "group_name": "decay",
        },
        {
            "params": partition.no_decay_parameters,
            "weight_decay": 0.0,
            "group_name": "no_decay",
        },
    ]
    backend_arguments: dict[str, Any]
    if backend == "auto":
        backend_arguments = {"foreach": None, "fused": None}
    elif backend == "foreach":
        backend_arguments = {"foreach": True, "fused": False}
    else:
        backend_arguments = {"foreach": False, "fused": True}
    optimizer = torch.optim.AdamW(
        groups,
        lr=config.learning_rate_at(0),
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        **backend_arguments,
    )
    return optimizer, partition


def set_learning_rate(
    optimizer: torch.optim.Optimizer,
    config: OptimizationConfig,
    step: int,
) -> float:
    """Set every optimizer group to the deterministic schedule value."""

    learning_rate = config.learning_rate_at(step)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def clip_gradients(model: nn.Module, maximum_norm: float) -> Tensor:
    """Clip the global gradient norm and return its pre-clip value."""

    if maximum_norm <= 0:
        raise ValueError("maximum_norm must be positive")
    result: Tensor = nn.utils.clip_grad_norm_(model.parameters(), maximum_norm)
    return result
