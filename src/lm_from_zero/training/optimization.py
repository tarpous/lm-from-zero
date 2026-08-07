"""Deterministic optimizer grouping and warmup-cosine scheduling."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Self, cast

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
OptimizerVariant = Literal["adamw", "hybrid_muon"]


@dataclass(frozen=True, slots=True)
class ParameterPartition:
    """Auditable decay/no-decay assignment for every trainable parameter."""

    decay_names: tuple[str, ...]
    no_decay_names: tuple[str, ...]
    decay_parameters: tuple[nn.Parameter, ...]
    no_decay_parameters: tuple[nn.Parameter, ...]


@dataclass(frozen=True, slots=True)
class HybridMuonPartition:
    """Stable Muon/AdamW assignment for a dense research variant."""

    muon_names: tuple[str, ...]
    muon_parameters: tuple[nn.Parameter, ...]
    adamw_decay_names: tuple[str, ...]
    adamw_decay_parameters: tuple[nn.Parameter, ...]
    adamw_no_decay_names: tuple[str, ...]
    adamw_no_decay_parameters: tuple[nn.Parameter, ...]

    @property
    def all_names(self) -> tuple[str, ...]:
        return self.muon_names + self.adamw_decay_names + self.adamw_no_decay_names

    @property
    def all_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            self.muon_parameters
            + self.adamw_decay_parameters
            + self.adamw_no_decay_parameters
        )


def partition_hybrid_muon(model: nn.Module) -> HybridMuonPartition:
    """Partition hidden branch matrices to Muon and the rest to AdamW.

    Muon is intentionally restricted to the hidden attention and MLP
    projections. Embeddings, output projections, normalization vectors, and
    biases remain on AdamW as recommended by the optimizer's reference recipe.
    """

    modules = dict(model.named_modules())
    muon_names: list[str] = []
    muon_parameters: list[nn.Parameter] = []
    decay_names: list[str] = []
    decay_parameters: list[nn.Parameter] = []
    no_decay_names: list[str] = []
    no_decay_parameters: list[nn.Parameter] = []
    eligible_suffixes = (
        ".self_attn.q_proj.weight",
        ".self_attn.k_proj.weight",
        ".self_attn.v_proj.weight",
        ".self_attn.o_proj.weight",
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if (
            parameter.ndim == 2
            and name.startswith("layers.")
            and name.endswith(eligible_suffixes)
        ):
            muon_names.append(name)
            muon_parameters.append(parameter)
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
    assigned = muon_parameters + decay_parameters + no_decay_parameters
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if len({id(parameter) for parameter in assigned}) != len(assigned):
        raise ValueError("a trainable parameter was assigned more than once")
    if {id(parameter) for parameter in assigned} != {
        id(parameter) for parameter in trainable
    }:
        raise ValueError(
            "hybrid Muon partition does not cover every trainable parameter"
        )
    if not muon_parameters:
        raise ValueError("hybrid Muon requires at least one eligible hidden matrix")
    return HybridMuonPartition(
        muon_names=tuple(muon_names),
        muon_parameters=tuple(muon_parameters),
        adamw_decay_names=tuple(decay_names),
        adamw_decay_parameters=tuple(decay_parameters),
        adamw_no_decay_names=tuple(no_decay_names),
        adamw_no_decay_parameters=tuple(no_decay_parameters),
    )


class HybridMuonAdamW(torch.optim.Optimizer):
    """Checkpointable optimizer that updates Muon and AdamW groups together."""

    def __init__(
        self,
        muon: torch.optim.Optimizer,
        adamw: torch.optim.AdamW,
        groups: list[dict[str, Any]],
    ) -> None:
        # The wrapper's groups are the authoritative scheduler-facing groups;
        # the child optimizers own their algorithm-specific state.
        super().__init__(groups, defaults={})
        self._muon = muon
        self._adamw = adamw
        self._group_names = tuple(str(group["group_name"]) for group in groups)

    def _sync_child_learning_rates(self) -> None:
        rates = {
            str(group["group_name"]): float(group["lr"]) for group in self.param_groups
        }
        for group in self._muon.param_groups:
            group["lr"] = rates["muon"]
        for group in self._adamw.param_groups:
            group["lr"] = rates[str(group["group_name"])]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self._muon.zero_grad(set_to_none=set_to_none)
        self._adamw.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        self._sync_child_learning_rates()
        loss = self._muon.step(closure=closure)
        adamw_loss = self._adamw.step()
        return loss if loss is not None else adamw_loss

    def state_dict(self) -> dict[str, Any]:
        """Return the standard project checkpoint state layout.

        The wrapper keeps algorithm-specific state in two child optimizers but
        remaps it to the wrapper's stable parameter-group IDs. This lets the
        existing restricted checkpoint validator treat Muon exactly like
        AdamW and preserves resume compatibility across process ranks.
        """

        wrapper_state = super().state_dict()
        state = cast(dict[int, Any], wrapper_state["state"])
        for child in (self._muon, self._adamw):
            child_state = child.state_dict()
            child_ids_by_parameter: dict[int, int] = {}
            for child_group, child_group_state in zip(
                child.param_groups,
                cast(list[dict[str, Any]], child_state["param_groups"]),
                strict=True,
            ):
                for parameter, parameter_id in zip(
                    child_group["params"],
                    child_group_state["params"],
                    strict=True,
                ):
                    child_ids_by_parameter[id(parameter)] = int(parameter_id)
            child_values = cast(dict[int, Any], child_state["state"])
            for group, group_state in zip(
                self.param_groups,
                cast(list[dict[str, Any]], wrapper_state["param_groups"]),
                strict=True,
            ):
                for parameter, parameter_id in zip(
                    group["params"], group_state["params"], strict=True
                ):
                    child_id = child_ids_by_parameter.get(id(parameter))
                    if child_id is not None and child_id in child_values:
                        state[int(parameter_id)] = deepcopy(child_values[child_id])
        return wrapper_state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if set(state_dict) != {"state", "param_groups"}:
            raise ValueError("hybrid Muon optimizer state format is incompatible")
        saved_groups = state_dict["param_groups"]
        if not isinstance(saved_groups, list) or len(saved_groups) != len(
            self.param_groups
        ):
            raise ValueError("hybrid Muon optimizer groups are incompatible")
        saved_names = tuple(
            str(cast(dict[str, Any], group).get("group_name")) for group in saved_groups
        )
        if saved_names != self._group_names:
            raise ValueError("hybrid Muon optimizer groups are incompatible")
        super().load_state_dict(state_dict)
        saved_state = cast(dict[int, Any], state_dict["state"])
        for child in (self._muon, self._adamw):
            child_template = child.state_dict()
            child_ids_by_parameter: dict[int, int] = {}
            for child_group, child_group_state in zip(
                child.param_groups,
                cast(list[dict[str, Any]], child_template["param_groups"]),
                strict=True,
            ):
                for parameter, parameter_id in zip(
                    child_group["params"],
                    child_group_state["params"],
                    strict=True,
                ):
                    child_ids_by_parameter[id(parameter)] = int(parameter_id)
            child_state: dict[str, Any] = {
                "state": {},
                "param_groups": deepcopy(child_template["param_groups"]),
            }
            wrapper_ids_by_parameter: dict[int, int] = {}
            for group, group_state in zip(
                self.param_groups,
                cast(list[dict[str, Any]], saved_groups),
                strict=True,
            ):
                for parameter, parameter_id in zip(
                    group["params"], group_state["params"], strict=True
                ):
                    wrapper_ids_by_parameter[id(parameter)] = int(parameter_id)
            for group in child.param_groups:
                for parameter in group["params"]:
                    parameter_id = child_ids_by_parameter[id(parameter)]
                    wrapper_id = wrapper_ids_by_parameter[id(parameter)]
                    if wrapper_id in saved_state:
                        cast(dict[int, Any], child_state["state"])[parameter_id] = (
                            deepcopy(saved_state[wrapper_id])
                        )
            child.load_state_dict(child_state)
        self._sync_child_learning_rates()


def build_hybrid_muon(
    model: nn.Module,
    config: OptimizationConfig,
    *,
    device_type: Literal["cpu", "cuda"] = "cpu",
) -> tuple[HybridMuonAdamW, HybridMuonPartition]:
    """Build the bounded hybrid Muon/AdamW research optimizer."""

    muon_type = getattr(torch.optim, "Muon", None)
    if muon_type is None:
        raise RuntimeError("this PyTorch build does not provide torch.optim.Muon")
    partition = partition_hybrid_muon(model)
    muon_group = {
        "params": partition.muon_parameters,
        "weight_decay": config.weight_decay,
        "group_name": "muon",
    }
    adamw_groups = [
        {
            "params": partition.adamw_decay_parameters,
            "weight_decay": config.weight_decay,
            "group_name": "adamw_decay",
        },
        {
            "params": partition.adamw_no_decay_parameters,
            "weight_decay": 0.0,
            "group_name": "adamw_no_decay",
        },
    ]
    learning_rate = config.learning_rate_at(0)
    muon = muon_type(
        [muon_group],
        lr=learning_rate,
        weight_decay=config.weight_decay,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adjust_lr_fn="match_rms_adamw",
    )
    adamw = torch.optim.AdamW(
        adamw_groups,
        lr=learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.epsilon,
        foreach=None if device_type == "cpu" else True,
        fused=False,
    )
    groups = [
        {
            "params": group["params"],
            "lr": learning_rate,
            "weight_decay": group["weight_decay"],
            "group_name": group["group_name"],
        }
        for group in (muon_group, *adamw_groups)
    ]
    return HybridMuonAdamW(muon, adamw, groups), partition


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
