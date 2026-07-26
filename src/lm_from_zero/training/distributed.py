"""Minimal deterministic torch.distributed lifecycle and collectives."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.distributed as dist
from torch import Tensor

DeviceKind = Literal["cpu", "cuda"]
Reduction = Literal["mean", "max"]


class DistributedError(RuntimeError):
    """Raised when the launched process topology is invalid or inconsistent."""


@dataclass(frozen=True, slots=True)
class DistributedContext:
    """One initialized process topology, or the single-process fallback."""

    rank: int
    world_size: int
    local_rank: int
    backend: str | None
    initialized_here: bool = False

    @classmethod
    def current(cls) -> DistributedContext:
        """Describe the active process group or return the local fallback."""

        if not dist.is_available() or not dist.is_initialized():
            return cls(rank=0, world_size=1, local_rank=0, backend=None)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        raw_local_rank = os.environ.get("LOCAL_RANK")
        local_rank = rank if raw_local_rank is None else int(raw_local_rank)
        return cls(
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            backend=str(dist.get_backend()),
        )

    @property
    def enabled(self) -> bool:
        """Return whether collectives span more than one process."""

        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        """Return whether this process owns logs and checkpoint publication."""

        return self.rank == 0

    def validate_topology(self, *, rank: int, world_size: int) -> None:
        """Require a configuration to match the active process group."""

        if rank != self.rank or world_size != self.world_size:
            raise DistributedError(
                "training rank/world size does not match the process group"
            )

    def torch_device(self, device: DeviceKind) -> torch.device:
        """Resolve the process-local training device."""

        if device == "cuda" and self.enabled:
            return torch.device("cuda", self.local_rank)
        return torch.device(device)

    def reduce_float(
        self,
        value: float,
        *,
        reduction: Reduction,
        device: torch.device,
    ) -> float:
        """Reduce one measured scalar without retaining autograd state."""

        if not self.enabled:
            return value
        tensor = torch.tensor(value, dtype=torch.float64, device=device)
        operation = dist.ReduceOp.SUM if reduction == "mean" else dist.ReduceOp.MAX
        dist.all_reduce(tensor, op=operation)
        if reduction == "mean":
            tensor /= self.world_size
        return float(tensor)

    def reduce_tensor_mean(self, tensor: Tensor) -> Tensor:
        """Return the detached arithmetic mean of one tensor across ranks."""

        result = tensor.detach().clone()
        if self.enabled:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
            result /= self.world_size
        return result

    def all_gather_object(self, value: Any) -> tuple[Any, ...]:
        """Gather trusted local process state in rank order."""

        if not self.enabled:
            return (value,)
        gathered: list[Any] = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return tuple(gathered)

    def broadcast_primary_object(self, value: Any) -> Any:
        """Broadcast one trusted control value from rank zero."""

        if not self.enabled:
            return value
        payload = [value if self.is_primary else None]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]


def _environment_world_size() -> int:
    raw = os.environ.get("WORLD_SIZE")
    if raw is None:
        return 1
    try:
        world_size = int(raw)
    except ValueError as error:
        raise DistributedError("WORLD_SIZE must be an integer") from error
    if world_size < 1:
        raise DistributedError("WORLD_SIZE must be positive")
    return world_size


@contextmanager
def distributed_session(device: str) -> Iterator[DistributedContext]:
    """Initialize a torchrun process group when WORLD_SIZE is greater than one."""

    if device not in {"cpu", "cuda"}:
        raise DistributedError("distributed device must be 'cpu' or 'cuda'")
    device_kind = cast(DeviceKind, device)
    if dist.is_available() and dist.is_initialized():
        current = DistributedContext.current()
        if device_kind == "cuda":
            torch.cuda.set_device(current.local_rank)
        yield current
        return

    world_size = _environment_world_size()
    if world_size == 1:
        yield DistributedContext.current()
        return
    required = ("RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise DistributedError(
            f"distributed launch environment is incomplete: {', '.join(missing)}"
        )
    if not dist.is_available():
        raise DistributedError("torch.distributed is unavailable")
    local_rank = int(os.environ["LOCAL_RANK"])
    if device_kind == "cuda":
        if not torch.cuda.is_available():
            raise DistributedError("CUDA DDP was requested but CUDA is unavailable")
        torch.cuda.set_device(local_rank)
    backend = "nccl" if device_kind == "cuda" else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    current = DistributedContext(
        rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        local_rank=local_rank,
        backend=str(dist.get_backend()),
        initialized_here=True,
    )
    try:
        yield current
    finally:
        if current.initialized_here and dist.is_initialized():
            dist.destroy_process_group()
