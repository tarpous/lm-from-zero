"""Runtime observations used by Milestone 6A calibration.

The helpers in this module intentionally keep profiling outside the measured
training window.  They return small deterministic values that can be recorded
in canonical calibration artifacts without retaining profiler internals.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass
from importlib import import_module
from typing import Literal, cast

import torch

ObservedSdpaBackend = Literal["flash", "efficient", "math", "unknown"]
OperatorProfiler = Callable[[Callable[[], object]], Iterable[str]]


@dataclass(frozen=True)
class CompileGraphBreakSummary:
    """Stable summary of graph breaks observed since the last reset."""

    graph_breaks: int
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ProcessCpuTimes:
    """One paired monotonic-wall and process-CPU clock observation."""

    wall_seconds: float
    process_seconds: float


def _torch_graph_break_counter() -> MutableMapping[object, int]:
    dynamo_utils = import_module("torch._dynamo.utils")
    counters = cast(Mapping[str, object], dynamo_utils.counters)
    return cast(MutableMapping[object, int], counters["graph_break"])


def reset_compile_graph_break_counters(
    counter: MutableMapping[object, int] | None = None,
) -> None:
    """Reset only torch.compile's graph-break counter category.

    ``counter`` is injectable so the behavior remains covered by the offline
    CPU test suite without mutating process-global Torch state.
    """

    (counter if counter is not None else _torch_graph_break_counter()).clear()


def read_compile_graph_break_counters(
    counter: Mapping[object, int] | None = None,
) -> CompileGraphBreakSummary:
    """Read graph-break counts with unique reasons in deterministic order."""

    source = counter if counter is not None else _torch_graph_break_counter()
    normalized: dict[str, int] = {}
    for raw_reason, raw_count in source.items():
        count = int(raw_count)
        if count <= 0:
            continue
        reason = " ".join(str(raw_reason).split())
        normalized[reason] = normalized.get(reason, 0) + count
    return CompileGraphBreakSummary(
        graph_breaks=sum(normalized.values()),
        reasons=tuple(sorted(normalized)),
    )


def classify_native_sdpa_backend(
    operator_names: Iterable[str],
) -> ObservedSdpaBackend | None:
    """Classify the native SDPA implementation visible in profiler operators.

    ``None`` means no native scaled-dot-product-attention operator was seen.
    ``unknown`` means the generic SDPA wrapper was present without a recognized
    implementation, or more than one implementation appeared in the profile.
    """

    names = tuple(name.casefold() for name in operator_names)
    has_sdpa = any(
        "scaled_dot_product" in name or "scaled dot product" in name for name in names
    )
    observed: set[ObservedSdpaBackend] = set()
    for name in names:
        if "_scaled_dot_product_flash_attention" in name:
            observed.add("flash")
        if "_scaled_dot_product_efficient_attention" in name:
            observed.add("efficient")
        if "_scaled_dot_product_attention_math" in name:
            observed.add("math")
    if len(observed) == 1:
        return next(iter(observed))
    if observed or has_sdpa:
        return "unknown"
    return None


def _profile_cuda_forward_operator_names(
    forward: Callable[[], object],
) -> tuple[str, ...]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to profile an SDPA backend")
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        forward()
        torch.cuda.synchronize()
    return tuple(event.key for event in profile.key_averages())


def profile_cuda_sdpa_backend(
    forward: Callable[[], object],
    *,
    profiler: OperatorProfiler | None = None,
) -> ObservedSdpaBackend | None:
    """Profile exactly one CUDA forward and classify its native SDPA backend.

    Call this helper before the measured timing window.  ``profiler`` is an
    injection point for offline tests; production callers should omit it.
    """

    collect = profiler if profiler is not None else _profile_cuda_forward_operator_names
    return classify_native_sdpa_backend(collect(forward))


def capture_process_cpu_times(
    *,
    wall_clock: Callable[[], float] = time.perf_counter,
    process_clock: Callable[[], float] = time.process_time,
) -> ProcessCpuTimes:
    """Capture paired clocks for later process-utilization calculation."""

    return ProcessCpuTimes(
        wall_seconds=wall_clock(),
        process_seconds=process_clock(),
    )


def process_cpu_utilization(
    start: ProcessCpuTimes,
    end: ProcessCpuTimes,
) -> float:
    """Return process CPU seconds divided by elapsed wall seconds.

    A value of ``1.0`` represents one fully occupied CPU core.  The result can
    exceed ``1.0`` when the process executes work on multiple threads.
    """

    wall_delta = end.wall_seconds - start.wall_seconds
    process_delta = end.process_seconds - start.process_seconds
    if wall_delta <= 0:
        raise ValueError("wall-clock delta must be positive")
    if process_delta < 0:
        raise ValueError("process-time delta cannot be negative")
    return process_delta / wall_delta
