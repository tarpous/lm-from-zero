"""Optional CUDA numerical oracle for the project-owned Mamba-2 SSD."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch.nn import functional as F

from lm_from_zero.models.mamba2 import ssd_chunked


class Mamba2OracleError(RuntimeError):
    """Raised when the optional Mamba-2 oracle cannot prove parity."""


class Mamba2OracleConfig(BaseModel):
    """Deterministic dimensions and tolerances for one oracle comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    batch_size: Annotated[int, Field(gt=0)] = 1
    sequence_length: Annotated[int, Field(gt=0)] = 257
    num_heads: Annotated[int, Field(gt=0)] = 12
    head_dim: Annotated[int, Field(gt=0)] = 64
    num_groups: Annotated[int, Field(gt=0)] = 4
    state_size: Annotated[int, Field(gt=0)] = 64
    chunk_size: Annotated[int, Field(gt=0)] = 128
    seed: int = 1337
    device: str = "cuda"
    atol: Annotated[float, Field(gt=0)] = 1e-4
    rtol: Annotated[float, Field(gt=0)] = 1e-4

    @model_validator(mode="after")
    def validate_groups_and_device(self) -> Mamba2OracleConfig:
        if self.num_heads % self.num_groups != 0:
            raise ValueError("num_heads must be divisible by num_groups")
        if self.device != "cpu" and not self.device.startswith("cuda"):
            raise ValueError("oracle device must be CPU or CUDA")
        return self


class Mamba2OracleResult(BaseModel):
    """Canonical evidence from project-owned versus optional-kernel SSD."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    format: Literal["lm-from-zero-mamba2-oracle"] = "lm-from-zero-mamba2-oracle"
    format_version: Literal[1] = 1
    recorded_at_utc: datetime
    oracle_package: Literal["mamba-ssm"] = "mamba-ssm"
    oracle_version: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    cuda_version: str | None
    device: str = Field(min_length=1)
    device_name: str = Field(min_length=1)
    config: Mamba2OracleConfig
    output_max_abs_error: Annotated[float, Field(ge=0)]
    state_max_abs_error: Annotated[float, Field(ge=0)]
    output_close: Literal[True]
    state_close: Literal[True]

    def canonical_bytes(self) -> bytes:
        """Return deterministic JSON bytes for the portable report."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


OracleScan = Callable[..., Any]


def _load_oracle_scan() -> OracleScan:
    try:
        module = import_module("mamba_ssm.ops.triton.ssd_combined")
        scan = module.mamba_chunk_scan_combined
    except (AttributeError, ImportError, OSError) as error:
        raise Mamba2OracleError(
            "mamba-ssm 2.3.2.post1 with working CUDA extensions is required"
        ) from error
    return cast(OracleScan, scan)


def _oracle_version() -> str:
    try:
        return version("mamba-ssm")
    except PackageNotFoundError as error:
        raise Mamba2OracleError("mamba-ssm package metadata is unavailable") from error


def verify_mamba2_oracle(
    config: Mamba2OracleConfig | None = None,
) -> Mamba2OracleResult:
    """Require project-owned chunked SSD parity with the optional CUDA oracle."""

    resolved = Mamba2OracleConfig() if config is None else config
    device = torch.device(resolved.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise Mamba2OracleError("CUDA is unavailable for the Mamba-2 oracle")
    scan = _load_oracle_scan()
    generator = torch.Generator(device=device).manual_seed(resolved.seed)
    shape = (
        resolved.batch_size,
        resolved.sequence_length,
        resolved.num_heads,
        resolved.head_dim,
    )
    x = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
    raw_dt = torch.randn(
        shape[:-1],
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    dt_bias = torch.randn(
        resolved.num_heads,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    negative_a = -torch.exp(
        torch.randn(
            resolved.num_heads,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
    )
    grouped_shape = (
        resolved.batch_size,
        resolved.sequence_length,
        resolved.num_groups,
        resolved.state_size,
    )
    b = torch.randn(
        grouped_shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    c = torch.randn(
        grouped_shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    d = torch.randn(
        resolved.num_heads,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    initial_state = torch.randn(
        resolved.batch_size,
        resolved.num_heads,
        resolved.head_dim,
        resolved.state_size,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    dt = F.softplus(raw_dt + dt_bias)
    project_output, project_state = ssd_chunked(
        x * dt.unsqueeze(-1),
        dt * negative_a,
        b,
        c,
        chunk_size=resolved.chunk_size,
        initial_state=initial_state,
    )
    project_output = project_output + d.view(1, 1, -1, 1) * x
    oracle_result = scan(
        x,
        raw_dt,
        negative_a,
        b,
        c,
        chunk_size=resolved.chunk_size,
        D=d,
        dt_bias=dt_bias,
        dt_softplus=True,
        initial_states=initial_state,
        return_final_states=True,
    )
    if not isinstance(oracle_result, tuple) or len(oracle_result) != 2:
        raise Mamba2OracleError("oracle did not return output and final state")
    oracle_output, oracle_state = oracle_result
    if not isinstance(oracle_output, torch.Tensor) or not isinstance(
        oracle_state, torch.Tensor
    ):
        raise Mamba2OracleError("oracle returned non-tensor results")
    output_error = float((project_output - oracle_output).abs().max().item())
    state_error = float((project_state - oracle_state).abs().max().item())
    output_close = torch.allclose(
        project_output,
        oracle_output,
        atol=resolved.atol,
        rtol=resolved.rtol,
    )
    state_close = torch.allclose(
        project_state,
        oracle_state,
        atol=resolved.atol,
        rtol=resolved.rtol,
    )
    if not output_close or not state_close:
        raise Mamba2OracleError(
            "project-owned SSD exceeds oracle tolerance: "
            f"output_max_abs_error={output_error}, "
            f"state_max_abs_error={state_error}"
        )
    device_name = (
        torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else "CPU reference injection"
    )
    return Mamba2OracleResult(
        recorded_at_utc=datetime.now(UTC),
        oracle_version=_oracle_version(),
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        device=str(device),
        device_name=device_name,
        config=resolved,
        output_max_abs_error=output_error,
        state_max_abs_error=state_error,
        output_close=True,
        state_close=True,
    )


def write_mamba2_oracle_report(
    path: str | Path,
    result: Mamba2OracleResult,
) -> None:
    """Atomically write a canonical Mamba-2 oracle report."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists():
        raise Mamba2OracleError("incomplete Mamba-2 oracle report exists")
    try:
        with temporary.open("xb") as handle:
            handle.write(result.canonical_bytes())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
