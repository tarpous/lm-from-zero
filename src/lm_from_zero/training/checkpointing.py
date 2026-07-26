"""Atomic, versioned training checkpoints with strict recovery validation."""

from __future__ import annotations

import copy
import json
import os
import platform
import random
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, Self, cast

import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    model_validator,
)
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import Tensor, nn
from torch.optim import Optimizer

from lm_from_zero.training.data import BatchCursor

CHECKPOINT_FORMAT = "lm-from-zero-training-checkpoint"
CHECKPOINT_FORMAT_VERSION = 1
MODEL_FILENAME = "model.safetensors"
RECOVERY_FILENAME = "recovery.pt"
MANIFEST_FILENAME = "manifest.json"
CHECKPOINT_ID_PATTERN = r"^step-[0-9]{12}$"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
_CHECKPOINT_DEPENDENCIES = ("numpy", "pydantic", "safetensors", "torch")

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]
Architecture = Literal["olmo2", "mamba2", "masked_diffusion"]


class CheckpointError(RuntimeError):
    """Raised when a checkpoint cannot be safely saved, validated, or restored."""


class Stateful(Protocol):
    """Minimal state interface used by optional gradient scalers."""

    def state_dict(self) -> dict[str, Any]:
        """Return serializable state."""

    def load_state_dict(self, state_dict: dict[str, Any]) -> object:
        """Restore serializable state."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _json_sha256(value: object) -> str:
    return sha256(_canonical_json_bytes(value)).hexdigest()


class ArtifactRecord(BaseModel):
    """Integrity metadata for one checkpoint payload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(min_length=1)
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256


class GitMetadata(BaseModel):
    """Source revision recorded at checkpoint creation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40,64}$")]
    dirty: bool


class RuntimeMetadata(BaseModel):
    """Dependency and hardware facts needed to audit recovery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    python_version: str = Field(min_length=1)
    operating_system: str = Field(min_length=1)
    machine: str = Field(min_length=1)
    torch_version: str = Field(min_length=1)
    cuda_version: str | None
    cuda_available: bool
    cuda_device_names: tuple[str, ...] = ()
    dependency_versions: dict[str, str]

    @model_validator(mode="after")
    def validate_dependency_versions(self) -> Self:
        missing = set(_CHECKPOINT_DEPENDENCIES) - self.dependency_versions.keys()
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"missing checkpoint dependency versions: {names}")
        return self


class CheckpointBinding(BaseModel):
    """Immutable model, data, process, and environment recovery binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture: Architecture
    resolved_model_config: dict[str, JsonValue]
    model_config_sha256: Sha256
    tokenizer_sha256: Sha256
    shard_manifest_sha256: Sha256
    rank: Annotated[int, Field(ge=0)]
    world_size: Annotated[int, Field(gt=0)]
    runtime: RuntimeMetadata
    git: GitMetadata

    @model_validator(mode="after")
    def validate_binding(self) -> Self:
        if self.rank >= self.world_size:
            raise ValueError("rank must be smaller than world_size")
        expected_hash = _json_sha256(self.resolved_model_config)
        if self.model_config_sha256 != expected_hash:
            raise ValueError("resolved model configuration hash does not match")
        return self


class CheckpointProgress(BaseModel):
    """Exact training progress associated with the serialized tensors."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    optimizer_step: Annotated[int, Field(ge=0)]
    scheduler_step: Annotated[int, Field(ge=0)]
    sequences_consumed: Annotated[int, Field(ge=0)]
    tokens_consumed: Annotated[int, Field(ge=0)]
    best_metric: float | None = None
    is_best: bool = False


class CheckpointLineage(BaseModel):
    """Parent relationship for a checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    checkpoint_id: Annotated[str, Field(pattern=CHECKPOINT_ID_PATTERN)]
    parent_checkpoint_id: (
        Annotated[str, Field(pattern=CHECKPOINT_ID_PATTERN)] | None
    ) = None
    parent_manifest_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def validate_parent_pair(self) -> Self:
        if (self.parent_checkpoint_id is None) != (self.parent_manifest_sha256 is None):
            raise ValueError("parent checkpoint ID and manifest hash must be paired")
        if self.parent_checkpoint_id == self.checkpoint_id:
            raise ValueError("a checkpoint cannot be its own parent")
        return self


class CheckpointManifest(BaseModel):
    """Canonical public checkpoint manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-training-checkpoint"] = (
        "lm-from-zero-training-checkpoint"
    )
    format_version: Literal[1] = 1
    created_at_utc: datetime
    binding: CheckpointBinding
    cursor: BatchCursor
    progress: CheckpointProgress
    lineage: CheckpointLineage
    model_artifact: ArtifactRecord
    recovery_artifact: ArtifactRecord
    recovery_loading: Literal["torch-load-weights-only"] = "torch-load-weights-only"

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.created_at_utc.tzinfo is None:
            raise ValueError("checkpoint timestamp must include a timezone")
        if self.model_artifact.filename != MODEL_FILENAME:
            raise ValueError("model artifact filename is unsupported")
        if self.recovery_artifact.filename != RECOVERY_FILENAME:
            raise ValueError("recovery artifact filename is unsupported")
        if self.cursor.build_manifest_sha256 != self.binding.shard_manifest_sha256:
            raise ValueError("cursor shard manifest does not match the binding")
        if self.cursor.tokenizer_hash != self.binding.tokenizer_sha256:
            raise ValueError("cursor tokenizer does not match the binding")
        if self.cursor.rank != self.binding.rank:
            raise ValueError("cursor rank does not match the binding")
        if self.cursor.world_size != self.binding.world_size:
            raise ValueError("cursor world size does not match the binding")
        if self.cursor.sequences_consumed != self.progress.sequences_consumed:
            raise ValueError("cursor and progress sequence counters disagree")
        if self.cursor.tokens_consumed != self.progress.tokens_consumed:
            raise ValueError("cursor and progress token counters disagree")
        expected_id = checkpoint_id(self.progress.optimizer_step)
        if self.lineage.checkpoint_id != expected_id:
            raise ValueError("checkpoint ID does not match the optimizer step")
        return self

    def canonical_bytes(self) -> bytes:
        """Return the canonical serialized manifest."""

        return _canonical_json_bytes(self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class RestoredCheckpoint:
    """Validated training state returned after a successful restore."""

    path: Path
    manifest: CheckpointManifest
    manifest_sha256: str
    scheduler_state: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ValidatedCheckpoint:
    path: Path
    manifest: CheckpointManifest
    manifest_sha256: str
    model_state: dict[str, Tensor]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    scaler_state: dict[str, Any] | None
    rng_state: dict[str, Any]


@dataclass(slots=True)
class CheckpointCadence:
    """Represent step and wall-clock triggers without duplicate-step saves."""

    last_saved_time_seconds: float
    step_interval: int = 250
    time_interval_seconds: float = 15 * 60
    last_saved_step: int | None = None

    def __post_init__(self) -> None:
        if self.last_saved_time_seconds < 0:
            raise ValueError("last saved time cannot be negative")
        if self.step_interval <= 0:
            raise ValueError("step interval must be positive")
        if self.time_interval_seconds <= 0:
            raise ValueError("time interval must be positive")
        if self.last_saved_step is not None and self.last_saved_step < 0:
            raise ValueError("last saved step cannot be negative")

    def due_reasons(
        self, optimizer_step: int, now_seconds: float
    ) -> frozenset[Literal["step", "time"]]:
        """Return the triggers due at this step, suppressing duplicate steps."""

        if optimizer_step < 0:
            raise ValueError("optimizer step cannot be negative")
        if now_seconds < self.last_saved_time_seconds:
            raise ValueError("checkpoint clock moved backwards")
        if optimizer_step == self.last_saved_step:
            return frozenset()
        reasons: set[Literal["step", "time"]] = set()
        if optimizer_step > 0 and optimizer_step % self.step_interval == 0:
            reasons.add("step")
        if now_seconds - self.last_saved_time_seconds >= self.time_interval_seconds:
            reasons.add("time")
        return frozenset(reasons)

    def mark_saved(self, optimizer_step: int, now_seconds: float) -> None:
        """Record one successful checkpoint publication."""

        if optimizer_step < 0:
            raise ValueError("optimizer step cannot be negative")
        if now_seconds < self.last_saved_time_seconds:
            raise ValueError("checkpoint clock moved backwards")
        self.last_saved_step = optimizer_step
        self.last_saved_time_seconds = now_seconds


def checkpoint_id(optimizer_step: int) -> str:
    """Return the stable directory ID for an optimizer step."""

    if not 0 <= optimizer_step <= 999_999_999_999:
        raise ValueError("optimizer step is outside the checkpoint ID range")
    return f"step-{optimizer_step:012d}"


def capture_runtime_metadata() -> RuntimeMetadata:
    """Capture dependency and hardware metadata without changing runtime state."""

    dependency_versions = {
        package: version(package) for package in _CHECKPOINT_DEPENDENCIES
    }
    cuda_available = torch.cuda.is_available()
    device_names = (
        tuple(
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        )
        if cuda_available
        else ()
    )
    return RuntimeMetadata(
        python_version=platform.python_version(),
        operating_system=f"{platform.system()} {platform.release()}",
        machine=platform.machine() or "unknown",
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        cuda_available=cuda_available,
        cuda_device_names=device_names,
        dependency_versions=dependency_versions,
    )


def capture_git_metadata(repository: str | Path) -> GitMetadata:
    """Capture the exact Git revision and dirty-state flag."""

    root = Path(repository)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise CheckpointError("could not capture Git checkpoint metadata") from error
    try:
        return GitMetadata(revision=revision, dirty=bool(status.strip()))
    except ValidationError as error:
        raise CheckpointError("Git returned an unsupported revision") from error


def create_checkpoint_binding(
    *,
    architecture: Architecture,
    resolved_model_config: Mapping[str, JsonValue],
    tokenizer_sha256: str,
    shard_manifest_sha256: str,
    rank: int,
    world_size: int,
    repository: str | Path,
) -> CheckpointBinding:
    """Build the complete immutable binding for a training run."""

    config = dict(resolved_model_config)
    return CheckpointBinding(
        architecture=architecture,
        resolved_model_config=config,
        model_config_sha256=_json_sha256(config),
        tokenizer_sha256=tokenizer_sha256,
        shard_manifest_sha256=shard_manifest_sha256,
        rank=rank,
        world_size=world_size,
        runtime=capture_runtime_metadata(),
        git=capture_git_metadata(repository),
    )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(path: Path) -> ArtifactRecord:
    return ArtifactRecord(
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=_file_sha256(path),
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_manifest(path: Path, manifest: CheckpointManifest) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(manifest.canonical_bytes())
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _capture_rng_state() -> dict[str, Any]:
    python_version, python_values, python_gauss = random.getstate()
    numpy_name, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached = (
        np.random.get_state()
    )
    numpy_keys_array = np.asarray(numpy_keys, dtype=np.uint32)
    return {
        "python": {
            "version": python_version,
            "values": list(python_values),
            "gauss": python_gauss,
        },
        "numpy": {
            "name": numpy_name,
            "keys": torch.from_numpy(numpy_keys_array.copy()),
            "position": numpy_position,
            "has_gauss": numpy_has_gauss,
            "cached_gaussian": numpy_cached,
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _as_dict(value: object, description: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise CheckpointError(f"{description} must be a string-keyed dictionary")
    return cast(dict[str, Any], value)


def _validate_rng_state(value: object) -> dict[str, Any]:
    state = _as_dict(value, "RNG state")
    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise CheckpointError("RNG state fields are missing or unsupported")

    python_state = _as_dict(state["python"], "Python RNG state")
    if set(python_state) != {"version", "values", "gauss"}:
        raise CheckpointError("Python RNG state fields are invalid")
    if not isinstance(python_state["version"], int):
        raise CheckpointError("Python RNG version is invalid")
    values = python_state["values"]
    if not isinstance(values, list) or not all(
        isinstance(item, int) for item in values
    ):
        raise CheckpointError("Python RNG values are invalid")
    if python_state["gauss"] is not None and not isinstance(
        python_state["gauss"], float
    ):
        raise CheckpointError("Python RNG Gaussian cache is invalid")

    numpy_state = _as_dict(state["numpy"], "NumPy RNG state")
    if set(numpy_state) != {
        "name",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise CheckpointError("NumPy RNG state fields are invalid")
    if not isinstance(numpy_state["name"], str):
        raise CheckpointError("NumPy RNG algorithm is invalid")
    numpy_keys = numpy_state["keys"]
    if (
        not isinstance(numpy_keys, Tensor)
        or numpy_keys.dtype != torch.uint32
        or numpy_keys.ndim != 1
    ):
        raise CheckpointError("NumPy RNG key tensor is invalid")
    if not isinstance(numpy_state["position"], int) or not isinstance(
        numpy_state["has_gauss"], int
    ):
        raise CheckpointError("NumPy RNG counters are invalid")
    if not isinstance(numpy_state["cached_gaussian"], float):
        raise CheckpointError("NumPy RNG Gaussian cache is invalid")

    torch_cpu = state["torch_cpu"]
    if (
        not isinstance(torch_cpu, Tensor)
        or torch_cpu.dtype != torch.uint8
        or torch_cpu.ndim != 1
    ):
        raise CheckpointError("Torch CPU RNG state is invalid")
    torch_cuda = state["torch_cuda"]
    if not isinstance(torch_cuda, list) or not all(
        isinstance(item, Tensor) and item.dtype == torch.uint8 and item.ndim == 1
        for item in torch_cuda
    ):
        raise CheckpointError("Torch CUDA RNG state is invalid")
    if torch.cuda.is_available() and len(torch_cuda) != torch.cuda.device_count():
        raise CheckpointError("Torch CUDA RNG device count does not match")
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    python_state = cast(dict[str, Any], state["python"])
    random.setstate(
        (
            cast(int, python_state["version"]),
            tuple(cast(list[int], python_state["values"])),
            cast(float | None, python_state["gauss"]),
        )
    )

    numpy_state = cast(dict[str, Any], state["numpy"])
    numpy_keys = cast(Tensor, numpy_state["keys"]).cpu().numpy().copy()
    np.random.set_state(
        (
            cast(str, numpy_state["name"]),
            numpy_keys,
            cast(int, numpy_state["position"]),
            cast(int, numpy_state["has_gauss"]),
            cast(float, numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(cast(Tensor, state["torch_cpu"]).cpu())
    cuda_states = cast(list[Tensor], state["torch_cuda"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])


def _validate_recovery_payload(
    payload: object,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any] | None,
    dict[str, Any],
]:
    recovery = _as_dict(payload, "recovery payload")
    expected_fields = {
        "format",
        "format_version",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "rng_state",
    }
    if set(recovery) != expected_fields:
        raise CheckpointError("recovery payload fields are missing or unsupported")
    if recovery["format"] != CHECKPOINT_FORMAT:
        raise CheckpointError("recovery payload format is unsupported")
    if recovery["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError("recovery payload version is unsupported")
    optimizer_state = _as_dict(recovery["optimizer_state"], "optimizer state")
    scheduler_state = _as_dict(recovery["scheduler_state"], "scheduler state")
    scaler_value = recovery["scaler_state"]
    scaler_state = (
        None if scaler_value is None else _as_dict(scaler_value, "scaler state")
    )
    rng_state = _validate_rng_state(recovery["rng_state"])
    return optimizer_state, scheduler_state, scaler_state, rng_state


def _validate_artifact(directory: Path, record: ArtifactRecord) -> Path:
    path = directory / record.filename
    if not path.is_file():
        raise CheckpointError(f"checkpoint artifact is missing: {record.filename}")
    if path.stat().st_size != record.size_bytes:
        raise CheckpointError(f"checkpoint artifact size mismatch: {record.filename}")
    if _file_sha256(path) != record.sha256:
        raise CheckpointError(f"checkpoint artifact hash mismatch: {record.filename}")
    return path


def _load_manifest(directory: Path) -> tuple[CheckpointManifest, str]:
    path = directory / MANIFEST_FILENAME
    if not path.is_file():
        raise CheckpointError("checkpoint manifest is missing")
    try:
        raw = path.read_bytes()
        manifest = CheckpointManifest.model_validate_json(raw)
    except (OSError, UnicodeDecodeError, ValidationError, ValueError) as error:
        raise CheckpointError("checkpoint manifest is invalid") from error
    if raw != manifest.canonical_bytes():
        raise CheckpointError("checkpoint manifest is not canonical JSON")
    return manifest, sha256(raw).hexdigest()


def _check_expected_binding(
    actual: CheckpointBinding, expected: CheckpointBinding
) -> None:
    checks = {
        "architecture": (actual.architecture, expected.architecture),
        "model configuration": (
            actual.model_config_sha256,
            expected.model_config_sha256,
        ),
        "tokenizer": (actual.tokenizer_sha256, expected.tokenizer_sha256),
        "shard manifest": (
            actual.shard_manifest_sha256,
            expected.shard_manifest_sha256,
        ),
        "rank": (actual.rank, expected.rank),
        "world size": (actual.world_size, expected.world_size),
        "dependency versions": (
            actual.runtime.dependency_versions,
            expected.runtime.dependency_versions,
        ),
    }
    mismatches = [name for name, values in checks.items() if values[0] != values[1]]
    if mismatches:
        raise CheckpointError(f"checkpoint binding mismatch: {', '.join(mismatches)}")


def _validate_model_state(model: nn.Module, saved: Mapping[str, Tensor]) -> None:
    live = model.state_dict()
    if saved.keys() != live.keys():
        missing = sorted(live.keys() - saved.keys())
        unexpected = sorted(saved.keys() - live.keys())
        raise CheckpointError(
            "model tensor names do not match; "
            f"missing={missing}, unexpected={unexpected}"
        )
    for name, live_tensor in live.items():
        saved_tensor = saved[name]
        if saved_tensor.shape != live_tensor.shape:
            raise CheckpointError(f"model tensor shape mismatch: {name}")
        if saved_tensor.dtype != live_tensor.dtype:
            raise CheckpointError(f"model tensor dtype mismatch: {name}")


def _validate_optimizer_state(
    optimizer: Optimizer, saved_state: Mapping[str, Any]
) -> None:
    if set(saved_state) != {"state", "param_groups"}:
        raise CheckpointError("optimizer state fields are invalid")
    saved_groups = saved_state["param_groups"]
    saved_values = saved_state["state"]
    if not isinstance(saved_groups, list) or not isinstance(saved_values, dict):
        raise CheckpointError("optimizer state structure is invalid")
    if len(saved_groups) != len(optimizer.param_groups):
        raise CheckpointError("optimizer parameter-group count does not match")

    known_ids: set[int] = set()
    parameters_by_id: dict[int, Tensor] = {}
    for saved_group_value, live_group in zip(
        saved_groups, optimizer.param_groups, strict=True
    ):
        if not isinstance(saved_group_value, dict):
            raise CheckpointError("optimizer parameter group is invalid")
        saved_parameters = saved_group_value.get("params")
        live_parameters = live_group["params"]
        if not isinstance(saved_parameters, list) or len(saved_parameters) != len(
            live_parameters
        ):
            raise CheckpointError("optimizer parameter-group layout does not match")
        for parameter_id, parameter in zip(
            saved_parameters, live_parameters, strict=True
        ):
            if not isinstance(parameter_id, int) or not isinstance(parameter, Tensor):
                raise CheckpointError("optimizer parameter reference is invalid")
            if parameter_id in known_ids:
                raise CheckpointError("optimizer parameter ID is duplicated")
            known_ids.add(parameter_id)
            parameters_by_id[parameter_id] = parameter

    if not all(isinstance(key, int) for key in saved_values):
        raise CheckpointError("optimizer state contains a non-integer parameter ID")
    if not set(cast(dict[int, Any], saved_values)).issubset(known_ids):
        raise CheckpointError("optimizer state references an unknown parameter")
    for parameter_id, state_value in cast(dict[int, Any], saved_values).items():
        if not isinstance(state_value, dict):
            raise CheckpointError("optimizer per-parameter state is invalid")
        parameter = parameters_by_id[parameter_id]
        for item in state_value.values():
            if (
                isinstance(item, Tensor)
                and item.numel() != 1
                and item.shape != parameter.shape
            ):
                raise CheckpointError("optimizer tensor shape does not match parameter")


def _validate_checkpoint(
    directory: Path,
    *,
    expected_binding: CheckpointBinding | None = None,
    model: nn.Module | None = None,
    optimizer: Optimizer | None = None,
    require_directory_name: bool = True,
) -> _ValidatedCheckpoint:
    if not directory.is_dir():
        raise CheckpointError("checkpoint directory does not exist")
    manifest, manifest_hash = _load_manifest(directory)
    if require_directory_name and directory.name != manifest.lineage.checkpoint_id:
        raise CheckpointError("checkpoint directory name does not match its manifest")
    if expected_binding is not None:
        _check_expected_binding(manifest.binding, expected_binding)

    model_path = _validate_artifact(directory, manifest.model_artifact)
    recovery_path = _validate_artifact(directory, manifest.recovery_artifact)
    try:
        model_state = load_safetensors(str(model_path), device="cpu")
    except Exception as error:
        raise CheckpointError("Safetensors model payload is invalid") from error
    try:
        recovery_payload = torch.load(
            recovery_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as error:
        raise CheckpointError("restricted recovery payload loading failed") from error
    optimizer_state, scheduler_state, scaler_state, rng_state = (
        _validate_recovery_payload(recovery_payload)
    )
    if model is not None:
        _validate_model_state(model, model_state)
    if optimizer is not None:
        _validate_optimizer_state(optimizer, optimizer_state)
    return _ValidatedCheckpoint(
        path=directory,
        manifest=manifest,
        manifest_sha256=manifest_hash,
        model_state=model_state,
        optimizer_state=optimizer_state,
        scheduler_state=scheduler_state,
        scaler_state=scaler_state,
        rng_state=rng_state,
    )


def validate_checkpoint(
    directory: str | Path,
    *,
    expected_binding: CheckpointBinding | None = None,
) -> CheckpointManifest:
    """Validate every checkpoint artifact without mutating training objects."""

    return _validate_checkpoint(
        Path(directory), expected_binding=expected_binding
    ).manifest


def _publish_directory(temporary: Path, destination: Path) -> None:
    os.replace(temporary, destination)


def save_checkpoint(
    root: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    cursor: BatchCursor,
    binding: CheckpointBinding,
    optimizer_step: int,
    scheduler_step: int,
    scheduler_state: Mapping[str, Any] | None = None,
    scaler: Stateful | None = None,
    best_metric: float | None = None,
    is_best: bool = False,
    parent_checkpoint: str | Path | None = None,
) -> Path:
    """Publish a complete checkpoint directory atomically."""

    identifier = checkpoint_id(optimizer_step)
    destination_root = Path(root)
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / identifier
    if destination.exists():
        existing = _validate_checkpoint(
            destination,
            expected_binding=binding,
            model=model,
            optimizer=optimizer,
        )
        if existing.manifest.progress.optimizer_step != optimizer_step:
            raise CheckpointError("existing checkpoint step does not match")
        return destination

    progress = CheckpointProgress(
        optimizer_step=optimizer_step,
        scheduler_step=scheduler_step,
        sequences_consumed=cursor.sequences_consumed,
        tokens_consumed=cursor.tokens_consumed,
        best_metric=best_metric,
        is_best=is_best,
    )
    parent_id: str | None = None
    parent_hash: str | None = None
    if parent_checkpoint is not None:
        parent = _validate_checkpoint(Path(parent_checkpoint), expected_binding=binding)
        parent_id = parent.manifest.lineage.checkpoint_id
        parent_hash = parent.manifest_sha256
        if parent.manifest.progress.optimizer_step >= optimizer_step:
            raise CheckpointError("parent checkpoint must precede the new checkpoint")
    lineage = CheckpointLineage(
        checkpoint_id=identifier,
        parent_checkpoint_id=parent_id,
        parent_manifest_sha256=parent_hash,
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{identifier}-", dir=destination_root))
    try:
        model_path = temporary / MODEL_FILENAME
        tensors = {
            name: tensor.detach().to(device="cpu").contiguous().clone()
            for name, tensor in model.state_dict().items()
        }
        save_safetensors(
            tensors,
            str(model_path),
            metadata={
                "format": CHECKPOINT_FORMAT,
                "format_version": str(CHECKPOINT_FORMAT_VERSION),
            },
        )
        _fsync_file(model_path)

        recovery_path = temporary / RECOVERY_FILENAME
        recovery_payload = {
            "format": CHECKPOINT_FORMAT,
            "format_version": CHECKPOINT_FORMAT_VERSION,
            "optimizer_state": copy.deepcopy(optimizer.state_dict()),
            "scheduler_state": dict(scheduler_state or {}),
            "scaler_state": (
                None if scaler is None else copy.deepcopy(scaler.state_dict())
            ),
            "rng_state": _capture_rng_state(),
        }
        torch.save(recovery_payload, recovery_path)
        _fsync_file(recovery_path)

        manifest = CheckpointManifest(
            created_at_utc=datetime.now(UTC),
            binding=binding,
            cursor=cursor,
            progress=progress,
            lineage=lineage,
            model_artifact=_artifact_record(model_path),
            recovery_artifact=_artifact_record(recovery_path),
        )
        _write_manifest(temporary / MANIFEST_FILENAME, manifest)
        _validate_checkpoint(
            temporary,
            expected_binding=binding,
            model=model,
            optimizer=optimizer,
            require_directory_name=False,
        )
        _publish_directory(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def restore_checkpoint(
    directory: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    expected_binding: CheckpointBinding,
    scaler: Stateful | None = None,
) -> RestoredCheckpoint:
    """Validate fully, then restore model, optimizer, scaler, and RNG state."""

    validated = _validate_checkpoint(
        Path(directory),
        expected_binding=expected_binding,
        model=model,
        optimizer=optimizer,
    )
    if (scaler is None) != (validated.scaler_state is None):
        raise CheckpointError("checkpoint scaler presence does not match the runtime")

    model.load_state_dict(validated.model_state, strict=True)
    optimizer.load_state_dict(validated.optimizer_state)
    if scaler is not None:
        scaler.load_state_dict(cast(dict[str, Any], validated.scaler_state))
    _restore_rng_state(validated.rng_state)
    return RestoredCheckpoint(
        path=validated.path,
        manifest=validated.manifest,
        manifest_sha256=validated.manifest_sha256,
        scheduler_state=copy.deepcopy(validated.scheduler_state),
    )


def load_checkpoint_model(
    directory: str | Path,
    *,
    model: nn.Module,
    expected_binding: CheckpointBinding,
) -> CheckpointManifest:
    """Validate the complete checkpoint, then load model tensors only."""

    validated = _validate_checkpoint(
        Path(directory),
        expected_binding=expected_binding,
        model=model,
    )
    model.load_state_dict(validated.model_state, strict=True)
    return validated.manifest


def apply_checkpoint_retention(
    root: str | Path,
    *,
    keep_latest: int = 3,
    best_checkpoint_id: str | None = None,
) -> tuple[Path, ...]:
    """Keep the latest valid checkpoints plus a distinct valid best checkpoint."""

    if keep_latest <= 0:
        raise ValueError("retention must keep at least one latest checkpoint")
    if best_checkpoint_id is not None:
        try:
            checkpoint_id(int(best_checkpoint_id.removeprefix("step-")))
        except ValueError as error:
            raise CheckpointError("best checkpoint ID is invalid") from error

    root_path = Path(root)
    valid: list[_ValidatedCheckpoint] = []
    if root_path.is_dir():
        for candidate in root_path.iterdir():
            if not candidate.is_dir() or not candidate.name.startswith("step-"):
                continue
            try:
                valid.append(_validate_checkpoint(candidate))
            except CheckpointError:
                continue
    valid.sort(key=lambda item: item.manifest.progress.optimizer_step)
    if not valid:
        return ()

    keep = {item.manifest.lineage.checkpoint_id for item in valid[-keep_latest:]}
    if best_checkpoint_id is None:
        best_candidates = [item for item in valid if item.manifest.progress.is_best]
        if best_candidates:
            keep.add(best_candidates[-1].manifest.lineage.checkpoint_id)
    else:
        valid_ids = {item.manifest.lineage.checkpoint_id for item in valid}
        if best_checkpoint_id not in valid_ids:
            raise CheckpointError("best checkpoint is not a valid recovery point")
        keep.add(best_checkpoint_id)

    for item in valid:
        if item.manifest.lineage.checkpoint_id not in keep:
            shutil.rmtree(item.path)
    retained = [item.path for item in valid if item.path.exists()]
    return tuple(retained)
