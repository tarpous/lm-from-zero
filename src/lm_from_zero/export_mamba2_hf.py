"""Validated Hugging Face export for the project-owned grouped Mamba-2 model."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import GenerationConfig
from transformers import Mamba2Config as TransformersMamba2Config

from lm_from_zero.export_hf import (
    EXPORT_MANIFEST_FILENAME,
    FP32_ATOL,
    FP32_RTOL,
    ExportArtifact,
    ExportError,
    _artifact,
    _fsync_file,
    _hugging_face_tokenizer,
    _publish_directory,
    _write_special_tokens_map,
)
from lm_from_zero.hf_mamba2_compat import GroupedMamba2ForCausalLM
from lm_from_zero.models import Mamba2Config, Mamba2ForCausalLM
from lm_from_zero.tokenizer.bpe import BYTE_TOKEN_OFFSET, ByteBPE
from lm_from_zero.tokenizer.pipeline import load_training_manifest
from lm_from_zero.training import load_checkpoint_model, validate_checkpoint

SHA256_PATTERN = r"^[0-9a-f]{64}$"
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]


class Mamba2HFExportManifest(BaseModel):
    """Canonical provenance and compatibility record for a Mamba-2 export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-hugging-face-export"] = (
        "lm-from-zero-hugging-face-export"
    )
    format_version: Literal[1] = 1
    architecture: Literal["Mamba2ForCausalLM"] = "Mamba2ForCausalLM"
    runtime_class: Literal["GroupedMamba2ForCausalLM"] = "GroupedMamba2ForCausalLM"
    requires_trust_remote_code: Literal[True] = True
    normalization: Literal["grouped_gated_rms_norm"] = "grouped_gated_rms_norm"
    native_unfused_transformers_parity: Literal[False] = False
    source_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    source_checkpoint_manifest_sha256: Sha256
    model_config_sha256: Sha256
    tokenizer_sha256: Sha256
    parameter_count: Annotated[int, Field(gt=0)]
    transformers_version: str = Field(min_length=1)
    fp32_atol: float = FP32_ATOL
    fp32_rtol: float = FP32_RTOL
    fp32_max_abs_error: Annotated[float, Field(ge=0)]
    cached_fp32_max_abs_error: Annotated[float, Field(ge=0)]
    tensor_map: dict[str, str]
    artifacts: tuple[ExportArtifact, ...]

    @model_validator(mode="after")
    def validate_contract(self) -> Mamba2HFExportManifest:
        if self.fp32_atol != FP32_ATOL or self.fp32_rtol != FP32_RTOL:
            raise ValueError("export fp32 tolerances do not match the contract")
        if len(self.tensor_map) != len(set(self.tensor_map.values())):
            raise ValueError("export tensor mapping contains duplicate destinations")
        names = [artifact.filename for artifact in self.artifacts]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("export artifacts must be unique and sorted")
        required = {
            "config.json",
            "generation_config.json",
            "hf_mamba2_compat.py",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        }
        missing = required.difference(names)
        if missing:
            raise ValueError(
                f"export artifacts are incomplete: {', '.join(sorted(missing))}"
            )
        return self

    def canonical_bytes(self) -> bytes:
        """Return the canonical JSON representation used on disk."""

        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


def mamba2_tensor_name_map(config: Mamba2Config) -> dict[str, str]:
    """Return the complete explicit internal-to-HF tensor mapping."""

    mapping = {
        "embed_tokens.weight": "backbone.embeddings.weight",
        "norm.weight": "backbone.norm_f.weight",
        "lm_head.weight": "lm_head.weight",
    }
    layer_suffixes = (
        "norm.weight",
        "mixer.dt_bias",
        "mixer.A_log",
        "mixer.D",
        "mixer.in_proj.weight",
        "mixer.conv1d.weight",
        "mixer.conv1d.bias",
        "mixer.norm.weight",
        "mixer.out_proj.weight",
    )
    for layer_index in range(config.num_hidden_layers):
        for suffix in layer_suffixes:
            source = f"layers.{layer_index}.{suffix}"
            mapping[source] = f"backbone.{source}"
    return mapping


def _transformers_config(config: Mamba2Config) -> TransformersMamba2Config:
    exported = TransformersMamba2Config(
        architectures=["GroupedMamba2ForCausalLM"],
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        state_size=config.state_size,
        num_hidden_layers=config.num_hidden_layers,
        expand=config.expand,
        conv_kernel=config.conv_kernel,
        n_groups=config.num_groups,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
        chunk_size=config.chunk_size,
        layer_norm_epsilon=config.rms_norm_eps,
        initializer_range=config.initializer_range,
        time_step_min=config.time_step_min,
        time_step_max=config.time_step_max,
        time_step_floor=config.time_step_floor,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        tie_word_embeddings=config.tie_word_embeddings,
        use_bias=config.use_bias,
        use_conv_bias=config.use_conv_bias,
        residual_in_fp32=config.residual_in_fp32,
        use_cache=config.use_cache,
    )
    exported.rms_norm_group_size = config.inner_size // config.num_groups
    return exported


def _mapped_transformers_model(
    source_model: Mamba2ForCausalLM,
) -> tuple[GroupedMamba2ForCausalLM, dict[str, str]]:
    mapping = mamba2_tensor_name_map(source_model.config)
    source_state = source_model.state_dict()
    expected_source = set(mapping)
    actual_source = set(source_state)
    if actual_source != expected_source:
        missing = sorted(expected_source - actual_source)
        unknown = sorted(actual_source - expected_source)
        raise ExportError(
            f"internal tensor set mismatch; missing={missing}, unknown={unknown}"
        )

    model = GroupedMamba2ForCausalLM(_transformers_config(source_model.config))
    model.eval()  # type: ignore[no-untyped-call]
    target_state = model.state_dict()
    expected_target = set(mapping.values())
    actual_target = set(target_state)
    if actual_target != expected_target:
        missing = sorted(expected_target - actual_target)
        unknown = sorted(actual_target - expected_target)
        raise ExportError(
            f"Hugging Face tensor set mismatch; missing={missing}, unknown={unknown}"
        )

    mapped = {target: source_state[source] for source, target in mapping.items()}
    for target, tensor in mapped.items():
        if tensor.shape != target_state[target].shape:
            raise ExportError(
                f"tensor shape mismatch for {target}: "
                f"{tuple(tensor.shape)} != {tuple(target_state[target].shape)}"
            )
    incompatible = model.load_state_dict(mapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ExportError("strict Hugging Face tensor loading failed")
    return model, mapping


def _verify_fp32_and_cache(
    source: Mamba2ForCausalLM,
    target: GroupedMamba2ForCausalLM,
) -> tuple[float, float]:
    sequence_length = min(4, source.config.max_position_embeddings - 1)
    ordinary_count = source.config.vocab_size - BYTE_TOKEN_OFFSET
    input_ids = (
        torch.arange(sequence_length, dtype=torch.long).unsqueeze(0) % ordinary_count
        + BYTE_TOKEN_OFFSET
    )
    input_ids[0, 0] = source.config.bos_token_id
    source.eval().float()
    target.eval().float()  # type: ignore[no-untyped-call]
    with torch.no_grad():
        source_output = source(input_ids, use_cache=True)
        target_output = target(input_ids=input_ids, use_cache=True)
    logits_error = float(
        (source_output.logits - target_output.logits).abs().max().item()
    )
    if not torch.allclose(
        source_output.logits,
        target_output.logits,
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    ):
        raise ExportError(
            "internal and Hugging Face fp32 logits exceed export tolerance: "
            f"maximum absolute error {logits_error}"
        )
    if source_output.cache is None or target_output.cache_params is None:
        raise ExportError("cache parity requires both runtimes to return state")
    next_token = source_output.logits[:, -1].argmax(dim=-1, keepdim=True)
    with torch.no_grad():
        source_next = source(
            next_token,
            cache=source_output.cache,
            use_cache=True,
        )
        target_next = target(
            input_ids=next_token,
            cache_params=target_output.cache_params,
            use_cache=True,
        )
    cache_error = float((source_next.logits - target_next.logits).abs().max().item())
    if not torch.allclose(
        source_next.logits,
        target_next.logits,
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    ):
        raise ExportError(
            "internal and Hugging Face cached fp32 logits exceed export tolerance: "
            f"maximum absolute error {cache_error}"
        )
    return logits_error, cache_error


def _write_manifest(path: Path, manifest: Mamba2HFExportManifest) -> None:
    with path.open("xb") as handle:
        handle.write(manifest.canonical_bytes())
        handle.flush()
        os.fsync(handle.fileno())


def export_mamba2_to_hugging_face(
    checkpoint_directory: str | Path,
    tokenizer_training_manifest: str | Path,
    output_directory: str | Path,
) -> Mamba2HFExportManifest:
    """Validate, convert, parity-check, and atomically publish a Mamba-2 export."""

    checkpoint_path = Path(checkpoint_directory)
    checkpoint = validate_checkpoint(checkpoint_path)
    if checkpoint.binding.architecture != "mamba2":
        raise ExportError("Mamba-2 Hugging Face export requires a Mamba-2 checkpoint")
    try:
        model_config = Mamba2Config.model_validate(
            checkpoint.binding.resolved_model_config
        )
    except ValueError as error:
        raise ExportError("checkpoint model configuration is invalid") from error
    if model_config.config_hash != checkpoint.binding.model_config_sha256:
        raise ExportError("checkpoint model configuration hash mismatch")
    if model_config.tokenizer_hash != checkpoint.binding.tokenizer_sha256:
        raise ExportError("checkpoint tokenizer binding is inconsistent")

    training_path = Path(tokenizer_training_manifest)
    training = load_training_manifest(training_path)
    if training.status != "complete":
        raise ExportError("tokenizer training must be complete before export")
    if training.tokenizer_hash != checkpoint.binding.tokenizer_sha256:
        raise ExportError("tokenizer manifest does not match the checkpoint")
    if training.realized_vocab_size != model_config.vocab_size:
        raise ExportError("tokenizer vocabulary does not match the model")
    tokenizer = ByteBPE.load(training_path.parent / training.tokenizer_file)
    if tokenizer.model_hash != training.tokenizer_hash:
        raise ExportError("tokenizer file hash does not match its manifest")

    source_model = Mamba2ForCausalLM(model_config)
    load_checkpoint_model(
        checkpoint_path,
        model=source_model,
        expected_binding=checkpoint.binding,
    )
    hugging_face_model, tensor_map = _mapped_transformers_model(source_model)
    logits_error, cache_error = _verify_fp32_and_cache(
        source_model,
        hugging_face_model,
    )
    hugging_face_tokenizer = _hugging_face_tokenizer(
        tokenizer,
        model_max_length=model_config.max_position_embeddings,
    )
    generation_config = GenerationConfig(  # type: ignore[no-untyped-call]
        bos_token_id=model_config.bos_token_id,
        eos_token_id=model_config.eos_token_id,
        pad_token_id=model_config.pad_token_id,
        use_cache=model_config.use_cache,
        suppress_tokens=[0, 3, 4, 5, 7],
    )
    hugging_face_model.generation_config = generation_config
    GroupedMamba2ForCausalLM.register_for_auto_class(  # type: ignore[no-untyped-call]
        "AutoModelForCausalLM"
    )

    destination = Path(output_directory)
    if destination.exists():
        raise ExportError(f"export destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        hugging_face_model.save_pretrained(temporary, safe_serialization=True)
        hugging_face_tokenizer.save_pretrained(temporary)
        _write_special_tokens_map(
            temporary / "special_tokens_map.json",
            hugging_face_tokenizer,
        )
        generation_config.save_pretrained(temporary)
        paths = sorted(
            path
            for path in temporary.iterdir()
            if path.is_file() and path.name != EXPORT_MANIFEST_FILENAME
        )
        artifacts = tuple(_artifact(path) for path in paths)
        from transformers import __version__ as transformers_version

        manifest = Mamba2HFExportManifest(
            source_checkpoint_id=checkpoint.lineage.checkpoint_id,
            source_checkpoint_manifest_sha256=sha256(
                checkpoint.canonical_bytes()
            ).hexdigest(),
            model_config_sha256=model_config.config_hash,
            tokenizer_sha256=tokenizer.model_hash,
            parameter_count=source_model.trainable_parameter_count(),
            transformers_version=transformers_version,
            fp32_max_abs_error=logits_error,
            cached_fp32_max_abs_error=cache_error,
            tensor_map=tensor_map,
            artifacts=artifacts,
        )
        for path in paths:
            _fsync_file(path)
        _write_manifest(temporary / EXPORT_MANIFEST_FILENAME, manifest)
        _publish_directory(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def load_mamba2_export_manifest(path: str | Path) -> Mamba2HFExportManifest:
    """Load a canonical Mamba-2 manifest and validate all recorded hashes."""

    manifest_path = Path(path)
    try:
        raw = manifest_path.read_bytes()
        manifest = Mamba2HFExportManifest.model_validate_json(raw)
    except (OSError, ValueError) as error:
        raise ExportError("export manifest is invalid") from error
    if raw != manifest.canonical_bytes():
        raise ExportError("export manifest is not canonical JSON")
    for artifact in manifest.artifacts:
        artifact_path = manifest_path.parent / artifact.filename
        if not artifact_path.is_file():
            raise ExportError(f"export artifact is missing: {artifact.filename}")
        if artifact_path.stat().st_size != artifact.size_bytes:
            raise ExportError(f"export artifact size mismatch: {artifact.filename}")
        if _artifact(artifact_path).sha256 != artifact.sha256:
            raise ExportError(f"export artifact hash mismatch: {artifact.filename}")
    return manifest
