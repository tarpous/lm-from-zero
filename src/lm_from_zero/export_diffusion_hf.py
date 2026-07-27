"""Validated self-contained Hugging Face export for masked diffusion models."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from transformers import AutoModelForMaskedLM

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
from lm_from_zero.generation.diffusion import (
    DiffusionGenerationConfig,
    generate_diffusion,
)
from lm_from_zero.hf_diffusion_config import LLaDAConfig
from lm_from_zero.hf_diffusion_model import LLaDAForMaskedDiffusion
from lm_from_zero.models import MaskedDiffusionConfig, MaskedDiffusionForMaskedLM
from lm_from_zero.tokenizer.bpe import BYTE_TOKEN_OFFSET, ByteBPE
from lm_from_zero.tokenizer.pipeline import load_training_manifest
from lm_from_zero.training import load_checkpoint_model, validate_checkpoint

SHA256_PATTERN = r"^[0-9a-f]{64}$"
Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]


class DiffusionHFExportManifest(BaseModel):
    """Canonical provenance and parity record for one diffusion export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-hugging-face-export"] = (
        "lm-from-zero-hugging-face-export"
    )
    format_version: Literal[1] = 1
    architecture: Literal["LLaDAForMaskedDiffusion"] = "LLaDAForMaskedDiffusion"
    source_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    source_checkpoint_manifest_sha256: Sha256
    model_config_sha256: Sha256
    tokenizer_sha256: Sha256
    parameter_count: Annotated[int, Field(gt=0)]
    transformers_version: str = Field(min_length=1)
    requires_trust_remote_code: Literal[True] = True
    mask_token_id: Literal[7] = 7
    fp32_atol: float = FP32_ATOL
    fp32_rtol: float = FP32_RTOL
    fp32_max_abs_error: Annotated[float, Field(ge=0)]
    fp32_loss_abs_error: Annotated[float, Field(ge=0)]
    deterministic_trajectory_matches: Literal[True] = True
    sampler_defaults: dict[str, Any]
    tensor_map: dict[str, str]
    artifacts: tuple[ExportArtifact, ...]

    @model_validator(mode="after")
    def validate_contract(self) -> DiffusionHFExportManifest:
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
            "hf_diffusion_config.py",
            "hf_diffusion_model.py",
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


def diffusion_tensor_name_map(
    config: MaskedDiffusionConfig,
) -> dict[str, str]:
    """Return the explicit complete internal-to-export tensor mapping."""

    mapping = {
        "embed_tokens.weight": "embed_tokens.weight",
        "norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    layer_suffixes = (
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    )
    for layer_index in range(config.num_hidden_layers):
        for suffix in layer_suffixes:
            name = f"layers.{layer_index}.{suffix}"
            mapping[name] = name
    return mapping


def _export_config(config: MaskedDiffusionConfig) -> LLaDAConfig:
    return LLaDAConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        max_position_embeddings=config.max_position_embeddings,
        rope_theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
        initializer_range=config.initializer_range,
        attention_dropout=config.attention_dropout,
        corruption_epsilon=config.corruption_epsilon,
        mask_token_id=config.mask_token_id,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        tie_word_embeddings=config.tie_word_embeddings,
        architectures=["LLaDAForMaskedDiffusion"],
        auto_map={
            "AutoConfig": "hf_diffusion_config.LLaDAConfig",
            "AutoModelForMaskedLM": ("hf_diffusion_model.LLaDAForMaskedDiffusion"),
        },
    )


def _mapped_export_model(
    source: MaskedDiffusionForMaskedLM,
) -> tuple[LLaDAForMaskedDiffusion, dict[str, str]]:
    mapping = diffusion_tensor_name_map(source.config)
    source_state = source.state_dict()
    if set(source_state) != set(mapping):
        missing = sorted(set(mapping) - set(source_state))
        unknown = sorted(set(source_state) - set(mapping))
        raise ExportError(
            f"internal tensor set mismatch; missing={missing}, unknown={unknown}"
        )
    target = LLaDAForMaskedDiffusion(_export_config(source.config))
    target_state = target.state_dict()
    derived_target_tensors = {"rotary_embedding.inv_freq"}
    expected_target = set(mapping.values()) | derived_target_tensors
    if set(target_state) != expected_target:
        missing = sorted(expected_target - set(target_state))
        unknown = sorted(set(target_state) - expected_target)
        raise ExportError(
            f"export tensor set mismatch; missing={missing}, unknown={unknown}"
        )
    converted: dict[str, torch.Tensor] = {}
    for source_name, target_name in mapping.items():
        tensor = source_state[source_name]
        if tensor.shape != target_state[target_name].shape:
            raise ExportError(
                f"tensor shape mismatch for {target_name}: "
                f"{tuple(tensor.shape)} != {tuple(target_state[target_name].shape)}"
            )
        converted[target_name] = tensor
    converted["rotary_embedding.inv_freq"] = source.rotary_embedding.inv_freq
    incompatible = target.load_state_dict(converted, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ExportError("strict diffusion export tensor loading failed")
    target.eval()  # type: ignore[no-untyped-call]
    return target, mapping


def _verify_fp32_loss_and_trajectory(
    source: MaskedDiffusionForMaskedLM,
    target: LLaDAForMaskedDiffusion,
) -> tuple[float, float]:
    ordinary = source.config.vocab_size - BYTE_TOKEN_OFFSET
    input_ids = torch.tensor(
        [
            [
                source.config.bos_token_id,
                BYTE_TOKEN_OFFSET,
                BYTE_TOKEN_OFFSET + (1 % ordinary),
                source.config.eos_token_id,
            ]
        ],
        dtype=torch.long,
    )
    corrupted = input_ids.clone()
    corrupted[0, 1:3] = source.config.mask_token_id
    labels = torch.full_like(input_ids, -100)
    labels[0, 1:3] = input_ids[0, 1:3]
    eligible = input_ids != source.config.bos_token_id
    time = torch.tensor([0.5], dtype=torch.float32)
    source.eval().float()
    target.eval().float()  # type: ignore[no-untyped-call]
    with torch.no_grad():
        source_output = source(
            corrupted,
            labels=labels,
            eligible_mask=eligible,
            time=time,
        )
        target_output = target(
            input_ids=corrupted,
            labels=labels,
            eligible_mask=eligible,
            time=time,
        )
    maximum_error = float(
        (source_output.logits - target_output.logits).abs().max().item()
    )
    if not torch.allclose(
        source_output.logits,
        target_output.logits,
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    ):
        raise ExportError(
            "internal and exported fp32 logits exceed tolerance: "
            f"maximum absolute error {maximum_error}"
        )
    assert source_output.loss is not None
    assert target_output.loss is not None
    loss_error = float((source_output.loss - target_output.loss).abs().item())
    if not torch.allclose(
        source_output.loss,
        target_output.loss,
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    ):
        raise ExportError(
            "internal and exported diffusion losses exceed tolerance: "
            f"absolute error {loss_error}"
        )
    response_length = min(3, source.config.max_position_embeddings - 2)
    if response_length <= 0:
        raise ExportError("model context is too short for trajectory parity")
    generation_config = DiffusionGenerationConfig(
        response_length=response_length,
        diffusion_steps=response_length,
    )
    prompt = [[source.config.bos_token_id, BYTE_TOKEN_OFFSET]]
    internal_result = generate_diffusion(source, prompt, generation_config)
    exported_result = generate_diffusion(
        cast(MaskedDiffusionForMaskedLM, target),
        prompt,
        generation_config,
    )
    if (
        internal_result.generated_token_ids != exported_result.generated_token_ids
        or internal_result.stop_reasons != exported_result.stop_reasons
    ):
        raise ExportError("deterministic internal/export trajectory mismatch")
    return maximum_error, loss_error


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _write_manifest(path: Path, manifest: DiffusionHFExportManifest) -> None:
    with path.open("xb") as handle:
        handle.write(manifest.canonical_bytes())
        handle.flush()
        os.fsync(handle.fileno())


def export_diffusion_to_hugging_face(
    checkpoint_directory: str | Path,
    tokenizer_training_manifest: str | Path,
    output_directory: str | Path,
) -> DiffusionHFExportManifest:
    """Validate, parity-check, and atomically publish a diffusion export."""

    checkpoint_path = Path(checkpoint_directory)
    checkpoint = validate_checkpoint(checkpoint_path)
    if checkpoint.binding.architecture != "masked_diffusion":
        raise ExportError(
            "diffusion Hugging Face export requires a masked-diffusion checkpoint"
        )
    try:
        model_config = MaskedDiffusionConfig.model_validate(
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

    source_model = MaskedDiffusionForMaskedLM(model_config)
    load_checkpoint_model(
        checkpoint_path,
        model=source_model,
        expected_binding=checkpoint.binding,
    )
    exported_model, tensor_map = _mapped_export_model(source_model)
    logits_error, loss_error = _verify_fp32_loss_and_trajectory(
        source_model,
        exported_model,
    )
    exported_tokenizer = _hugging_face_tokenizer(
        tokenizer,
        model_max_length=model_config.max_position_embeddings,
    )
    sampler_defaults = {
        "diffusion_steps": 64,
        "response_length": 64,
        "reveal_schedule": "linear",
        "remask_fraction": 0.0,
        "remask_strategy": "none",
        "strategy": "greedy",
        "temperature": 1.0,
    }

    destination = Path(output_directory)
    if destination.exists():
        raise ExportError(f"export destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        exported_model.save_pretrained(temporary, safe_serialization=True)
        exported_tokenizer.save_pretrained(temporary)
        _write_special_tokens_map(
            temporary / "special_tokens_map.json",
            exported_tokenizer,
        )
        shutil.copyfile(
            Path(__file__).with_name("hf_diffusion_config.py"),
            temporary / "hf_diffusion_config.py",
        )
        shutil.copyfile(
            Path(__file__).with_name("hf_diffusion_model.py"),
            temporary / "hf_diffusion_model.py",
        )
        _write_json(
            temporary / "generation_config.json",
            {
                "_from_model_config": False,
                "bos_token_id": model_config.bos_token_id,
                "eos_token_id": model_config.eos_token_id,
                "mask_token_id": model_config.mask_token_id,
                "pad_token_id": model_config.pad_token_id,
                **sampler_defaults,
            },
        )
        reloaded_model = AutoModelForMaskedLM.from_pretrained(
            temporary,
            local_files_only=True,
            trust_remote_code=True,
        )
        logits_error, loss_error = _verify_fp32_loss_and_trajectory(
            source_model,
            cast(LLaDAForMaskedDiffusion, reloaded_model),
        )
        paths = sorted(
            path
            for path in temporary.iterdir()
            if path.is_file() and path.name != EXPORT_MANIFEST_FILENAME
        )
        artifacts = tuple(_artifact(path) for path in paths)
        from transformers import __version__ as transformers_version

        manifest = DiffusionHFExportManifest(
            source_checkpoint_id=checkpoint.lineage.checkpoint_id,
            source_checkpoint_manifest_sha256=sha256(
                checkpoint.canonical_bytes()
            ).hexdigest(),
            model_config_sha256=model_config.config_hash,
            tokenizer_sha256=tokenizer.model_hash,
            parameter_count=source_model.trainable_parameter_count(),
            transformers_version=transformers_version,
            fp32_max_abs_error=logits_error,
            fp32_loss_abs_error=loss_error,
            sampler_defaults=sampler_defaults,
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


def load_diffusion_export_manifest(
    path: str | Path,
) -> DiffusionHFExportManifest:
    """Load a canonical diffusion export manifest and verify every artifact."""

    manifest_path = Path(path)
    try:
        raw = manifest_path.read_bytes()
        manifest = DiffusionHFExportManifest.model_validate_json(raw)
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
