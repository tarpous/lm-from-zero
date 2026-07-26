"""Validated, atomic Hugging Face export for project-owned dense models."""

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
from tokenizers import (  # type: ignore[import-untyped]
    Tokenizer,
    decoders,
    models,
    pre_tokenizers,
)
from transformers import (
    GenerationConfig,
    PreTrainedTokenizerFast,
)
from transformers import (
    Olmo2Config as HuggingFaceOlmo2Config,
)
from transformers import (
    Olmo2ForCausalLM as HuggingFaceOlmo2ForCausalLM,
)

from lm_from_zero.models import Olmo2Config, Olmo2ForCausalLM
from lm_from_zero.tokenizer.bpe import (
    BYTE_LEVEL_SYMBOLS,
    BYTE_TOKEN_OFFSET,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    ByteBPE,
)
from lm_from_zero.tokenizer.pipeline import load_training_manifest
from lm_from_zero.training import load_checkpoint_model, validate_checkpoint

EXPORT_FORMAT = "lm-from-zero-hugging-face-export"
EXPORT_FORMAT_VERSION = 1
EXPORT_MANIFEST_FILENAME = "export_manifest.json"
SHA256_PATTERN = r"^[0-9a-f]{64}$"
FP32_ATOL = 1e-5
FP32_RTOL = 1e-5

Sha256 = Annotated[str, Field(pattern=SHA256_PATTERN)]


class ExportError(RuntimeError):
    """Raised when an export cannot satisfy its compatibility contract."""


class ExportArtifact(BaseModel):
    """Integrity metadata for one exported Hugging Face file."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str = Field(min_length=1)
    size_bytes: Annotated[int, Field(gt=0)]
    sha256: Sha256

    @model_validator(mode="after")
    def validate_filename(self) -> ExportArtifact:
        if Path(self.filename).name != self.filename:
            raise ValueError("export artifact filenames must be flat")
        if self.filename == EXPORT_MANIFEST_FILENAME:
            raise ValueError("the export manifest cannot record itself")
        return self


class DenseHFExportManifest(BaseModel):
    """Canonical provenance and parity record for one dense HF export."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["lm-from-zero-hugging-face-export"] = (
        "lm-from-zero-hugging-face-export"
    )
    format_version: Literal[1] = 1
    architecture: Literal["Olmo2ForCausalLM"] = "Olmo2ForCausalLM"
    source_checkpoint_id: str = Field(pattern=r"^step-[0-9]{12}$")
    source_checkpoint_manifest_sha256: Sha256
    model_config_sha256: Sha256
    tokenizer_sha256: Sha256
    parameter_count: Annotated[int, Field(gt=0)]
    transformers_version: str = Field(min_length=1)
    fp32_atol: float = FP32_ATOL
    fp32_rtol: float = FP32_RTOL
    fp32_max_abs_error: Annotated[float, Field(ge=0)]
    tensor_map: dict[str, str]
    artifacts: tuple[ExportArtifact, ...]

    @model_validator(mode="after")
    def validate_contract(self) -> DenseHFExportManifest:
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


def dense_tensor_name_map(config: Olmo2Config) -> dict[str, str]:
    """Return the complete explicit internal-to-HF tensor mapping."""

    mapping = {
        "embed_tokens.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    layer_suffixes = (
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "post_attention_layernorm.weight",
        "post_feedforward_layernorm.weight",
    )
    for layer_index in range(config.num_hidden_layers):
        for suffix in layer_suffixes:
            source = f"layers.{layer_index}.{suffix}"
            mapping[source] = f"model.{source}"
    return mapping


def _hugging_face_config(config: Olmo2Config) -> HuggingFaceOlmo2Config:
    return HuggingFaceOlmo2Config(
        architectures=[config.export_architecture],
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        hidden_act="silu",
        max_position_embeddings=config.max_position_embeddings,
        initializer_range=config.initializer_range,
        use_cache=config.use_cache,
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        tie_word_embeddings=config.tie_word_embeddings,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": config.rope_theta,
        },
        attention_bias=config.attention_bias,
        attention_dropout=config.attention_dropout,
        rms_norm_eps=config.rms_norm_eps,
    )


def _mapped_hugging_face_model(
    source_model: Olmo2ForCausalLM,
) -> tuple[HuggingFaceOlmo2ForCausalLM, dict[str, str]]:
    mapping = dense_tensor_name_map(source_model.config)
    source_state = source_model.state_dict()
    expected_source = set(mapping)
    actual_source = set(source_state)
    if actual_source != expected_source:
        missing = sorted(expected_source - actual_source)
        unknown = sorted(actual_source - expected_source)
        raise ExportError(
            f"internal tensor set mismatch; missing={missing}, unknown={unknown}"
        )

    model = HuggingFaceOlmo2ForCausalLM(  # type: ignore[no-untyped-call]
        _hugging_face_config(source_model.config)
    )
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


def _token_text(tokenizer: ByteBPE, token_id: int) -> str:
    return "".join(
        BYTE_LEVEL_SYMBOLS[value] for value in tokenizer.token_bytes(token_id)
    )


def _hugging_face_tokenizer(
    tokenizer: ByteBPE,
    *,
    model_max_length: int,
) -> PreTrainedTokenizerFast:
    vocabulary = dict(SPECIAL_TOKEN_IDS)
    for token_id in range(BYTE_TOKEN_OFFSET, tokenizer.vocab_size):
        token = _token_text(tokenizer, token_id)
        if token in vocabulary:
            raise ExportError(
                f"ordinary tokenizer token collides with a special token: {token!r}"
            )
        vocabulary[token] = token_id
    merges = [
        (_token_text(tokenizer, left), _token_text(tokenizer, right))
        for left, right in tokenizer.merges
    ]
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=merges))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=tokenizer.pretokenizer == "gpt2",
    )
    backend.decoder = decoders.ByteLevel()
    exported = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=backend,
        bos_token=SPECIAL_TOKENS[1],
        eos_token=SPECIAL_TOKENS[2],
        pad_token=SPECIAL_TOKENS[0],
        additional_special_tokens=list(SPECIAL_TOKENS[3:]),
        model_max_length=model_max_length,
        clean_up_tokenization_spaces=False,
    )
    for token, token_id in SPECIAL_TOKEN_IDS.items():
        if exported.convert_tokens_to_ids(token) != token_id:
            raise ExportError(f"exported special-token ID mismatch: {token}")
    if len(exported) != tokenizer.vocab_size:
        raise ExportError("exported tokenizer vocabulary size mismatch")
    return exported


def _verify_fp32_logits(
    source: Olmo2ForCausalLM,
    target: HuggingFaceOlmo2ForCausalLM,
) -> float:
    sequence_length = min(4, source.config.max_position_embeddings)
    ordinary_count = source.config.vocab_size - BYTE_TOKEN_OFFSET
    input_ids = (
        torch.arange(sequence_length, dtype=torch.long).unsqueeze(0) % ordinary_count
        + BYTE_TOKEN_OFFSET
    )
    input_ids[0, 0] = source.config.bos_token_id
    source.eval().float()
    target.eval().float()  # type: ignore[no-untyped-call]
    with torch.no_grad():
        source_logits = source(input_ids).logits
        target_logits = target(input_ids=input_ids, use_cache=False).logits
    maximum_error = float((source_logits - target_logits).abs().max().item())
    if not torch.allclose(
        source_logits,
        target_logits,
        atol=FP32_ATOL,
        rtol=FP32_RTOL,
    ):
        raise ExportError(
            "internal and Hugging Face fp32 logits exceed export tolerance: "
            f"maximum absolute error {maximum_error}"
        )
    return maximum_error


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: Path) -> ExportArtifact:
    return ExportArtifact(
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=_sha256_file(path),
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_manifest(path: Path, manifest: DenseHFExportManifest) -> None:
    with path.open("xb") as handle:
        handle.write(manifest.canonical_bytes())
        handle.flush()
        os.fsync(handle.fileno())


def _write_special_tokens_map(
    path: Path,
    tokenizer: PreTrainedTokenizerFast,
) -> None:
    encoded = json.dumps(
        tokenizer.special_tokens_map,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_directory(temporary: Path, destination: Path) -> None:
    os.replace(temporary, destination)


def export_dense_to_hugging_face(
    checkpoint_directory: str | Path,
    tokenizer_training_manifest: str | Path,
    output_directory: str | Path,
) -> DenseHFExportManifest:
    """Validate, convert, parity-check, and atomically publish a dense export."""

    checkpoint_path = Path(checkpoint_directory)
    checkpoint = validate_checkpoint(checkpoint_path)
    if checkpoint.binding.architecture != "olmo2":
        raise ExportError("dense Hugging Face export requires an OLMo2 checkpoint")
    try:
        model_config = Olmo2Config.model_validate(
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

    source_model = Olmo2ForCausalLM(model_config)
    load_checkpoint_model(
        checkpoint_path,
        model=source_model,
        expected_binding=checkpoint.binding,
    )
    hugging_face_model, tensor_map = _mapped_hugging_face_model(source_model)
    maximum_error = _verify_fp32_logits(source_model, hugging_face_model)
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

    destination = Path(output_directory)
    if destination.exists():
        raise ExportError(f"export destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    try:
        hugging_face_model.save_pretrained(
            temporary,
            safe_serialization=True,
        )
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

        manifest = DenseHFExportManifest(
            source_checkpoint_id=checkpoint.lineage.checkpoint_id,
            source_checkpoint_manifest_sha256=sha256(
                checkpoint.canonical_bytes()
            ).hexdigest(),
            model_config_sha256=model_config.config_hash,
            tokenizer_sha256=tokenizer.model_hash,
            parameter_count=source_model.trainable_parameter_count(),
            transformers_version=transformers_version,
            fp32_max_abs_error=maximum_error,
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


def load_export_manifest(path: str | Path) -> DenseHFExportManifest:
    """Load a canonical export manifest and validate all recorded file hashes."""

    manifest_path = Path(path)
    try:
        raw = manifest_path.read_bytes()
        manifest = DenseHFExportManifest.model_validate_json(raw)
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
        if _sha256_file(artifact_path) != artifact.sha256:
            raise ExportError(f"export artifact hash mismatch: {artifact.filename}")
    return manifest
