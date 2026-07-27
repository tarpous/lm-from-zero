from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.export_hf import EXPORT_MANIFEST_FILENAME, ExportError
from lm_from_zero.export_mamba2_hf import (
    _mapped_transformers_model,
    export_mamba2_to_hugging_face,
    load_mamba2_export_manifest,
    mamba2_tensor_name_map,
)
from lm_from_zero.models import Mamba2Config, Mamba2ForCausalLM
from lm_from_zero.tokenizer.bpe import BYTE_TOKEN_OFFSET, ByteBPE
from lm_from_zero.tokenizer.pipeline import (
    TokenizerTrainingConfig,
    TokenizerTrainingManifest,
)
from lm_from_zero.training import (
    BatchCursor,
    create_checkpoint_binding,
    load_checkpoint_model,
    save_checkpoint,
    validate_checkpoint,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _tokenizer_artifact(root: Path) -> tuple[Path, ByteBPE]:
    directory = root / "tokenizer"
    directory.mkdir()
    tokenizer = ByteBPE(pretokenizer="gpt2")
    tokenizer.save(directory / "tokenizer.json")
    training = TokenizerTrainingManifest(
        status="complete",
        training_config=TokenizerTrainingConfig(
            target_vocab_size=tokenizer.vocab_size,
            min_frequency=1,
        ),
        source_dataset_id="test/dataset",
        source_revision="test-revision",
        source_sample_sha256="1" * 64,
        source_content_sha256="2" * 64,
        tokenizer_hash=tokenizer.model_hash,
        realized_vocab_size=tokenizer.vocab_size,
        merge_count=0,
        resumed_from_merge_count=0,
        corpus_sha256="3" * 64,
        document_count=1,
        corpus_bytes=16,
        segment_count=1,
        unique_segment_count=1,
        elapsed_seconds=0.1,
    )
    manifest_path = directory / "training.json"
    manifest_path.write_text(
        json.dumps(
            training.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    return manifest_path, tokenizer


def _checkpoint(root: Path, tokenizer: ByteBPE) -> tuple[Path, Mamba2ForCausalLM]:
    torch.manual_seed(43)
    config = Mamba2Config(
        model_name="mamba2-export-test",
        tokenizer_hash=tokenizer.model_hash,
        vocab_size=tokenizer.vocab_size,
        num_hidden_layers=2,
        hidden_size=32,
        state_size=8,
        expand=2,
        head_dim=8,
        num_heads=8,
        num_groups=2,
        conv_kernel=4,
        chunk_size=4,
        max_position_embeddings=16,
    )
    model = Mamba2ForCausalLM(config).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    binding = create_checkpoint_binding(
        architecture="mamba2",
        resolved_model_config=config.model_dump(mode="json"),
        tokenizer_sha256=tokenizer.model_hash,
        shard_manifest_sha256="4" * 64,
        rank=0,
        world_size=1,
        repository=REPOSITORY,
    )
    cursor = BatchCursor(
        build_manifest_sha256="4" * 64,
        tokenizer_hash=tokenizer.model_hash,
        split="train",
        sequence_length=4,
        seed=1337,
        rank=0,
        world_size=1,
        shuffle=True,
        next_local_window=1,
        sequences_consumed=1,
        tokens_consumed=4,
    )
    checkpoint = save_checkpoint(
        root / "checkpoints",
        model=model,
        optimizer=optimizer,
        cursor=cursor,
        binding=binding,
        optimizer_step=1,
        scheduler_step=1,
    )
    return checkpoint, model


class Mamba2HuggingFaceExportTests(unittest.TestCase):
    def test_auto_model_reload_tokenizer_logits_and_cache_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path, tokenizer = _tokenizer_artifact(root)
            checkpoint, internal = _checkpoint(root, tokenizer)
            output = root / "export"

            result = CliRunner().invoke(
                app,
                [
                    "export-mamba2-hf",
                    str(checkpoint),
                    str(training_path),
                    "--output-directory",
                    str(output),
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            cli_manifest = json.loads(result.stdout)
            manifest = load_mamba2_export_manifest(output / EXPORT_MANIFEST_FILENAME)
            self.assertEqual(cli_manifest, manifest.model_dump(mode="json"))
            self.assertEqual(manifest.fp32_max_abs_error, 0.0)
            self.assertEqual(manifest.cached_fp32_max_abs_error, 0.0)
            self.assertTrue(manifest.requires_trust_remote_code)
            self.assertFalse(manifest.native_unfused_transformers_parity)
            self.assertEqual(len(manifest.tensor_map), len(internal.state_dict()))

            config_payload = json.loads(
                (output / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                config_payload["auto_map"]["AutoModelForCausalLM"],
                "hf_mamba2_compat.GroupedMamba2ForCausalLM",
            )
            self.assertEqual(config_payload["rms_norm_group_size"], 32)
            exported = AutoModelForCausalLM.from_pretrained(
                output,
                local_files_only=True,
                trust_remote_code=True,
            )
            exported.eval()  # type: ignore[no-untyped-call]
            self.assertEqual(type(exported).__name__, "GroupedMamba2ForCausalLM")
            exported_tokenizer = AutoTokenizer.from_pretrained(
                output,
                local_files_only=True,
            )
            prompt = "Once upon a time.\n"
            self.assertEqual(
                tokenizer.encode(prompt),
                exported_tokenizer.encode(prompt, add_special_tokens=False),
            )

            checkpoint_manifest = validate_checkpoint(checkpoint)
            reloaded_internal = Mamba2ForCausalLM(internal.config).eval()
            load_checkpoint_model(
                checkpoint,
                model=reloaded_internal,
                expected_binding=checkpoint_manifest.binding,
            )
            input_ids = torch.tensor(
                [[1, BYTE_TOKEN_OFFSET, BYTE_TOKEN_OFFSET + 1, 2]],
                dtype=torch.long,
            )
            with torch.no_grad():
                internal_output = reloaded_internal(input_ids, use_cache=True)
                exported_output = exported(input_ids=input_ids, use_cache=True)
            torch.testing.assert_close(
                internal_output.logits,
                exported_output.logits,
                atol=1e-5,
                rtol=1e-5,
            )
            self.assertIsNotNone(internal_output.cache)
            self.assertIsNotNone(exported_output.cache_params)
            next_token = internal_output.logits[:, -1].argmax(dim=-1, keepdim=True)
            with torch.no_grad():
                internal_next = reloaded_internal(
                    next_token,
                    cache=internal_output.cache,
                    use_cache=True,
                )
                exported_next = exported(
                    input_ids=next_token,
                    cache_params=exported_output.cache_params,
                    use_cache=True,
                )
            torch.testing.assert_close(
                internal_next.logits,
                exported_next.logits,
                atol=1e-5,
                rtol=1e-5,
            )

    def test_explicit_mapping_and_atomic_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path, tokenizer = _tokenizer_artifact(root)
            checkpoint, model = _checkpoint(root, tokenizer)
            mapping = mamba2_tensor_name_map(model.config)
            self.assertEqual(set(mapping), set(model.state_dict()))
            self.assertEqual(len(mapping), len(set(mapping.values())))
            mapped, realized = _mapped_transformers_model(model)
            self.assertEqual(mapping, realized)
            self.assertEqual(
                mapped.config.rms_norm_group_size,
                model.config.inner_size // model.config.num_groups,
            )

            output = root / "failed"
            with (
                patch(
                    "lm_from_zero.export_mamba2_hf._publish_directory",
                    side_effect=OSError("simulated interruption"),
                ),
                self.assertRaisesRegex(OSError, "simulated interruption"),
            ):
                export_mamba2_to_hugging_face(
                    checkpoint,
                    training_path,
                    output,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".failed-*")), [])

    def test_rejects_wrong_architecture_existing_destination_and_corruption(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path, tokenizer = _tokenizer_artifact(root)
            checkpoint, _ = _checkpoint(root, tokenizer)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ExportError, "already exists"):
                export_mamba2_to_hugging_face(
                    checkpoint,
                    training_path,
                    existing,
                )

            output = root / "valid"
            export_mamba2_to_hugging_face(checkpoint, training_path, output)
            with (output / "config.json").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ExportError, "size mismatch"):
                load_mamba2_export_manifest(output / EXPORT_MANIFEST_FILENAME)

            checkpoint_manifest = validate_checkpoint(checkpoint)
            wrong_binding = checkpoint_manifest.binding.model_copy(
                update={"architecture": "olmo2"}
            )
            with (
                patch(
                    "lm_from_zero.export_mamba2_hf.validate_checkpoint",
                    return_value=checkpoint_manifest.model_copy(
                        update={"binding": wrong_binding}
                    ),
                ),
                self.assertRaisesRegex(ExportError, "requires a Mamba-2"),
            ):
                export_mamba2_to_hugging_face(
                    checkpoint,
                    training_path,
                    root / "wrong",
                )


if __name__ == "__main__":
    unittest.main()
