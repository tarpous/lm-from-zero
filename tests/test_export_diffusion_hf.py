from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.export_diffusion_hf import (
    _mapped_export_model,
    diffusion_tensor_name_map,
    export_diffusion_to_hugging_face,
    load_diffusion_export_manifest,
)
from lm_from_zero.export_hf import EXPORT_MANIFEST_FILENAME, ExportError
from lm_from_zero.models import MaskedDiffusionConfig, MaskedDiffusionForMaskedLM
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


def _checkpoint(
    root: Path,
    tokenizer: ByteBPE,
) -> tuple[Path, MaskedDiffusionForMaskedLM]:
    torch.manual_seed(47)
    config = MaskedDiffusionConfig(
        model_name="diffusion-export-test",
        tokenizer_hash=tokenizer.model_hash,
        vocab_size=tokenizer.vocab_size,
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=16,
    )
    model = MaskedDiffusionForMaskedLM(config).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    binding = create_checkpoint_binding(
        architecture="masked_diffusion",
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


class DiffusionHuggingFaceExportTests(unittest.TestCase):
    def test_auto_model_reload_tokenizer_logits_and_loss_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path, tokenizer = _tokenizer_artifact(root)
            checkpoint, internal = _checkpoint(root, tokenizer)
            output = root / "export"

            result = CliRunner().invoke(
                app,
                [
                    "export-diffusion-hf",
                    str(checkpoint),
                    str(training_path),
                    "--output-directory",
                    str(output),
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            cli_manifest = json.loads(result.stdout)
            manifest = load_diffusion_export_manifest(output / EXPORT_MANIFEST_FILENAME)
            self.assertEqual(cli_manifest, manifest.model_dump(mode="json"))
            self.assertEqual(manifest.fp32_max_abs_error, 0.0)
            self.assertEqual(manifest.fp32_loss_abs_error, 0.0)
            self.assertTrue(manifest.deterministic_trajectory_matches)
            self.assertTrue(manifest.requires_trust_remote_code)
            self.assertEqual(len(manifest.tensor_map), len(internal.state_dict()))

            config_payload = json.loads(
                (output / "config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                config_payload["auto_map"]["AutoConfig"],
                "hf_diffusion_config.LLaDAConfig",
            )
            self.assertEqual(
                config_payload["auto_map"]["AutoModelForMaskedLM"],
                "hf_diffusion_model.LLaDAForMaskedDiffusion",
            )
            self.assertEqual(config_payload["mask_token_id"], 7)
            exported = AutoModelForMaskedLM.from_pretrained(
                output,
                local_files_only=True,
                trust_remote_code=True,
            )
            exported.eval()
            self.assertEqual(type(exported).__name__, "LLaDAForMaskedDiffusion")
            for field in (
                "vocab_size",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "max_position_embeddings",
                "rope_theta",
                "rms_norm_eps",
                "initializer_range",
                "attention_dropout",
                "corruption_epsilon",
                "mask_token_id",
                "pad_token_id",
                "bos_token_id",
                "eos_token_id",
                "tie_word_embeddings",
                "head_dim",
            ):
                self.assertEqual(
                    getattr(exported.config, field),
                    getattr(internal.config, field),
                    field,
                )
            exported_tokenizer = AutoTokenizer.from_pretrained(
                output,
                local_files_only=True,
                trust_remote_code=True,
            )
            prompt = "Once upon a time.\n"
            self.assertEqual(
                tokenizer.encode(prompt),
                exported_tokenizer.encode(prompt, add_special_tokens=False),
            )

            checkpoint_manifest = validate_checkpoint(checkpoint)
            reloaded_internal = MaskedDiffusionForMaskedLM(internal.config).eval()
            load_checkpoint_model(
                checkpoint,
                model=reloaded_internal,
                expected_binding=checkpoint_manifest.binding,
            )
            self.assertEqual(
                set(reloaded_internal.state_dict()) | {"rotary_embedding.inv_freq"},
                set(exported.state_dict()),
            )
            for name, expected in reloaded_internal.state_dict().items():
                torch.testing.assert_close(
                    expected,
                    exported.state_dict()[name],
                    atol=0,
                    rtol=0,
                )
            torch.testing.assert_close(
                reloaded_internal.rotary_embedding.inv_freq,
                exported.rotary_embedding.inv_freq,
                atol=0,
                rtol=0,
            )
            input_ids = torch.tensor(
                [[1, 7, BYTE_TOKEN_OFFSET + 1, 2]],
                dtype=torch.long,
            )
            positions = torch.arange(input_ids.shape[1]).unsqueeze(0)
            internal_hidden = reloaded_internal.embed_tokens(input_ids)
            exported_hidden = exported.embed_tokens(input_ids)
            torch.testing.assert_close(
                internal_hidden,
                exported_hidden,
                atol=0,
                rtol=0,
            )
            internal_cosine, internal_sine = reloaded_internal.rotary_embedding(
                positions
            )
            exported_cosine, exported_sine = exported.rotary_embedding(positions)
            torch.testing.assert_close(
                internal_cosine,
                exported_cosine,
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                internal_sine,
                exported_sine,
                atol=0,
                rtol=0,
            )
            for internal_layer, exported_layer in zip(
                reloaded_internal.layers,
                exported.layers,
                strict=True,
            ):
                internal_hidden = internal_layer(
                    internal_hidden,
                    internal_cosine,
                    internal_sine,
                )
                exported_hidden = exported_layer(
                    exported_hidden,
                    exported_cosine,
                    exported_sine,
                )
                torch.testing.assert_close(
                    internal_hidden,
                    exported_hidden,
                    atol=1e-5,
                    rtol=1e-5,
                )
            labels = torch.tensor(
                [[-100, BYTE_TOKEN_OFFSET, -100, -100]],
                dtype=torch.long,
            )
            eligible = torch.tensor([[False, True, True, True]])
            time = torch.tensor([0.5])
            with torch.no_grad():
                internal_output = reloaded_internal(
                    input_ids,
                    labels=labels,
                    eligible_mask=eligible,
                    time=time,
                )
                exported_output = exported(
                    input_ids=input_ids,
                    labels=labels,
                    eligible_mask=eligible,
                    time=time,
                )
            torch.testing.assert_close(
                internal_output.logits,
                exported_output.logits,
                atol=1e-5,
                rtol=1e-5,
            )
            self.assertIsNotNone(internal_output.loss)
            self.assertIsNotNone(exported_output.loss)
            torch.testing.assert_close(
                internal_output.loss,
                exported_output.loss,
                atol=1e-5,
                rtol=1e-5,
            )

    def test_explicit_mapping_and_atomic_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path, tokenizer = _tokenizer_artifact(root)
            checkpoint, model = _checkpoint(root, tokenizer)
            mapping = diffusion_tensor_name_map(model.config)
            self.assertEqual(set(mapping), set(model.state_dict()))
            self.assertEqual(len(mapping), len(set(mapping.values())))
            mapped, realized = _mapped_export_model(model)
            self.assertEqual(mapping, realized)
            self.assertEqual(mapped.config.mask_token_id, 7)

            output = root / "failed"
            with (
                patch(
                    "lm_from_zero.export_diffusion_hf._publish_directory",
                    side_effect=OSError("simulated interruption"),
                ),
                self.assertRaisesRegex(OSError, "simulated interruption"),
            ):
                export_diffusion_to_hugging_face(
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
                export_diffusion_to_hugging_face(
                    checkpoint,
                    training_path,
                    existing,
                )

            output = root / "valid"
            export_diffusion_to_hugging_face(checkpoint, training_path, output)
            with (output / "config.json").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ExportError, "size mismatch"):
                load_diffusion_export_manifest(output / EXPORT_MANIFEST_FILENAME)

            checkpoint_manifest = validate_checkpoint(checkpoint)
            wrong_binding = checkpoint_manifest.binding.model_copy(
                update={"architecture": "olmo2"}
            )
            with (
                patch(
                    "lm_from_zero.export_diffusion_hf.validate_checkpoint",
                    return_value=checkpoint_manifest.model_copy(
                        update={"binding": wrong_binding}
                    ),
                ),
                self.assertRaisesRegex(ExportError, "requires a masked-diffusion"),
            ):
                export_diffusion_to_hugging_face(
                    checkpoint,
                    training_path,
                    root / "wrong",
                )


if __name__ == "__main__":
    unittest.main()
