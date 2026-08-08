from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from transformers import (
    AutoTokenizer,
)
from transformers import (
    Olmo2Config as HuggingFaceOlmo2Config,
)
from transformers import (
    Olmo2ForCausalLM as HuggingFaceOlmo2ForCausalLM,
)
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.export_hf import (
    EXPORT_MANIFEST_FILENAME,
    ExportError,
    _DenseExportSource,
    _mapped_hugging_face_model,
    dense_tensor_name_map,
    export_dense_to_hugging_face,
    load_export_manifest,
)
from lm_from_zero.models import Olmo2Config, Olmo2ForCausalLM
from lm_from_zero.post_training.preference_evaluation import DPOPolicyForInference
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


def _checkpoint(root: Path, tokenizer: ByteBPE) -> tuple[Path, Olmo2ForCausalLM]:
    torch.manual_seed(41)
    config = Olmo2Config(
        model_name="export-test",
        tokenizer_hash=tokenizer.model_hash,
        vocab_size=tokenizer.vocab_size,
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        max_position_embeddings=16,
    )
    model = Olmo2ForCausalLM(config).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    binding = create_checkpoint_binding(
        architecture="olmo2",
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


class HuggingFaceExportTests(unittest.TestCase):
    def test_native_generation_cli_streams_from_validated_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path, tokenizer = _tokenizer_artifact(root)
            checkpoint, _ = _checkpoint(root, tokenizer)
            result = CliRunner().invoke(
                app,
                [
                    "generate-dense",
                    str(checkpoint),
                    str(training_path),
                    "hello",
                    "--max-new-tokens",
                    "2",
                    "--stream",
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            records = [
                json.loads(line)
                for line in result.stdout.splitlines()
                if line.startswith("{")
            ]
            self.assertEqual(
                [record["event"] for record in records],
                [
                    "token",
                    "token",
                    "complete",
                ],
            )
            self.assertEqual(records[-1]["model_forwards"], 2)
            self.assertEqual(records[-1]["generated_token_count"], 2)

    def test_standard_reload_tokenizer_logits_and_cache_parity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path, tokenizer = _tokenizer_artifact(root)
            checkpoint, internal = _checkpoint(root, tokenizer)
            output = root / "export"

            result = CliRunner().invoke(
                app,
                [
                    "export-dense-hf",
                    str(checkpoint),
                    str(training_path),
                    "--output-directory",
                    str(output),
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            cli_manifest = json.loads(result.stdout)
            manifest = load_export_manifest(output / EXPORT_MANIFEST_FILENAME)
            self.assertEqual(cli_manifest, manifest.model_dump(mode="json"))
            self.assertLessEqual(manifest.fp32_max_abs_error, 1e-5)
            self.assertEqual(
                len(manifest.tensor_map),
                len(internal.state_dict()),
            )

            config = HuggingFaceOlmo2Config.from_pretrained(
                output,
                local_files_only=True,
            )
            self.assertEqual(config.model_type, "olmo2")
            self.assertEqual(config.architectures, ["Olmo2ForCausalLM"])
            self.assertIsInstance(config.rope_parameters, dict)
            assert isinstance(config.rope_parameters, dict)
            self.assertEqual(config.rope_parameters["rope_theta"], 10_000.0)
            self.assertFalse(config.tie_word_embeddings)
            exported = HuggingFaceOlmo2ForCausalLM.from_pretrained(
                output,
                local_files_only=True,
            )
            exported.eval()  # type: ignore[no-untyped-call]
            exported_tokenizer = AutoTokenizer.from_pretrained(
                output,
                local_files_only=True,
            )
            self.assertEqual(len(exported_tokenizer), tokenizer.vocab_size)
            prompt = "Once upon a time.\n"
            self.assertEqual(
                tokenizer.encode(prompt),
                exported_tokenizer.encode(prompt, add_special_tokens=False),
            )
            special_prompt = "<|bos|>hello<|eos|>"
            self.assertEqual(
                tokenizer.encode(
                    special_prompt,
                    allowed_special={"<|bos|>", "<|eos|>"},
                ),
                exported_tokenizer.encode(
                    special_prompt,
                    add_special_tokens=False,
                ),
            )

            manifest_checkpoint = validate_checkpoint(checkpoint)
            reloaded_internal = Olmo2ForCausalLM(internal.config).eval()
            load_checkpoint_model(
                checkpoint,
                model=reloaded_internal,
                expected_binding=manifest_checkpoint.binding,
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
            self.assertIsNotNone(exported_output.past_key_values)
            next_token = internal_output.logits[:, -1].argmax(dim=-1, keepdim=True)
            with torch.no_grad():
                internal_next = reloaded_internal(
                    next_token,
                    cache=internal_output.cache,
                    use_cache=True,
                )
                exported_next = exported(
                    input_ids=next_token,
                    past_key_values=exported_output.past_key_values,
                    use_cache=True,
                )
                internal_full = reloaded_internal(
                    torch.cat((input_ids, next_token), dim=1)
                )
            torch.testing.assert_close(
                internal_next.logits,
                exported_next.logits,
                atol=1e-5,
                rtol=1e-5,
            )
            torch.testing.assert_close(
                internal_next.logits[:, -1],
                internal_full.logits[:, -1],
                atol=1e-5,
                rtol=1e-5,
            )
            self.assertTrue(
                torch.equal(
                    internal_next.logits[:, -1].argmax(dim=-1),
                    exported_next.logits[:, -1].argmax(dim=-1),
                )
            )

    def test_dpo_checkpoint_dispatches_to_validated_export_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path, tokenizer = _tokenizer_artifact(root)
            checkpoint, internal = _checkpoint(root, tokenizer)
            (checkpoint / "manifest.json").write_text(
                '{"format":"lm-from-zero-dpo-checkpoint"}',
                encoding="utf-8",
            )
            source = _DenseExportSource(
                model=internal,
                checkpoint_id="step-000000000001",
                checkpoint_manifest_sha256="a" * 64,
                checkpoint_format="lm-from-zero-dpo-checkpoint",
                model_config_sha256=internal.config.config_hash,
                tokenizer_sha256=tokenizer.model_hash,
            )
            output = root / "export"
            with patch(
                "lm_from_zero.export_hf._load_dpo_export_source",
                return_value=source,
            ) as loader:
                manifest = export_dense_to_hugging_face(
                    checkpoint,
                    training_path,
                    output,
                )
            loader.assert_called_once_with(checkpoint, tokenizer)
            self.assertEqual(
                manifest.source_checkpoint_format,
                "lm-from-zero-dpo-checkpoint",
            )
            self.assertEqual(manifest.source_checkpoint_id, source.checkpoint_id)

    def test_native_generation_dispatches_to_validated_dpo_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path, tokenizer = _tokenizer_artifact(root)
            checkpoint, internal = _checkpoint(root, tokenizer)
            (checkpoint / "manifest.json").write_text(
                '{"format":"lm-from-zero-dpo-checkpoint"}',
                encoding="utf-8",
            )
            policy = DPOPolicyForInference(
                model=internal,
                checkpoint_id="step-000000000001",
                checkpoint_manifest_sha256="a" * 64,
                model_config_sha256=internal.config.config_hash,
                model_variant="baseline",
            )
            with patch(
                "lm_from_zero.post_training.preference_evaluation.load_final_dpo_policy",
                return_value=policy,
            ) as loader:
                result = CliRunner().invoke(
                    app,
                    [
                        "generate-dense",
                        str(checkpoint),
                        str(training_path),
                        "hello",
                        "--max-new-tokens",
                        "1",
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            loader.assert_called_once_with(checkpoint, tokenizer)

    def test_rejects_mismatches_existing_destinations_and_corruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path, tokenizer = _tokenizer_artifact(root)
            checkpoint, _ = _checkpoint(root, tokenizer)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaisesRegex(ExportError, "already exists"):
                export_dense_to_hugging_face(
                    checkpoint,
                    training_path,
                    existing,
                )

            training_payload = json.loads(training_path.read_text(encoding="utf-8"))
            training_payload["tokenizer_hash"] = "f" * 64
            wrong_manifest = training_path.with_name("wrong-training.json")
            wrong_manifest.write_text(json.dumps(training_payload), encoding="utf-8")
            with self.assertRaisesRegex(ExportError, "does not match the checkpoint"):
                export_dense_to_hugging_face(
                    checkpoint,
                    wrong_manifest,
                    root / "mismatch",
                )

            output = root / "valid"
            export_dense_to_hugging_face(checkpoint, training_path, output)
            with (output / "config.json").open("ab") as handle:
                handle.write(b"\n")
            with self.assertRaisesRegex(ExportError, "size mismatch"):
                load_export_manifest(output / EXPORT_MANIFEST_FILENAME)

    def test_atomic_failure_and_explicit_tensor_set_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training_path, tokenizer = _tokenizer_artifact(root)
            checkpoint, model = _checkpoint(root, tokenizer)
            output = root / "failed"
            with (
                patch(
                    "lm_from_zero.export_hf._publish_directory",
                    side_effect=OSError("simulated interruption"),
                ),
                self.assertRaisesRegex(OSError, "simulated interruption"),
            ):
                export_dense_to_hugging_face(checkpoint, training_path, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".failed-*")), [])

            mapping = dense_tensor_name_map(model.config)
            self.assertEqual(set(mapping), set(model.state_dict()))
            self.assertEqual(len(mapping), len(set(mapping.values())))
            incomplete = dict(mapping)
            incomplete.pop("lm_head.weight")
            with (
                patch(
                    "lm_from_zero.export_hf.dense_tensor_name_map",
                    return_value=incomplete,
                ),
                self.assertRaisesRegex(ExportError, "internal tensor set mismatch"),
            ):
                _mapped_hugging_face_model(model)


if __name__ == "__main__":
    unittest.main()
