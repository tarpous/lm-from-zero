from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.data import SplitPolicy
from lm_from_zero.diffusion_evaluation import (
    DiffusionEvaluationConfig,
    DiffusionEvaluationError,
    append_diffusion_evaluation_result,
    evaluate_diffusion,
)
from lm_from_zero.models import MaskedDiffusionConfig, MaskedDiffusionForMaskedLM
from lm_from_zero.sampling import SamplingConfig, sample_text_records
from lm_from_zero.sharding import build_token_shards
from lm_from_zero.tokenizer.bpe import INITIAL_VOCAB_SIZE
from lm_from_zero.tokenizer.pipeline import (
    TokenizerTrainingConfig,
    train_tokenizer_from_sample,
)
from lm_from_zero.training import (
    CausalBatchConfig,
    ShardBatchSource,
    create_checkpoint_binding,
    load_checkpoint_model,
    save_checkpoint,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _shard_build(root: Path) -> Path:
    texts = [
        "alpha beta gamma",
        "delta epsilon zeta",
        "one two three four",
        "red green blue gold",
        "small stories repeat",
        "deterministic tokens here",
    ]
    sample_path = sample_text_records(
        ({"text": text} for text in texts),
        root / "sample",
        SamplingConfig(
            target_text_bytes=sum(len(text.encode()) for text in texts),
            max_storage_bytes=100_000,
        ),
    )
    tokenizer_directory = root / "tokenizer"
    train_tokenizer_from_sample(
        sample_path,
        tokenizer_directory,
        TokenizerTrainingConfig(
            target_vocab_size=INITIAL_VOCAB_SIZE + 8,
            min_frequency=1,
        ),
    )
    output = root / "build"
    build_token_shards(
        sample_path,
        tokenizer_directory / "training.json",
        output,
        split_policy=SplitPolicy(validation_buckets=0, test_buckets=0),
        max_tokens_per_shard=50,
    )
    return output / "build.json"


def _model_config(
    tokenizer_hash: str,
    vocab_size: int,
) -> MaskedDiffusionConfig:
    return MaskedDiffusionConfig(
        model_name="diffusion-evaluation-test",
        tokenizer_hash=tokenizer_hash,
        vocab_size=vocab_size,
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=8,
    )


class DiffusionEvaluationTests(unittest.TestCase):
    def test_seeded_metrics_jsonl_checkpoint_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_path = _shard_build(root)
            batch_config = CausalBatchConfig(
                split="train",
                sequence_length=4,
                micro_batch_size=1,
                shuffle=False,
            )
            source = ShardBatchSource(build_path, batch_config)
            model_config = _model_config(
                source.build.tokenizer_hash,
                source.build.tokenizer_vocab_size,
            )
            torch.manual_seed(53)
            model = MaskedDiffusionForMaskedLM(model_config)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            binding = create_checkpoint_binding(
                architecture="masked_diffusion",
                resolved_model_config=model_config.model_dump(mode="json"),
                tokenizer_sha256=source.build.tokenizer_hash,
                shard_manifest_sha256=source.build_manifest_sha256,
                rank=0,
                world_size=1,
                repository=REPOSITORY,
            )
            checkpoint = save_checkpoint(
                root / "checkpoints",
                model=model,
                optimizer=optimizer,
                cursor=source.initial_cursor(),
                binding=binding,
                optimizer_step=0,
                scheduler_step=0,
            )
            restored = MaskedDiffusionForMaskedLM(model_config)
            load_checkpoint_model(
                checkpoint,
                model=restored,
                expected_binding=binding,
            )

            config = DiffusionEvaluationConfig(
                max_batches=2,
                corruption_samples_per_batch=2,
                seed=19,
            )
            restored.train()
            result = evaluate_diffusion(
                restored,
                source,
                config,
                clock=iter((10.0, 12.0)).__next__,
            )
            self.assertTrue(restored.training)
            self.assertEqual(result.batch_count, 2)
            self.assertEqual(result.source_sequence_count, 2)
            self.assertEqual(result.evaluated_example_count, 4)
            self.assertEqual(result.model_forwards, 4)
            self.assertEqual(result.cursor_after.tokens_consumed, 8)
            self.assertGreater(result.eligible_token_count, 0)
            self.assertGreater(result.masked_token_count, 0)
            self.assertLessEqual(
                result.masked_token_count,
                result.eligible_token_count,
            )
            self.assertFalse(result.causal_perplexity_applicable)
            self.assertNotIn("perplexity", result.model_dump())

            repeated = evaluate_diffusion(
                restored,
                source,
                config,
                clock=iter((20.0, 22.0)).__next__,
            )
            self.assertEqual(
                result.masked_token_count,
                repeated.masked_token_count,
            )
            self.assertEqual(
                result.masked_reconstruction_loss_nats,
                repeated.masked_reconstruction_loss_nats,
            )
            self.assertEqual(
                result.variational_upper_bound_nats,
                repeated.variational_upper_bound_nats,
            )

            jsonl = root / "evaluation.jsonl"
            append_diffusion_evaluation_result(jsonl, result)
            append_diffusion_evaluation_result(jsonl, result)
            self.assertEqual(
                jsonl.read_text().splitlines(),
                [result.canonical_json(), result.canonical_json()],
            )

            cli_jsonl = root / "cli-evaluation.jsonl"
            cli_result = CliRunner().invoke(
                app,
                [
                    "evaluate-diffusion",
                    str(checkpoint),
                    str(build_path),
                    "--max-batches",
                    "1",
                    "--corruption-samples-per-batch",
                    "2",
                    "--sequence-length",
                    "4",
                    "--batch-size",
                    "1",
                    "--split",
                    "train",
                    "--seed",
                    "19",
                    "--jsonl-output",
                    str(cli_jsonl),
                ],
            )
            self.assertEqual(cli_result.exit_code, 0, cli_result.output)
            cli_payload = json.loads(cli_result.stdout)
            self.assertEqual(cli_payload["model_forwards"], 2)
            self.assertFalse(cli_payload["causal_perplexity_applicable"])
            self.assertNotIn("perplexity", cli_payload)
            self.assertEqual(len(cli_jsonl.read_text().splitlines()), 1)

    def test_rejects_mismatch_repetition_cuda_and_static_clock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_path = _shard_build(Path(directory))
            source = ShardBatchSource(
                build_path,
                CausalBatchConfig(
                    split="train",
                    sequence_length=4,
                    micro_batch_size=1,
                    shuffle=False,
                ),
            )
            model = MaskedDiffusionForMaskedLM(
                _model_config(
                    source.build.tokenizer_hash,
                    source.build.tokenizer_vocab_size,
                )
            )
            wrong_model = MaskedDiffusionForMaskedLM(
                _model_config("f" * 64, source.build.tokenizer_vocab_size)
            )

            with self.assertRaisesRegex(DiffusionEvaluationError, "tokenizer"):
                evaluate_diffusion(
                    wrong_model,
                    source,
                    DiffusionEvaluationConfig(max_batches=1),
                )
            with self.assertRaisesRegex(DiffusionEvaluationError, "repeat"):
                evaluate_diffusion(
                    model,
                    source,
                    DiffusionEvaluationConfig(max_batches=source.window_count + 1),
                )
            with self.assertRaisesRegex(DiffusionEvaluationError, "clock"):
                evaluate_diffusion(
                    model,
                    source,
                    DiffusionEvaluationConfig(max_batches=1),
                    clock=lambda: 1.0,
                )
            with (
                patch("torch.cuda.is_available", return_value=False),
                self.assertRaisesRegex(DiffusionEvaluationError, "unavailable"),
            ):
                evaluate_diffusion(
                    model,
                    source,
                    DiffusionEvaluationConfig(
                        max_batches=1,
                        device="cuda",
                    ),
                )


if __name__ == "__main__":
    unittest.main()
