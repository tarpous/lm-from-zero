from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.data import SplitPolicy
from lm_from_zero.evaluation import (
    CausalEvaluationConfig,
    EvaluationError,
    append_evaluation_result,
    evaluate_causal_loss,
)
from lm_from_zero.models import Olmo2Config, Olmo2ForCausalLM
from lm_from_zero.sampling import SamplingConfig, sample_text_records
from lm_from_zero.sharding import build_token_shards
from lm_from_zero.tokenizer.bpe import INITIAL_VOCAB_SIZE
from lm_from_zero.tokenizer.pipeline import (
    TokenizerTrainingConfig,
    train_tokenizer_from_sample,
)
from lm_from_zero.training import (
    CausalBatchConfig,
    OptimizationConfig,
    ShardBatchSource,
    build_adamw,
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


def _model_config(tokenizer_hash: str, vocab_size: int) -> Olmo2Config:
    return Olmo2Config(
        model_name="evaluation-test",
        tokenizer_hash=tokenizer_hash,
        vocab_size=vocab_size,
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        max_position_embeddings=8,
    )


class DenseEvaluationTests(unittest.TestCase):
    def test_checkpoint_model_load_evaluation_jsonl_and_cli(self) -> None:
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
            torch.manual_seed(17)
            model = Olmo2ForCausalLM(model_config)
            optimizer, _ = build_adamw(model, OptimizationConfig(total_steps=2))
            binding = create_checkpoint_binding(
                architecture="olmo2",
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

            restored_model = Olmo2ForCausalLM(model_config)
            load_checkpoint_model(
                checkpoint,
                model=restored_model,
                expected_binding=binding,
            )
            for expected, actual in zip(
                model.parameters(),
                restored_model.parameters(),
                strict=True,
            ):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

            restored_model.train()
            clock_values = iter((10.0, 12.0))
            result = evaluate_causal_loss(
                restored_model,
                source,
                CausalEvaluationConfig(max_batches=2),
                clock=clock_values.__next__,
            )
            self.assertTrue(restored_model.training)
            self.assertEqual(result.batch_count, 2)
            self.assertEqual(result.sequence_count, 2)
            self.assertEqual(result.predicted_token_count, 6)
            self.assertEqual(result.cursor_after.tokens_consumed, 8)
            self.assertEqual(result.elapsed_seconds, 2)
            self.assertEqual(result.predicted_tokens_per_second, 3)
            self.assertAlmostEqual(result.perplexity, math.exp(result.mean_loss))
            self.assertEqual(
                json.dumps(
                    json.loads(result.canonical_json()),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                result.canonical_json(),
            )

            jsonl = root / "evaluation.jsonl"
            append_evaluation_result(jsonl, result)
            append_evaluation_result(jsonl, result)
            self.assertEqual(
                jsonl.read_text().splitlines(),
                [result.canonical_json(), result.canonical_json()],
            )

            cli_jsonl = root / "cli-evaluation.jsonl"
            cli_result = CliRunner().invoke(
                app,
                [
                    "evaluate-dense",
                    str(checkpoint),
                    str(build_path),
                    "--max-batches",
                    "1",
                    "--sequence-length",
                    "4",
                    "--batch-size",
                    "1",
                    "--split",
                    "train",
                    "--jsonl-output",
                    str(cli_jsonl),
                ],
            )
            self.assertEqual(cli_result.exit_code, 0, cli_result.output)
            self.assertEqual(json.loads(cli_result.stdout)["batch_count"], 1)
            self.assertEqual(len(cli_jsonl.read_text().splitlines()), 1)

    def test_evaluation_rejects_mismatch_repetition_and_static_clock(self) -> None:
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
            model = Olmo2ForCausalLM(
                _model_config(
                    source.build.tokenizer_hash,
                    source.build.tokenizer_vocab_size,
                )
            )
            wrong_model = Olmo2ForCausalLM(
                _model_config("f" * 64, source.build.tokenizer_vocab_size)
            )

            with self.assertRaisesRegex(EvaluationError, "tokenizer"):
                evaluate_causal_loss(
                    wrong_model,
                    source,
                    CausalEvaluationConfig(max_batches=1),
                )
            with self.assertRaisesRegex(EvaluationError, "repeat"):
                evaluate_causal_loss(
                    model,
                    source,
                    CausalEvaluationConfig(max_batches=source.window_count + 1),
                )
            with self.assertRaisesRegex(EvaluationError, "clock"):
                evaluate_causal_loss(
                    model,
                    source,
                    CausalEvaluationConfig(max_batches=1),
                    clock=lambda: 1.0,
                )


if __name__ == "__main__":
    unittest.main()
