from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from pydantic import ValidationError
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.data import SplitPolicy
from lm_from_zero.models import Olmo2Config, Olmo2ForCausalLM
from lm_from_zero.sampling import SamplingConfig, sample_text_records
from lm_from_zero.sharding import build_token_shards
from lm_from_zero.tokenizer.bpe import INITIAL_VOCAB_SIZE
from lm_from_zero.tokenizer.pipeline import (
    TokenizerTrainingConfig,
    train_tokenizer_from_sample,
)
from lm_from_zero.training import (
    BatchCursor,
    CausalBatch,
    CausalBatchConfig,
    DenseTrainer,
    DenseTrainingConfig,
    OptimizationConfig,
    ShardBatchSource,
    TrainingRunError,
    build_adamw,
    create_dense_run_plan,
    optimizer_steps_for_token_budget,
    seed_training,
    train_accumulated_step,
    validate_checkpoint,
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


def _model_config(tokenizer_hash: str, vocab_size: int = 272) -> Olmo2Config:
    return Olmo2Config(
        model_name="runner-test",
        tokenizer_hash=tokenizer_hash,
        vocab_size=vocab_size,
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        max_position_embeddings=8,
    )


def _training_config(
    model: Olmo2Config,
    batch: CausalBatchConfig,
    *,
    total_steps: int = 2,
    accumulation_steps: int = 2,
) -> DenseTrainingConfig:
    return DenseTrainingConfig(
        model=model,
        batch=batch,
        optimization=OptimizationConfig(
            total_steps=total_steps,
            gradient_clip_norm=1_000,
        ),
        gradient_accumulation_steps=accumulation_steps,
        checkpoint_every_steps=10,
        checkpoint_every_seconds=10_000,
    )


def _cursor(sequences: int) -> BatchCursor:
    return BatchCursor(
        build_manifest_sha256="1" * 64,
        tokenizer_hash="0" * 64,
        split="train",
        sequence_length=4,
        seed=1337,
        rank=0,
        world_size=1,
        shuffle=True,
        next_local_window=sequences,
        sequences_consumed=sequences,
        tokens_consumed=4 * sequences,
    )


def _batch(rows: list[list[int]], before: int, after: int) -> CausalBatch:
    values = torch.tensor(rows)
    return CausalBatch(
        input_ids=values,
        labels=values.clone(),
        cursor_before=_cursor(before),
        cursor_after=_cursor(after),
    )


class DenseRunnerTests(unittest.TestCase):
    def test_gradient_accumulation_matches_one_large_batch(self) -> None:
        model_config = _model_config("0" * 64)
        large_config = _training_config(
            model_config,
            CausalBatchConfig(sequence_length=4, micro_batch_size=2),
            accumulation_steps=1,
        )
        accumulated_config = _training_config(
            model_config,
            CausalBatchConfig(sequence_length=4, micro_batch_size=1),
            accumulation_steps=2,
        )
        seed_training(19, cuda=False)
        large_model = Olmo2ForCausalLM(model_config)
        accumulated_model = Olmo2ForCausalLM(model_config)
        accumulated_model.load_state_dict(large_model.state_dict())
        large_optimizer, _ = build_adamw(large_model, large_config.optimization)
        accumulated_optimizer, _ = build_adamw(
            accumulated_model,
            accumulated_config.optimization,
        )
        rows = [[8, 9, 10, 11], [12, 13, 14, 15]]

        large_metrics = train_accumulated_step(
            large_model,
            large_optimizer,
            [_batch(rows, 0, 2)],
            large_config,
            zero_based_step=0,
        )
        accumulated_metrics = train_accumulated_step(
            accumulated_model,
            accumulated_optimizer,
            [
                _batch([rows[0]], 0, 1),
                _batch([rows[1]], 1, 2),
            ],
            accumulated_config,
            zero_based_step=0,
        )

        self.assertAlmostEqual(large_metrics.loss, accumulated_metrics.loss)
        self.assertEqual(large_metrics.tokens_consumed, 8)
        self.assertGreater(large_metrics.elapsed_seconds, 0)
        self.assertGreater(large_metrics.tokens_per_second, 0)
        self.assertIsNone(large_metrics.peak_cuda_memory_allocated_bytes)
        self.assertIsNone(large_metrics.peak_cuda_memory_reserved_bytes)
        for large, accumulated in zip(
            large_model.parameters(),
            accumulated_model.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(large, accumulated, atol=1e-7, rtol=1e-6)

    def test_dry_run_plan_and_configuration_reject_invalid_inputs(self) -> None:
        self.assertEqual(
            optimizer_steps_for_token_budget(
                17,
                sequence_length=4,
                micro_batch_size=1,
                gradient_accumulation_steps=2,
            ),
            3,
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            optimizer_steps_for_token_budget(
                0,
                sequence_length=4,
                micro_batch_size=1,
                gradient_accumulation_steps=2,
            )
        with self.assertRaisesRegex(ValidationError, "single-process"):
            _training_config(
                _model_config("0" * 64),
                CausalBatchConfig(
                    sequence_length=4,
                    micro_batch_size=1,
                    rank=1,
                    world_size=2,
                ),
            )
        with self.assertRaisesRegex(ValidationError, "CPU compilation"):
            DenseTrainingConfig(
                model=_model_config("0" * 64),
                batch=CausalBatchConfig(
                    sequence_length=4,
                    micro_batch_size=1,
                ),
                optimization=OptimizationConfig(total_steps=2),
                compile_model=True,
            )

    def test_runner_dry_run_checkpoint_resume_and_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_path = _shard_build(root)
            batch_config = CausalBatchConfig(
                sequence_length=4,
                micro_batch_size=1,
                seed=23,
            )
            source = ShardBatchSource(build_path, batch_config)
            model_config = _model_config(
                source.build.tokenizer_hash,
                source.build.tokenizer_vocab_size,
            )
            config = _training_config(model_config, batch_config)
            plan = create_dense_run_plan(
                config,
                source,
                root / "resumed-checkpoints",
                estimated_tokens_per_second=8,
            )
            self.assertEqual(plan.tokens_per_optimizer_step, 8)
            self.assertEqual(plan.total_training_tokens, 16)
            self.assertEqual(plan.estimated_seconds, 2)
            self.assertEqual(
                plan.estimated_checkpoint_bytes,
                12 * model_config.parameter_breakdown().total,
            )

            seed_training(config.seed, cuda=False)
            interrupted_model = Olmo2ForCausalLM(model_config)
            interrupted = DenseTrainer(
                model=interrupted_model,
                source=source,
                config=config,
                checkpoint_directory=root / "resumed-checkpoints",
                repository=REPOSITORY,
                jsonl_log=root / "resumed.jsonl",
            )
            first = interrupted.run(stop_after_optimizer_step=1)
            self.assertEqual(first.optimizer_step, 1)

            seed_training(config.seed, cuda=False)
            resumed_model = Olmo2ForCausalLM(model_config)
            resumed = DenseTrainer(
                model=resumed_model,
                source=ShardBatchSource(build_path, batch_config),
                config=config,
                checkpoint_directory=root / "resumed-checkpoints",
                repository=REPOSITORY,
                jsonl_log=root / "resumed.jsonl",
            )
            resumed_result = resumed.run(resume_from=first.last_checkpoint)
            self.assertEqual(resumed_result.optimizer_step, 2)
            self.assertEqual(resumed_result.cursor.tokens_consumed, 16)
            self.assertEqual(len(resumed_result.metrics), 1)
            manifest = validate_checkpoint(resumed_result.last_checkpoint)
            self.assertEqual(
                manifest.lineage.parent_checkpoint_id,
                first.last_checkpoint.name,
            )

            seed_training(config.seed, cuda=False)
            uninterrupted_model = Olmo2ForCausalLM(model_config)
            uninterrupted = DenseTrainer(
                model=uninterrupted_model,
                source=ShardBatchSource(build_path, batch_config),
                config=config,
                checkpoint_directory=root / "uninterrupted-checkpoints",
                repository=REPOSITORY,
                jsonl_log=root / "uninterrupted.jsonl",
            )
            uninterrupted_result = uninterrupted.run()
            self.assertEqual(uninterrupted_result.optimizer_step, 2)
            for expected, actual in zip(
                uninterrupted_model.parameters(),
                resumed_model.parameters(),
                strict=True,
            ):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

            records = [
                json.loads(line)
                for line in (root / "resumed.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["event"] for record in records],
                [
                    "run_start",
                    "optimizer_step",
                    "run_stopped",
                    "run_resume",
                    "optimizer_step",
                    "run_complete",
                ],
            )

            with self.assertRaisesRegex(TrainingRunError, "bounded stop"):
                resumed.run(
                    resume_from=resumed_result.last_checkpoint,
                    stop_after_optimizer_step=2,
                )

    def test_pretrain_cli_defaults_to_dry_run_and_gates_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_path = root / "build.json"
            build_path.write_text("{}")
            build = SimpleNamespace(
                tokenizer_hash="a" * 64,
                tokenizer_vocab_size=16_000,
            )
            source = object()
            plan = SimpleNamespace(model_dump_json=lambda: '{"planned":true}')
            result = SimpleNamespace(model_dump_json=lambda: '{"trained":true}')
            run = Mock(return_value=result)
            trainer = SimpleNamespace(run=run)
            arguments = [
                "pretrain-dense",
                str(build_path),
                "--checkpoint-directory",
                str(root / "checkpoints"),
                "--jsonl-log",
                str(root / "run.jsonl"),
                "--target-tokens",
                "16384",
                "--stop-after-optimizer-step",
                "2",
            ]

            with (
                patch("lm_from_zero.cli.validate_shard_build", return_value=build),
                patch(
                    "lm_from_zero.training.ShardBatchSource",
                    return_value=source,
                ),
                patch(
                    "lm_from_zero.training.create_dense_run_plan",
                    return_value=plan,
                ),
            ):
                dry_run = CliRunner().invoke(app, arguments)

            self.assertEqual(
                dry_run.exit_code,
                0,
                f"{dry_run.output}\n{dry_run.exception!r}",
            )
            self.assertEqual(dry_run.stdout.strip(), '{"planned":true}')

            with (
                patch("lm_from_zero.cli.validate_shard_build", return_value=build),
                patch(
                    "lm_from_zero.training.ShardBatchSource",
                    return_value=source,
                ),
                patch(
                    "lm_from_zero.training.create_dense_run_plan",
                    return_value=plan,
                ),
                patch("lm_from_zero.training.seed_training") as seed,
                patch("lm_from_zero.models.Olmo2ForCausalLM"),
                patch(
                    "lm_from_zero.training.DenseTrainer",
                    return_value=trainer,
                ),
            ):
                execution = CliRunner().invoke(app, [*arguments, "--execute"])

            self.assertEqual(execution.exit_code, 0, execution.output)
            self.assertEqual(
                execution.stdout.splitlines(),
                ['{"planned":true}', '{"trained":true}'],
            )
            seed.assert_called_once_with(1_337, cuda=True)
            run.assert_called_once_with(
                resume_from=None,
                stop_after_optimizer_step=2,
            )

            wrong_build = SimpleNamespace(
                tokenizer_hash="a" * 64,
                tokenizer_vocab_size=272,
            )
            with patch(
                "lm_from_zero.cli.validate_shard_build",
                return_value=wrong_build,
            ):
                rejected = CliRunner().invoke(app, arguments)
            self.assertNotEqual(rejected.exit_code, 0)
            self.assertIsInstance(rejected.exception, Exception)


if __name__ == "__main__":
    unittest.main()
