from __future__ import annotations

import json
import math
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import polars as pl
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pydantic import ValidationError
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.data import SplitPolicy
from lm_from_zero.evaluation import CausalEvaluationConfig, evaluate_causal_loss
from lm_from_zero.models import (
    Mamba2Config,
    Mamba2ForCausalLM,
    MaskedDiffusionConfig,
    MaskedDiffusionForMaskedLM,
    Olmo2Config,
    Olmo2ForCausalLM,
)
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
    DiffusionTrainer,
    DiffusionTrainingConfig,
    DistributedContext,
    Mamba2Trainer,
    Mamba2TrainingConfig,
    OptimizationConfig,
    ShardBatchSource,
    TrainingRunError,
    build_adamw,
    create_dense_run_plan,
    create_diffusion_run_plan,
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


def _mamba2_training_config(
    tokenizer_hash: str,
    vocab_size: int,
    batch: CausalBatchConfig,
) -> Mamba2TrainingConfig:
    return Mamba2TrainingConfig(
        model=Mamba2Config(
            model_name="runner-mamba2-test",
            tokenizer_hash=tokenizer_hash,
            vocab_size=vocab_size,
            num_hidden_layers=1,
            hidden_size=16,
            state_size=4,
            expand=2,
            head_dim=8,
            num_heads=4,
            num_groups=2,
            conv_kernel=3,
            chunk_size=4,
            max_position_embeddings=8,
        ),
        batch=batch,
        optimization=OptimizationConfig(
            total_steps=2,
            gradient_clip_norm=1_000,
        ),
        checkpoint_every_steps=10,
        checkpoint_every_seconds=10_000,
    )


def _diffusion_training_config(
    tokenizer_hash: str,
    vocab_size: int,
    batch: CausalBatchConfig,
) -> DiffusionTrainingConfig:
    return DiffusionTrainingConfig(
        model=MaskedDiffusionConfig(
            model_name="runner-diffusion-test",
            tokenizer_hash=tokenizer_hash,
            vocab_size=vocab_size,
            num_hidden_layers=1,
            hidden_size=16,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=8,
        ),
        batch=batch,
        optimization=OptimizationConfig(
            total_steps=2,
            gradient_clip_norm=1_000,
        ),
        checkpoint_every_steps=10,
        checkpoint_every_seconds=10_000,
    )


def _cursor(
    sequences: int,
    *,
    rank: int = 0,
    world_size: int = 1,
) -> BatchCursor:
    return BatchCursor(
        build_manifest_sha256="1" * 64,
        tokenizer_hash="0" * 64,
        split="train",
        sequence_length=4,
        seed=1337,
        rank=rank,
        world_size=world_size,
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


def _parameter_sha256(model: Olmo2ForCausalLM) -> str:
    digest = sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _ddp_runner_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    build_path: str,
    output_root: str,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        root = Path(output_root)
        context = DistributedContext.current()
        batch_config = CausalBatchConfig(
            sequence_length=4,
            micro_batch_size=1,
            seed=23,
            rank=rank,
            world_size=world_size,
        )
        source = ShardBatchSource(build_path, batch_config)
        model_config = _model_config(
            source.build.tokenizer_hash,
            source.build.tokenizer_vocab_size,
        )
        config = _training_config(
            model_config,
            batch_config,
            total_steps=2,
            accumulation_steps=2,
        )
        seed_training(config.seed, cuda=False)
        first_model = Olmo2ForCausalLM(model_config)
        first_trainer = DenseTrainer(
            model=first_model,
            source=source,
            config=config,
            checkpoint_directory=root / "ddp-checkpoints",
            repository=REPOSITORY,
            jsonl_log=root / "ddp.jsonl",
            distributed=context,
        )
        first = first_trainer.run(stop_after_optimizer_step=1)

        seed_training(config.seed, cuda=False)
        resumed_model = Olmo2ForCausalLM(model_config)
        resumed_trainer = DenseTrainer(
            model=resumed_model,
            source=ShardBatchSource(build_path, batch_config),
            config=config,
            checkpoint_directory=root / "ddp-checkpoints",
            repository=REPOSITORY,
            jsonl_log=root / "ddp.jsonl",
            distributed=context,
        )
        resumed = resumed_trainer.run(resume_from=first.last_checkpoint)

        seed_training(config.seed, cuda=False)
        uninterrupted_model = Olmo2ForCausalLM(model_config)
        uninterrupted_trainer = DenseTrainer(
            model=uninterrupted_model,
            source=ShardBatchSource(build_path, batch_config),
            config=config,
            checkpoint_directory=root / "ddp-uninterrupted-checkpoints",
            repository=REPOSITORY,
            jsonl_log=root / "ddp-uninterrupted.jsonl",
            distributed=context,
        )
        uninterrupted_trainer.run()
        (root / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "cursor_rank": resumed.cursor.rank,
                    "global_tokens": resumed.metrics[-1].tokens_consumed,
                    "local_tokens": resumed.cursor.tokens_consumed,
                    "loss": resumed.metrics[-1].loss,
                    "model_sha256": _parameter_sha256(resumed_model),
                    "uninterrupted_model_sha256": _parameter_sha256(
                        uninterrupted_model
                    ),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


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

    def test_distributed_step_reports_globally_reduced_metrics(self) -> None:
        model_config = _model_config("0" * 64)
        config = _training_config(
            model_config,
            CausalBatchConfig(
                sequence_length=4,
                micro_batch_size=1,
                rank=0,
                world_size=2,
            ),
            accumulation_steps=1,
        )
        model = Olmo2ForCausalLM(model_config)
        optimizer, _ = build_adamw(model, config.optimization)
        values = torch.tensor([[8, 9, 10, 11]])
        batch = CausalBatch(
            input_ids=values,
            labels=values.clone(),
            cursor_before=_cursor(0, world_size=2),
            cursor_after=_cursor(1, world_size=2),
        )
        context = DistributedContext(
            rank=0,
            world_size=2,
            local_rank=0,
            backend="gloo",
        )

        def reduce(tensor: torch.Tensor, *, op: object) -> None:
            if op == dist.ReduceOp.SUM:
                tensor.mul_(2)

        with patch("torch.distributed.all_reduce", side_effect=reduce):
            metrics = train_accumulated_step(
                model,
                optimizer,
                [batch],
                config,
                zero_based_step=0,
                distributed=context,
            )
        self.assertEqual(metrics.tokens_consumed, 8)
        self.assertEqual(
            metrics.tokens_per_second,
            config.tokens_per_optimizer_step / metrics.elapsed_seconds,
        )

    def test_rank_recovery_and_primary_checkpoint_coordination(self) -> None:
        model_config = _model_config("0" * 64)
        config = _training_config(
            model_config,
            CausalBatchConfig(
                sequence_length=4,
                micro_batch_size=1,
                rank=1,
                world_size=2,
            ),
            total_steps=2,
            accumulation_steps=2,
        )
        cursor_zero = _cursor(2, rank=0, world_size=2)
        cursor_one = _cursor(2, rank=1, world_size=2)
        trainer = object.__new__(DenseTrainer)
        trainer.config = config
        trainer.source = cast(
            Any,
            SimpleNamespace(initial_cursor=lambda: _cursor(0, rank=1, world_size=2)),
        )
        trainer.distributed = DistributedContext(
            rank=1,
            world_size=2,
            local_rank=1,
            backend="gloo",
        )
        rank_one_state = trainer._rank_recovery_state(cursor_one)
        rank_zero_state = {
            **rank_one_state,
            "cursor": cursor_zero.model_dump(mode="json"),
            "rank": 0,
        }
        scheduler_state = trainer._base_scheduler_state(1)
        scheduler_state["rank_states"] = [rank_zero_state, rank_one_state]
        restored_cursor = trainer._restore_distributed_state(scheduler_state, 1)
        self.assertEqual(restored_cursor, cursor_one)

        primary = Mock(spec=DistributedContext)
        primary.rank = 0
        primary.world_size = 2
        primary.is_primary = True
        primary.enabled = True
        primary.all_gather_object.return_value = (rank_zero_state, rank_one_state)
        primary.broadcast_primary_object.return_value = None
        trainer.distributed = cast(Any, primary)
        trainer.checkpoint_directory = Path("checkpoints")
        trainer.model = Olmo2ForCausalLM(model_config)
        trainer.optimizer, _ = build_adamw(
            trainer.model,
            config.optimization,
        )
        trainer.binding = Mock()
        expected_checkpoint = Path("checkpoints/step-000000000001")
        with (
            patch(
                "lm_from_zero.training.runner.save_checkpoint",
                return_value=expected_checkpoint,
            ) as save,
            patch("lm_from_zero.training.runner.apply_checkpoint_retention") as retain,
        ):
            checkpoint = trainer._save(
                optimizer_step=1,
                cursor=cursor_zero,
                parent=None,
            )
        self.assertEqual(checkpoint, expected_checkpoint)
        self.assertEqual(
            len(save.call_args.kwargs["scheduler_state"]["rank_states"]),
            2,
        )
        retain.assert_called_once()

        primary.is_primary = False
        primary.broadcast_primary_object.return_value = "disk failure"
        with self.assertRaisesRegex(TrainingRunError, "disk failure"):
            trainer._save(
                optimizer_step=1,
                cursor=cursor_zero,
                parent=None,
            )

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
        rank_zero = _training_config(
            _model_config("0" * 64),
            CausalBatchConfig(
                sequence_length=4,
                micro_batch_size=1,
                rank=0,
                world_size=2,
            ),
        )
        rank_one = _training_config(
            _model_config("0" * 64),
            CausalBatchConfig(
                sequence_length=4,
                micro_batch_size=1,
                rank=1,
                world_size=2,
            ),
        )
        self.assertEqual(rank_zero.config_hash, rank_one.config_hash)
        self.assertEqual(rank_zero.local_tokens_per_optimizer_step, 8)
        self.assertEqual(rank_zero.tokens_per_optimizer_step, 16)
        self.assertEqual(
            optimizer_steps_for_token_budget(
                17,
                sequence_length=4,
                micro_batch_size=1,
                gradient_accumulation_steps=2,
                world_size=2,
            ),
            2,
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
                tensorboard_directory=root / "tensorboard",
                parquet_log=root / "metrics.parquet",
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
                tensorboard_directory=root / "tensorboard",
                parquet_log=root / "metrics.parquet",
            )
            resumed_result = resumed.run(resume_from=first.last_checkpoint)
            self.assertEqual(resumed_result.optimizer_step, 2)
            self.assertEqual(resumed_result.cursor.tokens_consumed, 16)
            self.assertEqual(len(resumed_result.metrics), 1)
            metric_frame = pl.read_parquet(root / "metrics.parquet")
            self.assertEqual(metric_frame["optimizer_step"].to_list(), [1, 2])
            self.assertTrue(any((root / "tensorboard").glob("events.out.tfevents.*")))
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

    def test_mamba2_runner_checkpoint_resume_is_cpu_bit_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_path = _shard_build(root)
            batch_config = CausalBatchConfig(
                sequence_length=4,
                micro_batch_size=1,
                seed=31,
            )
            source = ShardBatchSource(build_path, batch_config)
            config = _mamba2_training_config(
                source.build.tokenizer_hash,
                source.build.tokenizer_vocab_size,
                batch_config,
            )

            seed_training(config.seed, cuda=False)
            interrupted_model = Mamba2ForCausalLM(config.model)
            interrupted = Mamba2Trainer(
                model=interrupted_model,
                source=source,
                config=config,
                checkpoint_directory=root / "mamba2-resumed-checkpoints",
                repository=REPOSITORY,
                jsonl_log=root / "mamba2-resumed.jsonl",
            )
            first = interrupted.run(stop_after_optimizer_step=1)
            first_manifest = validate_checkpoint(first.last_checkpoint)
            self.assertEqual(first_manifest.binding.architecture, "mamba2")

            resumed_model = Mamba2ForCausalLM(config.model)
            resumed = Mamba2Trainer(
                model=resumed_model,
                source=ShardBatchSource(build_path, batch_config),
                config=config,
                checkpoint_directory=root / "mamba2-resumed-checkpoints",
                repository=REPOSITORY,
                jsonl_log=root / "mamba2-resumed.jsonl",
            )
            resumed_result = resumed.run(
                resume_from=first.last_checkpoint,
                stop_after_optimizer_step=2,
            )

            seed_training(config.seed, cuda=False)
            uninterrupted_model = Mamba2ForCausalLM(config.model)
            uninterrupted = Mamba2Trainer(
                model=uninterrupted_model,
                source=ShardBatchSource(build_path, batch_config),
                config=config,
                checkpoint_directory=root / "mamba2-uninterrupted-checkpoints",
                repository=REPOSITORY,
                jsonl_log=root / "mamba2-uninterrupted.jsonl",
            )
            uninterrupted_result = uninterrupted.run()

            self.assertEqual(resumed_result.optimizer_step, 2)
            self.assertEqual(uninterrupted_result.optimizer_step, 2)
            for expected, actual in zip(
                uninterrupted_model.parameters(),
                resumed_model.parameters(),
                strict=True,
            ):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            evaluation_batch = CausalBatchConfig(
                split="train",
                sequence_length=4,
                micro_batch_size=1,
                shuffle=False,
            )
            evaluation = evaluate_causal_loss(
                resumed_model,
                ShardBatchSource(build_path, evaluation_batch),
                CausalEvaluationConfig(max_batches=1),
            )
            self.assertEqual(evaluation.model_config_sha256, config.model.config_hash)
            self.assertTrue(math.isfinite(evaluation.perplexity))

    def test_diffusion_runner_checkpoint_resume_is_cpu_bit_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_path = _shard_build(root)
            batch_config = CausalBatchConfig(
                sequence_length=4,
                micro_batch_size=1,
                seed=37,
            )
            source = ShardBatchSource(build_path, batch_config)
            config = _diffusion_training_config(
                source.build.tokenizer_hash,
                source.build.tokenizer_vocab_size,
                batch_config,
            )
            plan = create_diffusion_run_plan(
                config,
                source,
                root / "diffusion-resumed-checkpoints",
                reference_training_flops=10_000_000,
            )
            self.assertEqual(plan.reference_training_flops, 10_000_000)
            self.assertEqual(
                plan.training_flop_ratio,
                plan.estimated_training_flops / 10_000_000,
            )

            seed_training(config.seed, cuda=False)
            interrupted_model = MaskedDiffusionForMaskedLM(config.model)
            interrupted = DiffusionTrainer(
                model=interrupted_model,
                source=source,
                config=config,
                checkpoint_directory=root / "diffusion-resumed-checkpoints",
                repository=REPOSITORY,
                jsonl_log=root / "diffusion-resumed.jsonl",
            )
            first = interrupted.run(stop_after_optimizer_step=1)
            first_manifest = validate_checkpoint(first.last_checkpoint)
            self.assertEqual(
                first_manifest.binding.architecture,
                "masked_diffusion",
            )

            resumed_model = MaskedDiffusionForMaskedLM(config.model)
            resumed = DiffusionTrainer(
                model=resumed_model,
                source=ShardBatchSource(build_path, batch_config),
                config=config,
                checkpoint_directory=root / "diffusion-resumed-checkpoints",
                repository=REPOSITORY,
                jsonl_log=root / "diffusion-resumed.jsonl",
            )
            resumed_result = resumed.run(
                resume_from=first.last_checkpoint,
                stop_after_optimizer_step=2,
            )

            seed_training(config.seed, cuda=False)
            uninterrupted_model = MaskedDiffusionForMaskedLM(config.model)
            uninterrupted = DiffusionTrainer(
                model=uninterrupted_model,
                source=ShardBatchSource(build_path, batch_config),
                config=config,
                checkpoint_directory=root / "diffusion-uninterrupted-checkpoints",
                repository=REPOSITORY,
                jsonl_log=root / "diffusion-uninterrupted.jsonl",
            )
            uninterrupted_result = uninterrupted.run()

            self.assertEqual(resumed_result.optimizer_step, 2)
            self.assertEqual(uninterrupted_result.optimizer_step, 2)
            self.assertEqual(
                resumed_result.metrics[-1].loss,
                uninterrupted_result.metrics[-1].loss,
            )
            for expected, actual in zip(
                uninterrupted_model.parameters(),
                resumed_model.parameters(),
                strict=True,
            ):
                torch.testing.assert_close(expected, actual, rtol=0, atol=0)

    def test_cpu_ddp_reduces_metrics_and_coordinates_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_path = _shard_build(root)
            rendezvous = root / "distributed-init"
            spawn_processes = cast(Any, mp.spawn)  # type: ignore[attr-defined]
            spawn_processes(
                _ddp_runner_worker,
                args=(2, str(rendezvous), str(build_path), str(root)),
                nprocs=2,
                join=True,
            )

            ranks = [
                json.loads((root / f"rank-{rank}.json").read_text())
                for rank in range(2)
            ]
            self.assertEqual([record["cursor_rank"] for record in ranks], [0, 1])
            self.assertEqual({record["local_tokens"] for record in ranks}, {16})
            self.assertEqual({record["global_tokens"] for record in ranks}, {32})
            self.assertEqual(len({record["loss"] for record in ranks}), 1)
            self.assertEqual(len({record["model_sha256"] for record in ranks}), 1)
            self.assertTrue(
                all(
                    record["model_sha256"] == record["uninterrupted_model_sha256"]
                    for record in ranks
                )
            )

            records = [
                json.loads(line)
                for line in (root / "ddp.jsonl").read_text().splitlines()
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
            self.assertEqual(records[-1]["tokens_consumed"], 32)
            manifest = validate_checkpoint(
                root / "ddp-checkpoints" / "step-000000000002"
            )
            self.assertEqual(manifest.binding.rank, 0)
            self.assertEqual(manifest.binding.world_size, 2)

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

            mamba_arguments = [
                "pretrain-mamba2",
                *arguments[1:],
            ]
            with (
                patch("lm_from_zero.cli.validate_shard_build", return_value=build),
                patch(
                    "lm_from_zero.training.ShardBatchSource",
                    return_value=source,
                ),
                patch(
                    "lm_from_zero.training.create_mamba2_run_plan",
                    return_value=plan,
                ),
            ):
                mamba_dry_run = CliRunner().invoke(app, mamba_arguments)
            self.assertEqual(
                mamba_dry_run.exit_code,
                0,
                f"{mamba_dry_run.output}\n{mamba_dry_run.exception!r}",
            )
            self.assertEqual(mamba_dry_run.stdout.strip(), '{"planned":true}')

            diffusion_arguments = [
                "pretrain-diffusion",
                *arguments[1:],
            ]
            with (
                patch("lm_from_zero.cli.validate_shard_build", return_value=build),
                patch(
                    "lm_from_zero.training.ShardBatchSource",
                    return_value=source,
                ),
                patch(
                    "lm_from_zero.training.create_diffusion_run_plan",
                    return_value=plan,
                ),
            ):
                diffusion_dry_run = CliRunner().invoke(app, diffusion_arguments)
            self.assertEqual(
                diffusion_dry_run.exit_code,
                0,
                f"{diffusion_dry_run.output}\n{diffusion_dry_run.exception!r}",
            )
            self.assertEqual(diffusion_dry_run.stdout.strip(), '{"planned":true}')

    def test_mamba2_summary_evaluation_and_generation_cli_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            build_manifest = root / "build.json"
            build_manifest.write_text("{}")
            tokenizer_manifest = root / "training.json"
            tokenizer_manifest.write_text("{}")
            tokenizer_hash = "a" * 64
            config = Mamba2Config(
                tokenizer_hash=tokenizer_hash,
                vocab_size=272,
                num_hidden_layers=1,
                hidden_size=16,
                state_size=4,
                expand=2,
                head_dim=8,
                num_heads=4,
                num_groups=2,
                conv_kernel=3,
                chunk_size=4,
                max_position_embeddings=8,
            )
            training_manifest = SimpleNamespace(
                status="complete",
                realized_vocab_size=16_000,
                tokenizer_hash=tokenizer_hash,
                tokenizer_file="tokenizer.json",
            )

            pinned_count = (
                Mamba2Config(tokenizer_hash=tokenizer_hash).parameter_breakdown().total
            )
            summary_model = SimpleNamespace(
                trainable_parameter_count=lambda: pinned_count
            )
            with (
                patch(
                    "lm_from_zero.cli.load_training_manifest",
                    return_value=training_manifest,
                ),
                patch(
                    "lm_from_zero.models.Mamba2ForCausalLM",
                    return_value=summary_model,
                ),
            ):
                summary = CliRunner().invoke(
                    app,
                    ["mamba2-model-summary", str(tokenizer_manifest)],
                )
            self.assertEqual(summary.exit_code, 0, summary.output)
            self.assertEqual(
                json.loads(summary.stdout)["parameters"]["total"], 19_943_164
            )

            binding = SimpleNamespace(
                architecture="mamba2",
                resolved_model_config=config.model_dump(mode="json"),
                tokenizer_sha256=tokenizer_hash,
            )
            manifest = SimpleNamespace(binding=binding)
            model = Mock()
            evaluation = SimpleNamespace(canonical_json=lambda: '{"evaluated":true}')
            with (
                patch(
                    "lm_from_zero.training.validate_checkpoint",
                    return_value=manifest,
                ),
                patch("lm_from_zero.training.ShardBatchSource"),
                patch(
                    "lm_from_zero.models.Mamba2ForCausalLM",
                    return_value=model,
                ),
                patch("lm_from_zero.training.load_checkpoint_model"),
                patch(
                    "lm_from_zero.evaluation.evaluate_causal_loss",
                    return_value=evaluation,
                ),
            ):
                evaluated = CliRunner().invoke(
                    app,
                    [
                        "evaluate-mamba2",
                        str(checkpoint),
                        str(build_manifest),
                        "--sequence-length",
                        "4",
                        "--max-batches",
                        "1",
                    ],
                )
            self.assertEqual(evaluated.exit_code, 0, evaluated.output)
            self.assertEqual(evaluated.stdout.strip(), '{"evaluated":true}')

            tokenizer = SimpleNamespace(
                model_hash=tokenizer_hash,
                encode=Mock(return_value=[8, 9]),
                decode=Mock(return_value="generated"),
            )
            generated_result = SimpleNamespace(
                generated_token_ids=((10, 11),),
                model_dump=lambda mode: {
                    "prompt_token_counts": [2],
                    "generated_token_ids": [[10, 11]],
                    "stop_reasons": ["max_new_tokens"],
                    "model_forwards": 2,
                    "generated_token_count": 2,
                    "elapsed_seconds": 0.1,
                    "tokens_per_second": 20.0,
                },
            )
            with (
                patch(
                    "lm_from_zero.training.validate_checkpoint",
                    return_value=manifest,
                ),
                patch(
                    "lm_from_zero.cli.load_training_manifest",
                    return_value=training_manifest,
                ),
                patch(
                    "lm_from_zero.tokenizer.bpe.ByteBPE.load",
                    return_value=tokenizer,
                ),
                patch(
                    "lm_from_zero.models.Mamba2ForCausalLM",
                    return_value=model,
                ),
                patch("lm_from_zero.training.load_checkpoint_model"),
                patch(
                    "lm_from_zero.generation.generate_causal",
                    return_value=generated_result,
                ),
            ):
                generated = CliRunner().invoke(
                    app,
                    [
                        "generate-mamba2",
                        str(checkpoint),
                        str(tokenizer_manifest),
                        "hello",
                        "--max-new-tokens",
                        "2",
                    ],
                )
            self.assertEqual(generated.exit_code, 0, generated.output)
            self.assertEqual(
                json.loads(generated.stdout)["generated_text"], "generated"
            )


if __name__ == "__main__":
    unittest.main()
