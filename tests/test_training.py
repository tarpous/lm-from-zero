from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from typing import cast

import torch
from pydantic import ValidationError
from torch import Tensor, nn

from lm_from_zero.data import DataValidationError, SplitPolicy
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
    CausalBatchConfig,
    OptimizationConfig,
    ShardBatchSource,
    build_adamw,
    clip_gradients,
    partition_parameters,
    set_learning_rate,
)


def _tiny_model() -> Olmo2ForCausalLM:
    config = Olmo2Config(
        model_name="training-test",
        tokenizer_hash="0" * 64,
        vocab_size=272,
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        max_position_embeddings=8,
    )
    return Olmo2ForCausalLM(config)


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


class TrainingDataTests(unittest.TestCase):
    def test_cursor_resume_recreates_the_exact_next_batch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_path = _shard_build(Path(directory))
            config = CausalBatchConfig(
                sequence_length=4,
                micro_batch_size=2,
                seed=19,
            )
            source = ShardBatchSource(build_path, config)
            first = source.next_batch()
            uninterrupted = source.next_batch(first.cursor_after)
            serialized = BatchCursor.model_validate_json(
                first.cursor_after.model_dump_json()
            )
            recreated = ShardBatchSource(build_path, config)
            resumed = recreated.next_batch(serialized)

        torch.testing.assert_close(resumed.input_ids, uninterrupted.input_ids)
        torch.testing.assert_close(resumed.labels, uninterrupted.labels)
        self.assertEqual(resumed.cursor_after, uninterrupted.cursor_after)
        self.assertEqual(first.input_ids.dtype, torch.long)
        self.assertNotEqual(first.input_ids.data_ptr(), first.labels.data_ptr())
        self.assertEqual(first.cursor_after.tokens_consumed, 8)

    def test_rank_partitions_are_disjoint_and_cover_every_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_path = _shard_build(Path(directory))
            complete = ShardBatchSource(
                build_path,
                CausalBatchConfig(sequence_length=4, micro_batch_size=1, shuffle=False),
            )
            rank_sets = [
                set(
                    ShardBatchSource(
                        build_path,
                        CausalBatchConfig(
                            sequence_length=4,
                            micro_batch_size=1,
                            rank=rank,
                            world_size=3,
                            shuffle=False,
                        ),
                    ).rank_window_ids(0)
                )
                for rank in range(3)
            ]

        self.assertEqual(set.union(*rank_sets), set(complete.rank_window_ids(0)))
        self.assertTrue(rank_sets[0].isdisjoint(rank_sets[1]))
        self.assertTrue(rank_sets[0].isdisjoint(rank_sets[2]))
        self.assertTrue(rank_sets[1].isdisjoint(rank_sets[2]))

    def test_shuffle_is_deterministic_and_changes_by_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_path = _shard_build(Path(directory))
            config = CausalBatchConfig(
                sequence_length=4,
                micro_batch_size=1,
                seed=23,
            )
            first = ShardBatchSource(build_path, config)
            second = ShardBatchSource(build_path, config)

            self.assertEqual(first.rank_window_ids(0), second.rank_window_ids(0))
            self.assertNotEqual(first.rank_window_ids(0), first.rank_window_ids(1))

    def test_epoch_rollover_and_cursor_mismatch_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            build_path = _shard_build(Path(directory))
            probe = ShardBatchSource(
                build_path,
                CausalBatchConfig(sequence_length=4, micro_batch_size=1),
            )
            source = ShardBatchSource(
                build_path,
                CausalBatchConfig(
                    sequence_length=4,
                    micro_batch_size=probe.window_count + 1,
                ),
            )
            batch = source.next_batch()
            wrong_seed = batch.cursor_after.model_copy(update={"seed": 999})
            beyond = source.initial_cursor().model_copy(
                update={"next_local_window": source.window_count + 1}
            )

            self.assertEqual(batch.cursor_after.epoch, 1)
            self.assertEqual(batch.cursor_after.next_local_window, 1)
            with self.assertRaisesRegex(DataValidationError, "seed"):
                source.next_batch(wrong_seed)
            with self.assertRaisesRegex(DataValidationError, "beyond"):
                source.next_batch(beyond)

    def test_invalid_batch_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            CausalBatchConfig(rank=2, world_size=2)
        with self.assertRaises(ValidationError):
            BatchCursor(
                build_manifest_sha256="0" * 64,
                tokenizer_hash="1" * 64,
                split="train",
                sequence_length=4,
                seed=1,
                rank=0,
                world_size=1,
                shuffle=True,
                sequences_consumed=1,
                tokens_consumed=3,
            )


class OptimizationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1337)

    def test_parameter_partition_is_complete_and_semantic(self) -> None:
        model = _tiny_model()
        partition = partition_parameters(model)
        all_names = set(partition.decay_names) | set(partition.no_decay_names)

        self.assertEqual(all_names, {name for name, _ in model.named_parameters()})
        self.assertTrue(set(partition.decay_names).isdisjoint(partition.no_decay_names))
        self.assertIn("lm_head.weight", partition.decay_names)
        self.assertIn("embed_tokens.weight", partition.no_decay_names)
        self.assertIn("norm.weight", partition.no_decay_names)
        self.assertTrue(
            all(
                "layernorm.weight" in name
                or "norm.weight" in name
                or name == "embed_tokens.weight"
                for name in partition.no_decay_names
            )
        )

    def test_adamw_groups_and_learning_rate_schedule(self) -> None:
        model = _tiny_model()
        config = OptimizationConfig(total_steps=100)
        optimizer, _ = build_adamw(model, config)

        self.assertEqual(config.warmup_steps, 2)
        self.assertEqual(config.learning_rate_at(0), 0.0005)
        self.assertEqual(config.learning_rate_at(1), 0.001)
        self.assertEqual(config.learning_rate_at(2), 0.001)
        self.assertAlmostEqual(config.learning_rate_at(99), 0.0001)
        self.assertAlmostEqual(config.learning_rate_at(100), 0.0001)
        self.assertEqual(
            [group["weight_decay"] for group in optimizer.param_groups],
            [0.1, 0.0],
        )
        self.assertEqual(
            [group["group_name"] for group in optimizer.param_groups],
            ["decay", "no_decay"],
        )
        self.assertEqual(
            set_learning_rate(optimizer, config, 50), config.learning_rate_at(50)
        )
        self.assertTrue(
            all(
                group["lr"] == config.learning_rate_at(50)
                for group in optimizer.param_groups
            )
        )

    def test_adamw_backend_is_explicit_and_fused_rejects_cpu(self) -> None:
        model = _tiny_model()
        config = OptimizationConfig(total_steps=10)
        foreach, _ = build_adamw(model, config, backend="foreach")

        self.assertTrue(foreach.defaults["foreach"])
        self.assertFalse(foreach.defaults["fused"])
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            build_adamw(model, config, backend="fused", device_type="cpu")

    def test_optimizer_state_round_trip_reproduces_next_cpu_step(self) -> None:
        first = _tiny_model()
        config = OptimizationConfig(total_steps=10)
        first_optimizer, _ = build_adamw(first, config)
        input_ids = torch.tensor([[8, 9, 10, 11]])

        first_loss = cast(Tensor, first(input_ids, labels=input_ids).loss)
        torch.autograd.backward(first_loss)
        first_optimizer.step()
        first_optimizer.zero_grad(set_to_none=True)

        second = _tiny_model()
        second.load_state_dict(first.state_dict())
        second_optimizer, _ = build_adamw(second, config)
        second_optimizer.load_state_dict(copy.deepcopy(first_optimizer.state_dict()))

        for model, optimizer in (
            (first, first_optimizer),
            (second, second_optimizer),
        ):
            loss = cast(Tensor, model(input_ids, labels=input_ids).loss)
            torch.autograd.backward(loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        for first_parameter, second_parameter in zip(
            first.parameters(), second.parameters(), strict=True
        ):
            torch.testing.assert_close(
                first_parameter, second_parameter, rtol=0, atol=0
            )

    def test_gradient_clipping_and_invalid_policy(self) -> None:
        model = nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.full_like(model.weight, 100.0)
        norm = clip_gradients(model, 1.0)

        self.assertGreater(float(norm), 1.0)
        self.assertLessEqual(float(torch.linalg.vector_norm(model.weight.grad)), 1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            clip_gradients(model, 0.0)
        with self.assertRaisesRegex(ValueError, "negative"):
            OptimizationConfig(total_steps=10).learning_rate_at(-1)
        with self.assertRaises(ValidationError):
            OptimizationConfig(total_steps=2, warmup_fraction=0.9)


if __name__ == "__main__":
    unittest.main()
