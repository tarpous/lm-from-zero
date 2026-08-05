from __future__ import annotations

import copy
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from torch import Tensor

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
    CausalBatchConfig,
    CheckpointBinding,
    CheckpointCadence,
    CheckpointError,
    OptimizationConfig,
    ShardBatchSource,
    apply_checkpoint_retention,
    build_adamw,
    checkpointing,
    create_checkpoint_binding,
    restore_checkpoint,
    save_checkpoint,
    set_learning_rate,
    validate_checkpoint,
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _tiny_model(tokenizer_hash: str = "0" * 64) -> Olmo2ForCausalLM:
    config = Olmo2Config(
        model_name="checkpoint-test",
        tokenizer_hash=tokenizer_hash,
        vocab_size=272,
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        max_position_embeddings=8,
    )
    return Olmo2ForCausalLM(config)


def _binding(
    model: Olmo2ForCausalLM,
    *,
    tokenizer_hash: str = "0" * 64,
    shard_hash: str = "1" * 64,
    rank: int = 0,
    world_size: int = 1,
) -> CheckpointBinding:
    return create_checkpoint_binding(
        architecture="olmo2",
        resolved_model_config=model.config.model_dump(mode="json"),
        tokenizer_sha256=tokenizer_hash,
        shard_manifest_sha256=shard_hash,
        rank=rank,
        world_size=world_size,
        repository=REPOSITORY,
    )


def _cursor(
    *,
    tokenizer_hash: str = "0" * 64,
    shard_hash: str = "1" * 64,
    rank: int = 0,
    world_size: int = 1,
    sequences: int = 2,
) -> BatchCursor:
    return BatchCursor(
        build_manifest_sha256=shard_hash,
        tokenizer_hash=tokenizer_hash,
        split="train",
        sequence_length=4,
        seed=1337,
        rank=rank,
        world_size=world_size,
        shuffle=True,
        next_local_window=sequences,
        sequences_consumed=sequences,
        tokens_consumed=sequences * 4,
    )


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


def _assert_nested_equal(
    testcase: unittest.TestCase, first: object, second: object
) -> None:
    if isinstance(first, Tensor):
        testcase.assertIsInstance(second, Tensor)
        torch.testing.assert_close(first, cast(Tensor, second), rtol=0, atol=0)
        return
    if isinstance(first, dict):
        testcase.assertIsInstance(second, dict)
        second_dict = cast(dict[object, object], second)
        testcase.assertEqual(first.keys(), second_dict.keys())
        for key, value in first.items():
            _assert_nested_equal(testcase, value, second_dict[key])
        return
    if isinstance(first, (list, tuple)):
        testcase.assertIsInstance(second, type(first))
        second_sequence = cast(list[object] | tuple[object, ...], second)
        testcase.assertEqual(len(first), len(second_sequence))
        for left, right in zip(first, second_sequence, strict=True):
            _assert_nested_equal(testcase, left, right)
        return
    testcase.assertEqual(first, second)


class CheckpointRoundTripTests(unittest.TestCase):
    def setUp(self) -> None:
        random.seed(1337)
        np.random.seed(1337)
        torch.manual_seed(1337)

    def test_safetensors_model_round_trip_preserves_names_and_dtypes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = _tiny_model()
            optimizer, _ = build_adamw(model, OptimizationConfig(total_steps=10))
            expected = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            path = save_checkpoint(
                Path(directory),
                model=model,
                optimizer=optimizer,
                cursor=_cursor(),
                binding=_binding(model),
                optimizer_step=1,
                scheduler_step=1,
            )
            tensors = load_safetensors(path / "model.safetensors")

            self.assertEqual(tensors.keys(), expected.keys())
            for name, tensor in tensors.items():
                self.assertEqual(tensor.dtype, expected[name].dtype)
                torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)

            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
            restore_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                expected_binding=_binding(model),
            )
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)

    def test_full_checkpoint_reproduces_exact_next_cpu_step_and_rng(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_path = _shard_build(root)
            batch_config = CausalBatchConfig(
                sequence_length=4,
                micro_batch_size=2,
                seed=19,
            )
            source = ShardBatchSource(build_path, batch_config)
            tokenizer_hash = source.build.tokenizer_hash
            model = _tiny_model(tokenizer_hash)
            optimization = OptimizationConfig(total_steps=10)
            optimizer, _ = build_adamw(model, optimization)

            first_batch = source.next_batch()
            set_learning_rate(optimizer, optimization, 0)
            first_loss = cast(
                Tensor,
                model(first_batch.input_ids, labels=first_batch.labels).loss,
            )
            torch.autograd.backward(first_loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            binding = _binding(
                model,
                tokenizer_hash=tokenizer_hash,
                shard_hash=source.build_manifest_sha256,
            )
            path = save_checkpoint(
                root / "checkpoints",
                model=model,
                optimizer=optimizer,
                cursor=first_batch.cursor_after,
                binding=binding,
                optimizer_step=1,
                scheduler_step=1,
                scheduler_state={"next_step": 1, "tokens": 8},
                best_metric=2.5,
            )

            expected_python = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = torch.rand(4)
            next_batch = source.next_batch(first_batch.cursor_after)
            set_learning_rate(optimizer, optimization, 1)
            uninterrupted_loss = cast(
                Tensor,
                model(next_batch.input_ids, labels=next_batch.labels).loss,
            )
            torch.autograd.backward(uninterrupted_loss)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            expected_model = copy.deepcopy(model.state_dict())
            expected_optimizer = copy.deepcopy(optimizer.state_dict())

            resumed_source = ShardBatchSource(build_path, batch_config)
            resumed_model = _tiny_model(tokenizer_hash)
            resumed_optimizer, _ = build_adamw(resumed_model, optimization)
            restored = restore_checkpoint(
                path,
                model=resumed_model,
                optimizer=resumed_optimizer,
                expected_binding=binding,
            )

            self.assertEqual(restored.manifest.cursor, first_batch.cursor_after)
            self.assertEqual(restored.manifest.progress.optimizer_step, 1)
            self.assertEqual(restored.manifest.progress.tokens_consumed, 8)
            self.assertEqual(restored.manifest.progress.best_metric, 2.5)
            self.assertEqual(restored.scheduler_state, {"next_step": 1, "tokens": 8})
            self.assertEqual(random.random(), expected_python)
            self.assertEqual(float(np.random.random()), expected_numpy)
            torch.testing.assert_close(torch.rand(4), expected_torch, rtol=0, atol=0)

            resumed_batch = resumed_source.next_batch(restored.manifest.cursor)
            torch.testing.assert_close(
                resumed_batch.input_ids, next_batch.input_ids, rtol=0, atol=0
            )
            set_learning_rate(resumed_optimizer, optimization, 1)
            resumed_loss = cast(
                Tensor,
                resumed_model(
                    resumed_batch.input_ids,
                    labels=resumed_batch.labels,
                ).loss,
            )
            torch.testing.assert_close(
                resumed_loss, uninterrupted_loss.detach(), rtol=0, atol=0
            )
            torch.autograd.backward(resumed_loss)
            resumed_optimizer.step()
            resumed_optimizer.zero_grad(set_to_none=True)

            for name, tensor in resumed_model.state_dict().items():
                torch.testing.assert_close(tensor, expected_model[name], rtol=0, atol=0)
            _assert_nested_equal(
                self, resumed_optimizer.state_dict(), expected_optimizer
            )


class CheckpointValidationTests(unittest.TestCase):
    def _saved(
        self, root: Path
    ) -> tuple[Path, Olmo2ForCausalLM, Any, CheckpointBinding]:
        model = _tiny_model()
        optimizer, _ = build_adamw(model, OptimizationConfig(total_steps=10))
        binding = _binding(model)
        path = save_checkpoint(
            root,
            model=model,
            optimizer=optimizer,
            cursor=_cursor(),
            binding=binding,
            optimizer_step=1,
            scheduler_step=1,
        )
        return path, model, optimizer, binding

    def test_missing_and_corrupt_artifacts_fail_before_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _, _, binding = self._saved(root / "source")
            cases = {
                "manifest-missing": ("manifest.json", "delete"),
                "model-missing": ("model.safetensors", "delete"),
                "recovery-missing": ("recovery.pt", "delete"),
                "model-corrupt": ("model.safetensors", "append"),
                "recovery-corrupt": ("recovery.pt", "append"),
                "manifest-corrupt": ("manifest.json", "append"),
            }
            for case_name, (filename, operation) in cases.items():
                with self.subTest(case=case_name):
                    target = root / case_name / source.name
                    shutil.copytree(source, target)
                    artifact = target / filename
                    if operation == "delete":
                        artifact.unlink()
                    else:
                        artifact.write_bytes(artifact.read_bytes() + b"x")
                    with self.assertRaises(CheckpointError):
                        validate_checkpoint(target, expected_binding=binding)

    def test_binding_and_format_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _, _, binding = self._saved(root)
            mismatches = {
                "architecture": binding.model_copy(update={"architecture": "mamba2"}),
                "model configuration": binding.model_copy(
                    update={"model_config_sha256": "2" * 64}
                ),
                "tokenizer": binding.model_copy(update={"tokenizer_sha256": "2" * 64}),
                "shard manifest": binding.model_copy(
                    update={"shard_manifest_sha256": "2" * 64}
                ),
                "rank": binding.model_copy(update={"rank": 1}),
                "world size": binding.model_copy(update={"world_size": 2}),
            }
            for message, expected in mismatches.items():
                with (
                    self.subTest(binding=message),
                    self.assertRaisesRegex(CheckpointError, message),
                ):
                    validate_checkpoint(path, expected_binding=expected)

            manifest_path = path / "manifest.json"
            payload = cast(dict[str, object], json.loads(manifest_path.read_text()))
            payload["format_version"] = 2
            manifest_path.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaisesRegex(CheckpointError, "manifest"):
                validate_checkpoint(path)

    def test_save_interruption_preserves_previous_recovery_point(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, model, optimizer, binding = self._saved(root)
            with (
                patch.object(
                    checkpointing,
                    "_publish_directory",
                    side_effect=OSError("simulated interruption"),
                ),
                self.assertRaisesRegex(OSError, "simulated"),
            ):
                save_checkpoint(
                    root,
                    model=model,
                    optimizer=optimizer,
                    cursor=_cursor(),
                    binding=binding,
                    optimizer_step=2,
                    scheduler_step=2,
                    parent_checkpoint=first,
                )

            self.assertEqual(validate_checkpoint(first), validate_checkpoint(first))
            self.assertFalse((root / "step-000000000002").exists())
            self.assertEqual(
                [item.name for item in root.iterdir()],
                ["step-000000000001"],
            )

    def test_duplicate_save_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, model, optimizer, binding = self._saved(root)
            manifest = path / "manifest.json"
            original = manifest.read_bytes()
            duplicate = save_checkpoint(
                root,
                model=model,
                optimizer=optimizer,
                cursor=_cursor(),
                binding=binding,
                optimizer_step=1,
                scheduler_step=1,
            )
            self.assertEqual(duplicate, path)
            self.assertEqual(manifest.read_bytes(), original)


class CheckpointPolicyTests(unittest.TestCase):
    def test_retention_keeps_latest_three_plus_distinct_best(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = _tiny_model()
            optimizer, _ = build_adamw(model, OptimizationConfig(total_steps=10))
            binding = _binding(model)
            parent: Path | None = None
            for step in range(1, 6):
                parent = save_checkpoint(
                    root,
                    model=model,
                    optimizer=optimizer,
                    cursor=_cursor(),
                    binding=binding,
                    optimizer_step=step,
                    scheduler_step=step,
                    is_best=step == 1,
                    parent_checkpoint=parent,
                )

            retained = apply_checkpoint_retention(
                root,
                keep_latest=3,
                best_checkpoint_id="step-000000000001",
            )
            self.assertEqual(
                {path.name for path in retained},
                {
                    "step-000000000001",
                    "step-000000000003",
                    "step-000000000004",
                    "step-000000000005",
                },
            )
            for path in retained:
                validate_checkpoint(path, expected_binding=binding)
            self.assertFalse((root / "step-000000000002").exists())

    def test_step_and_time_triggers_write_each_step_at_most_once(self) -> None:
        cadence = CheckpointCadence(
            last_saved_time_seconds=0,
            step_interval=250,
        )

        self.assertEqual(cadence.due_reasons(249, 899), frozenset())
        self.assertEqual(
            cadence.due_reasons(250, 900),
            frozenset({"step", "time"}),
        )
        cadence.mark_saved(250, 900)
        self.assertEqual(cadence.due_reasons(250, 1_800), frozenset())
        self.assertEqual(
            cadence.due_reasons(251, 1_800),
            frozenset({"time"}),
        )
        cadence.mark_saved(251, 1_800)
        self.assertEqual(cadence.due_reasons(500, 1_801), frozenset({"step"}))

        with self.assertRaisesRegex(ValueError, "backwards"):
            cadence.due_reasons(501, 1_799)

    def test_default_cadence_is_time_only(self) -> None:
        cadence = CheckpointCadence(last_saved_time_seconds=0)

        self.assertEqual(cadence.due_reasons(250, 899), frozenset())
        self.assertEqual(cadence.due_reasons(250, 900), frozenset({"time"}))
        cadence.mark_saved(250, 900)
        self.assertEqual(cadence.due_reasons(250, 1_800), frozenset())
        self.assertEqual(cadence.due_reasons(500, 1_800), frozenset({"time"}))

        with self.assertRaisesRegex(ValueError, "backwards"):
            cadence.due_reasons(501, 899)

    def test_step_interval_must_be_positive_when_enabled(self) -> None:
        CheckpointCadence(last_saved_time_seconds=0, step_interval=None)

        for step_interval in (0, -1):
            with (
                self.subTest(step_interval=step_interval),
                self.assertRaisesRegex(ValueError, "step interval"),
            ):
                CheckpointCadence(
                    last_saved_time_seconds=0,
                    step_interval=step_interval,
                )


if __name__ == "__main__":
    unittest.main()
