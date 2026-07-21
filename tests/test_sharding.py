from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.data import (
    DataValidationError,
    SplitPolicy,
    assign_split,
    document_hash,
)
from lm_from_zero.sampling import SamplingConfig, sample_text_records
from lm_from_zero.sharding import build_token_shards, validate_shard_build
from lm_from_zero.tokenizer.bpe import INITIAL_VOCAB_SIZE
from lm_from_zero.tokenizer.pipeline import (
    TokenizerTrainingConfig,
    train_tokenizer_from_sample,
)


class ShardBuildTests(unittest.TestCase):
    policy = SplitPolicy(
        seed=7,
        bucket_count=3,
        validation_buckets=1,
        test_buckets=1,
    )

    def _artifacts(self, root: Path) -> tuple[Path, Path]:
        by_split: dict[str, list[str]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        candidate = 0
        while any(len(items) < 2 for items in by_split.values()):
            text = f"small deterministic story number {candidate}."
            split = assign_split(document_hash(text), self.policy)
            if len(by_split[split]) < 2:
                by_split[split].append(text)
            candidate += 1
        texts = [text for items in by_split.values() for text in items]
        sample_path = sample_text_records(
            ({"text": text} for text in texts),
            root / "sample",
            SamplingConfig(
                target_text_bytes=sum(len(text.encode("utf-8")) for text in texts),
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
        return sample_path, tokenizer_directory / "training.json"

    def test_build_is_atomic_complete_and_revalidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path, training_path = self._artifacts(root)
            output = root / "build"

            built = build_token_shards(
                sample_path,
                training_path,
                output,
                split_policy=self.policy,
                max_tokens_per_shard=100,
            )
            validated = validate_shard_build(output / "build.json")

            self.assertTrue(output.is_dir())
            self.assertFalse(root.joinpath(".build.partial").exists())
            self.assertEqual(built, validated)
            self.assertEqual([item.document_count for item in built.splits], [2, 2, 2])
            self.assertGreater(built.total_token_count, built.source_document_count)

    def test_build_rejects_mismatched_tokenizer_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path, training_path = self._artifacts(root)
            payload = json.loads(training_path.read_text(encoding="utf-8"))
            payload["source_sample_sha256"] = "0" * 64
            training_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(DataValidationError, "source does not match"):
                build_token_shards(
                    sample_path,
                    training_path,
                    root / "build",
                    split_policy=self.policy,
                )

    def test_validation_rejects_tampered_split_totals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path, training_path = self._artifacts(root)
            output = root / "build"
            build_token_shards(
                sample_path,
                training_path,
                output,
                split_policy=self.policy,
            )
            build_path = output / "build.json"
            payload = json.loads(build_path.read_text(encoding="utf-8"))
            payload["splits"][0]["token_count"] += 1
            payload["total_token_count"] += 1
            build_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(DataValidationError, "split token count"):
                validate_shard_build(build_path)

    def test_failed_build_does_not_publish_destination(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path, training_path = self._artifacts(root)
            output = root / "build"

            with self.assertRaisesRegex(DataValidationError, "exceeds"):
                build_token_shards(
                    sample_path,
                    training_path,
                    output,
                    split_policy=self.policy,
                    max_tokens_per_shard=1,
                )

            self.assertFalse(output.exists())
            self.assertTrue(root.joinpath(".build.partial").is_dir())

    def test_cli_builds_and_verifies_shard_set(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path, training_path = self._artifacts(root)
            output = root / "build"
            runner = CliRunner()

            build_result = runner.invoke(
                app,
                [
                    "build-token-shards",
                    str(sample_path),
                    str(training_path),
                    "--output-directory",
                    str(output),
                    "--split-seed",
                    "7",
                    "--bucket-count",
                    "3",
                    "--validation-buckets",
                    "1",
                    "--test-buckets",
                    "1",
                ],
            )
            verify_result = runner.invoke(
                app,
                ["verify-shard-build", str(output / "build.json")],
            )

        self.assertEqual(build_result.exit_code, 0, cast(str, build_result.exception))
        self.assertEqual(verify_result.exit_code, 0, cast(str, verify_result.exception))
        self.assertEqual(json.loads(verify_result.stdout)["status"], "valid")


if __name__ == "__main__":
    unittest.main()
