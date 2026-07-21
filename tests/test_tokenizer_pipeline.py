from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.data import DataValidationError
from lm_from_zero.sampling import SamplingConfig, load_sample, sample_text_records
from lm_from_zero.tokenizer.bpe import INITIAL_VOCAB_SIZE, ByteBPE
from lm_from_zero.tokenizer.pipeline import (
    TokenizerTrainingConfig,
    load_training_manifest,
    train_tokenizer_from_sample,
)
from lm_from_zero.tokenizer.trainer import train_bpe


class TokenizerPipelineTests(unittest.TestCase):
    def _sample(self, directory: Path) -> tuple[Path, list[str]]:
        records = [
            {"text": "one fish two fish"},
            {"text": "red fish blue fish"},
            {"text": "one blue bird"},
            {"text": "two red birds"},
        ]
        manifest_path = sample_text_records(
            records,
            directory,
            SamplingConfig(target_text_bytes=50, max_storage_bytes=100_000),
        )
        _, iterator = load_sample(manifest_path)
        return manifest_path, list(iterator)

    def test_interrupted_checkpoint_replays_to_one_shot_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path, texts = self._sample(root / "sample")
            output = root / "tokenizer"
            config = TokenizerTrainingConfig(
                target_vocab_size=INITIAL_VOCAB_SIZE + 16,
                min_frequency=1,
                checkpoint_every_merges=3,
            )

            paused = train_tokenizer_from_sample(
                sample_path,
                output,
                config,
                stop_after_new_merges=5,
            )
            resumed = train_tokenizer_from_sample(sample_path, output, config)
            expected = train_bpe(
                texts,
                target_vocab_size=config.target_vocab_size,
                min_frequency=config.min_frequency,
                pretokenizer=config.pretokenizer,
            )

            final_tokenizer = ByteBPE.load(output / "tokenizer.json")

        self.assertEqual(paused.status, "in_progress")
        self.assertEqual(paused.merge_count, 5)
        self.assertEqual(resumed.status, "complete")
        self.assertEqual(resumed.resumed_from_merge_count, 5)
        self.assertEqual(final_tokenizer.merges, expected.tokenizer.merges)
        self.assertEqual(resumed.corpus_sha256, expected.stats.corpus_sha256)

    def test_completed_training_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path, _ = self._sample(root / "sample")
            output = root / "tokenizer"
            config = TokenizerTrainingConfig(
                target_vocab_size=INITIAL_VOCAB_SIZE + 4,
                min_frequency=1,
            )
            first = train_tokenizer_from_sample(sample_path, output, config)
            second = train_tokenizer_from_sample(sample_path, output, config)

        self.assertEqual(first, second)

    def test_resume_rejects_changed_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path, _ = self._sample(root / "sample")
            output = root / "tokenizer"
            original = TokenizerTrainingConfig(
                target_vocab_size=INITIAL_VOCAB_SIZE + 8,
                min_frequency=1,
            )
            train_tokenizer_from_sample(
                sample_path,
                output,
                original,
                stop_after_new_merges=2,
            )
            changed = original.model_copy(update={"min_frequency": 2})

            with self.assertRaisesRegex(DataValidationError, "configuration"):
                train_tokenizer_from_sample(sample_path, output, changed)

    def test_resume_rejects_corrupt_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path, _ = self._sample(root / "sample")
            output = root / "tokenizer"
            config = TokenizerTrainingConfig(
                target_vocab_size=INITIAL_VOCAB_SIZE + 8,
                min_frequency=1,
            )
            train_tokenizer_from_sample(
                sample_path,
                output,
                config,
                stop_after_new_merges=2,
            )
            (output / "tokenizer.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unrecognized tokenizer"):
                train_tokenizer_from_sample(sample_path, output, config)

    def test_training_manifest_rejects_incomplete_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.json"
            path.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(DataValidationError, "invalid tokenizer"):
                load_training_manifest(path)

    def test_stop_count_must_be_positive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path, _ = self._sample(root / "sample")

            with self.assertRaisesRegex(ValueError, "must be positive"):
                train_tokenizer_from_sample(
                    sample_path,
                    root / "tokenizer",
                    TokenizerTrainingConfig(),
                    stop_after_new_merges=0,
                )

    def test_train_tokenizer_cli_writes_complete_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path, _ = self._sample(root / "sample")
            output = root / "tokenizer"

            result = CliRunner().invoke(
                app,
                [
                    "train-tokenizer",
                    str(sample_path),
                    "--output-directory",
                    str(output),
                    "--target-vocab-size",
                    str(INITIAL_VOCAB_SIZE + 4),
                    "--min-frequency",
                    "1",
                ],
            )

        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn('"status":"complete"', result.stdout)


if __name__ == "__main__":
    unittest.main()
