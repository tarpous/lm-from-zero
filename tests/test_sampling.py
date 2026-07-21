from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.data import DataValidationError
from lm_from_zero.sampling import (
    SamplingConfig,
    load_sample,
    sample_text_records,
)


class SamplingTests(unittest.TestCase):
    def test_deterministic_prefix_records_provenance_and_deduplicates(self) -> None:
        records = [
            {"text": "alpha"},
            {"text": ""},
            {"text": "alpha"},
            {"text": "beta"},
            {"text": "gamma"},
        ]
        config = SamplingConfig(target_text_bytes=9, max_storage_bytes=100_000)
        with (
            tempfile.TemporaryDirectory() as first_directory,
            tempfile.TemporaryDirectory() as second_directory,
        ):
            first_path = sample_text_records(records, first_directory, config)
            second_path = sample_text_records(records, second_directory, config)
            first = json.loads(first_path.read_text(encoding="utf-8"))
            second = json.loads(second_path.read_text(encoding="utf-8"))

        self.assertEqual(first, second)
        self.assertEqual(first["document_count"], 2)
        self.assertEqual(first["actual_text_bytes"], 9)
        self.assertEqual(first["duplicate_rows_skipped"], 1)
        self.assertEqual(first["empty_rows_skipped"], 1)
        self.assertEqual(first["next_source_index"], 4)

    def test_load_sample_returns_exact_text_and_checks_all_counts(self) -> None:
        records = [{"text": "naïve"}, {"text": "café"}]
        config = SamplingConfig(target_text_bytes=10, max_storage_bytes=100_000)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = sample_text_records(records, directory, config)
            manifest, texts = load_sample(manifest_path)
            loaded = list(texts)

        self.assertEqual(loaded, ["naïve", "café"])
        self.assertEqual(manifest.document_count, 2)

    def test_non_text_rows_are_rejected_without_partial_files(self) -> None:
        config = SamplingConfig(target_text_bytes=1, max_storage_bytes=100_000)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(DataValidationError, "is not text"):
                sample_text_records([{"text": 3}], directory, config)

            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_short_source_is_rejected(self) -> None:
        config = SamplingConfig(target_text_bytes=10, max_storage_bytes=100_000)
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(DataValidationError, "source ended"),
        ):
            sample_text_records([{"text": "short"}], directory, config)

    def test_storage_limit_is_enforced(self) -> None:
        config = SamplingConfig(target_text_bytes=10, max_storage_bytes=15)
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(DataValidationError, "safety limit"),
        ):
            sample_text_records([{"text": "long enough"}], directory, config)

    def test_existing_artifact_is_not_overwritten(self) -> None:
        config = SamplingConfig(target_text_bytes=1, max_storage_bytes=100_000)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "documents.jsonl"
            path.write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(DataValidationError, "refusing to overwrite"):
                sample_text_records([{"text": "x"}], directory, config)

            self.assertEqual(path.read_text(encoding="utf-8"), "keep")

    def test_artifact_checksum_corruption_is_rejected(self) -> None:
        config = SamplingConfig(target_text_bytes=4, max_storage_bytes=100_000)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = sample_text_records([{"text": "text"}], directory, config)
            sample_path = Path(directory) / "documents.jsonl"
            sample_path.write_text("corrupt", encoding="utf-8")

            with self.assertRaisesRegex(DataValidationError, "checksum mismatch"):
                load_sample(manifest_path)

    def test_config_rejects_non_commit_revision(self) -> None:
        with self.assertRaises(ValidationError):
            SamplingConfig(revision="main")

    def test_config_rejects_impossible_storage_budget(self) -> None:
        with self.assertRaises(ValidationError):
            SamplingConfig(target_text_bytes=10, max_storage_bytes=10)

    def test_verify_sample_cli_checks_every_document(self) -> None:
        config = SamplingConfig(target_text_bytes=9, max_storage_bytes=100_000)
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = sample_text_records(
                [{"text": "alpha"}, {"text": "beta"}], directory, config
            )

            result = CliRunner().invoke(app, ["verify-sample", str(manifest_path)])

        self.assertEqual(result.exit_code, 0, result.output)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "valid")
        self.assertEqual(payload["document_count"], 2)


if __name__ == "__main__":
    unittest.main()
