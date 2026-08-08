from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from pydantic import ValidationError
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.post_training.preference_dataset import (
    PREFERENCE_DATASET_REVISION,
    PreferenceDatasetError,
    PreferenceHoldoutManifest,
    PreferenceMixManifest,
    prepare_preference_holdout,
    prepare_preference_mix,
)
from lm_from_zero.tokenizer.bpe import ByteBPE


def _row(index: int, *, invalid: bool = False) -> dict[str, Any]:
    prompt = {"role": "user", "content": f"question {index}"}
    chosen = {"role": "assistant", "content": f"good answer {index}"}
    rejected = {"role": "assistant", "content": f"bad answer {index}"}
    if invalid:
        rejected = {"role": "user", "content": f"bad answer {index}"}
    return {
        "prompt_id": f"prompt-{index}",
        "chosen": [prompt, chosen],
        "rejected": [prompt, rejected],
        "score_chosen": 8.5,
        "score_rejected": 2.0,
    }


class PreferenceDatasetTests(unittest.TestCase):
    def test_prepare_is_deterministic_and_records_provenance(self) -> None:
        rows = [_row(index) for index in range(8)]
        rows.append(_row(99, invalid=True))

        def loader(_: Path) -> list[dict[str, Any]]:
            return rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train_prefs.parquet"
            source.write_bytes(b"test parquet placeholder")
            tokenizer_path = root / "tokenizer.json"
            ByteBPE().save(tokenizer_path)
            first = prepare_preference_mix(
                source,
                root / "first",
                tokenizer_path,
                target_pairs=5,
                selection_seed=2027,
                loader=loader,
            )
            second = prepare_preference_mix(
                source,
                root / "second",
                tokenizer_path,
                target_pairs=5,
                selection_seed=2027,
                loader=loader,
            )
            first_bytes = (root / "first" / "records.jsonl").read_bytes()
            second_bytes = (root / "second" / "records.jsonl").read_bytes()
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(first.records_sha256, second.records_sha256)
            self.assertEqual(first.source_rows, 9)
            self.assertEqual(first.valid_pairs, 8)
            self.assertEqual(first.rejected_rows, 1)
            self.assertEqual(first.selected_pairs, 5)
            self.assertEqual(
                first.records_sha256,
                hashlib.sha256(first_bytes).hexdigest(),
            )
            self.assertEqual(
                (root / "first" / "manifest.json").read_text(encoding="utf-8").strip(),
                first.canonical_json(),
            )
            loaded = PreferenceMixManifest.model_validate_json(
                (root / "first" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded.dataset_revision, PREFERENCE_DATASET_REVISION)
            self.assertTrue(
                all(
                    json.loads(line)["format"] == "lm-from-zero-preference-record"
                    for line in first_bytes.splitlines()
                )
            )

    def test_prepare_rejects_insufficient_valid_pairs_and_invalid_manifest_input(
        self,
    ) -> None:
        def loader(_: Path) -> list[dict[str, Any]]:
            return [_row(1, invalid=True)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train_prefs.parquet"
            source.write_bytes(b"test parquet placeholder")
            tokenizer_path = root / "tokenizer.json"
            ByteBPE().save(tokenizer_path)
            with self.assertRaisesRegex(
                PreferenceDatasetError, "valid preference pairs"
            ):
                prepare_preference_mix(
                    source,
                    root / "output",
                    tokenizer_path,
                    target_pairs=1,
                    loader=loader,
                )
            with self.assertRaisesRegex(PreferenceDatasetError, "does not exist"):
                prepare_preference_mix(
                    root / "missing.parquet",
                    root / "output",
                    tokenizer_path,
                    loader=loader,
                )
            with self.assertRaisesRegex(PreferenceDatasetError, "must be positive"):
                prepare_preference_mix(
                    source,
                    root / "output",
                    tokenizer_path,
                    target_pairs=0,
                    loader=loader,
                )

    def test_manifest_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            PreferenceMixManifest.model_validate(
                {
                    "format": "lm-from-zero-preference-mix-manifest",
                    "format_version": 1,
                    "objective": "dpo",
                }
            )

    def test_holdout_excludes_training_records_deterministically(self) -> None:
        rows = [_row(index) for index in range(8)]

        def loader(_: Path) -> list[dict[str, Any]]:
            return rows

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train_prefs.parquet"
            source.write_bytes(b"test parquet placeholder")
            tokenizer_path = root / "tokenizer.json"
            ByteBPE().save(tokenizer_path)
            train = prepare_preference_mix(
                source,
                root / "train",
                tokenizer_path,
                target_pairs=5,
                selection_seed=2027,
                loader=loader,
            )
            holdout = prepare_preference_holdout(
                source,
                root / "train" / "manifest.json",
                root / "holdout",
                tokenizer_path,
                loader=loader,
            )
            train_keys = {
                (json.loads(line)["source_index"], json.loads(line)["prompt_id"])
                for line in (root / "train" / "records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            }
            holdout_records = [
                json.loads(line)
                for line in (root / "holdout" / "records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(holdout.format, "lm-from-zero-preference-holdout-manifest")
            self.assertEqual(holdout.train_records_sha256, train.records_sha256)
            self.assertEqual(holdout.excluded_pairs, 5)
            self.assertEqual(holdout.selected_pairs, 3)
            self.assertTrue(
                all(
                    (record["source_index"], record["prompt_id"]) not in train_keys
                    for record in holdout_records
                )
            )
            loaded = PreferenceHoldoutManifest.model_validate_json(
                (root / "holdout" / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(loaded.records_sha256, holdout.records_sha256)

    def test_holdout_cli_passes_the_bound_artifacts_to_the_preparer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "train_prefs.parquet"
            training_manifest = root / "train-manifest.json"
            tokenizer = root / "tokenizer.json"
            for path in (source, training_manifest, tokenizer):
                path.write_bytes(b"placeholder")
            prepared = Mock()
            prepared.canonical_json.return_value = "{}"
            with patch(
                "lm_from_zero.post_training.preference_dataset."
                "prepare_preference_holdout",
                return_value=prepared,
            ) as prepare:
                result = CliRunner().invoke(
                    app,
                    [
                        "prepare-dpo-holdout",
                        str(source),
                        str(training_manifest),
                        str(tokenizer),
                        "--output",
                        str(root / "holdout"),
                        "--target-pairs",
                        "3",
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(result.stdout.strip(), "{}")
            prepare.assert_called_once_with(
                source,
                training_manifest,
                root / "holdout",
                tokenizer,
                target_pairs=3,
            )


if __name__ == "__main__":
    unittest.main()
