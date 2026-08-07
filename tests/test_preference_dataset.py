from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from lm_from_zero.post_training.preference_dataset import (
    PREFERENCE_DATASET_REVISION,
    PreferenceDatasetError,
    PreferenceMixManifest,
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


if __name__ == "__main__":
    unittest.main()
