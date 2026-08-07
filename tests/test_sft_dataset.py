from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from lm_from_zero.post_training.dataset import (
    DATASET_REVISION,
    SFTDatasetError,
    SFTMixManifest,
    SFTSourceSpec,
    allocate_sft_counts,
    prepare_sft_mix,
)


def _row(*, role: str = "assistant") -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": "prompt"},
            {"role": role, "content": "answer"},
        ]
    }


class SFTDatasetTests(unittest.TestCase):
    def test_weighted_allocation_is_exact_and_stable(self) -> None:
        sources = (
            SFTSourceSpec(split="b", available_examples=20, weight=0.5),
            SFTSourceSpec(split="a", available_examples=10, weight=1.0),
        )
        self.assertEqual(allocate_sft_counts(sources, 20), {"b": 10, "a": 10})
        self.assertEqual(allocate_sft_counts(sources, 1), {"b": 0, "a": 1})
        with self.assertRaisesRegex(SFTDatasetError, "sources"):
            allocate_sft_counts((), 1)
        with self.assertRaisesRegex(SFTDatasetError, "positive"):
            allocate_sft_counts(sources, 0)
        with self.assertRaisesRegex(SFTDatasetError, "capacity"):
            allocate_sft_counts(sources, 31)

    def test_prepare_writes_canonical_mix_and_rejects_invalid_rows(self) -> None:
        calls: list[str] = []

        def loader(split: str) -> Iterator[dict[str, Any]]:
            calls.append(split)
            if split == "hermes_function_calling_v1_no_think":
                yield _row(role="tool")
            while True:
                yield _row()

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = prepare_sft_mix(
                output,
                target_examples=100,
                selection_seed=2027,
                loader=loader,
            )
            records_path = output / "records.jsonl"
            manifest_path = output / "manifest.json"
            lines = records_path.read_bytes().splitlines(keepends=True)
            self.assertEqual(len(lines), 100)
            self.assertEqual(manifest.selected_examples, 100)
            self.assertEqual(manifest.selection_seed, 2027)
            self.assertEqual(
                manifest.records_sha256,
                hashlib.sha256(records_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                manifest_path.read_text(encoding="utf-8").strip(),
                manifest.canonical_json(),
            )
            loaded = SFTMixManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(loaded.dataset_revision, DATASET_REVISION)
            self.assertIn("hermes_function_calling_v1_no_think", calls)
            hermes = next(
                item
                for item in manifest.sources
                if item.split == "hermes_function_calling_v1_no_think"
            )
            self.assertEqual(hermes.rejected_examples, 1)
            self.assertTrue(
                all(
                    json.loads(line)["format"] == "lm-from-zero-sft-record"
                    for line in lines
                )
            )

    def test_prepare_rejects_missing_valid_source_rows(self) -> None:
        def empty_loader(split: str) -> Iterator[dict[str, Any]]:
            if split == "smoltalk_smollm3_everyday_conversations_no_think":
                return iter(())
            return iter((_row(),))

        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(SFTDatasetError, "yielded"),
        ):
            prepare_sft_mix(directory, target_examples=100, loader=empty_loader)


if __name__ == "__main__":
    unittest.main()
