from __future__ import annotations

import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from typing import cast

import numpy as np
from pydantic import ValidationError
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.data import (
    DataValidationError,
    RankCursor,
    ShardCursor,
    ShardManifest,
    SplitPolicy,
    assign_split,
    document_hash,
    load_manifest,
    split_documents,
    validate_shard,
    validate_shard_directory,
    write_token_shards,
)
from lm_from_zero.tokenizer.bpe import ByteBPE


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = ByteBPE.train(
            [b"alpha beta gamma"], target_vocab_size=270, min_frequency=1
        )

    def _write_train_shards(self, directory: Path) -> tuple[Path, ...]:
        policy = SplitPolicy(validation_buckets=0, test_buckets=0)
        documents = split_documents([b"one", b"two", b"three"], policy)["train"]
        return write_token_shards(
            documents,
            self.tokenizer,
            directory,
            split="train",
            max_tokens_per_shard=8,
            source_name="synthetic",
            source_revision="fixture-v1",
            split_seed=policy.seed,
        )

    def test_split_assignment_is_stable_across_input_order(self) -> None:
        documents = ["alpha", "beta", "gamma", "delta"]
        policy = SplitPolicy(
            seed=9,
            bucket_count=4,
            validation_buckets=1,
            test_buckets=1,
        )

        forward = split_documents(documents, policy)
        reverse = split_documents(reversed(documents), policy)
        forward_map = {
            document.content_hash: split
            for split, items in forward.items()
            for document in items
        }
        reverse_map = {
            document.content_hash: split
            for split, items in reverse.items()
            for document in items
        }

        self.assertEqual(forward_map, reverse_map)

    def test_duplicate_documents_are_rejected(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "duplicate document"):
            split_documents([b"same", b"same"], SplitPolicy())

    def test_invalid_content_hash_is_rejected(self) -> None:
        with self.assertRaisesRegex(DataValidationError, "64 hexadecimal"):
            assign_split("short", SplitPolicy())
        with self.assertRaisesRegex(DataValidationError, "valid hexadecimal"):
            assign_split("z" * 64, SplitPolicy())
        with self.assertRaisesRegex(DataValidationError, "lowercase"):
            assign_split("A" * 64, SplitPolicy())

    def test_split_policy_requires_train_buckets(self) -> None:
        with self.assertRaises(ValidationError):
            SplitPolicy(bucket_count=2, validation_buckets=1, test_buckets=1)

    def test_shards_have_eos_boundaries_and_resume_cursors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifests = self._write_train_shards(Path(directory))
            shards = validate_shard_directory(
                directory,
                expected_tokenizer_hash=self.tokenizer.model_hash,
                expected_vocab_size=self.tokenizer.vocab_size,
            )
            metadata = [load_manifest(path) for path in manifests]

        self.assertEqual(len(manifests), 2)
        self.assertEqual([item.cursor.next_document_index for item in metadata], [2, 3])
        self.assertEqual(sum(int(np.count_nonzero(shard == 2)) for shard in shards), 3)
        self.assertEqual(
            {value for item in metadata for value in item.source_document_hashes},
            {document_hash(b"one"), document_hash(b"two"), document_hash(b"three")},
        )

    def test_wrong_tokenizer_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_train_shards(Path(directory))[0]

            with self.assertRaisesRegex(DataValidationError, "tokenizer hash"):
                validate_shard(manifest, expected_tokenizer_hash="0" * 64)

    def test_truncated_shard_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_train_shards(Path(directory))[0]
            manifest = load_manifest(manifest_path)
            data_path = manifest_path.parent / manifest.data_file
            with data_path.open("r+b") as handle:
                handle.truncate(manifest.byte_count - 2)

            with self.assertRaisesRegex(DataValidationError, "byte length"):
                validate_shard(manifest_path)

    def test_checksum_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_train_shards(Path(directory))[0]
            manifest = load_manifest(manifest_path)
            data_path = manifest_path.parent / manifest.data_file
            with data_path.open("r+b") as handle:
                first = handle.read(1)
                handle.seek(0)
                handle.write(bytes((first[0] ^ 1,)))

            with self.assertRaisesRegex(DataValidationError, "checksum"):
                validate_shard(manifest_path)

    def test_invalid_token_id_is_rejected_after_integrity_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest_path = self._write_train_shards(Path(directory))[0]
            manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = load_manifest(manifest_path)
            data_path = manifest_path.parent / manifest.data_file
            tokens = np.memmap(data_path, dtype="<u2", mode="r+")
            tokens[0] = self.tokenizer.vocab_size
            tokens.flush()
            del tokens
            manifest_payload["sha256"] = sha256(data_path.read_bytes()).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest_payload),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DataValidationError, "outside the vocabulary"):
                validate_shard(
                    manifest_path,
                    expected_vocab_size=self.tokenizer.vocab_size,
                )

    def test_incomplete_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "train-00000.json"
            manifest.write_text('{"status":"complete"}', encoding="utf-8")

            with self.assertRaisesRegex(DataValidationError, "invalid shard manifest"):
                load_manifest(manifest)

    def test_orphan_and_partial_artifacts_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            orphan = Path(directory) / "train-00000.bin"
            orphan.write_bytes(b"\x00\x00")
            with self.assertRaisesRegex(DataValidationError, "orphaned"):
                validate_shard_directory(directory)

        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / ".train-00000.bin.partial"
            partial.write_bytes(b"\x00\x00")
            with self.assertRaisesRegex(DataValidationError, "incomplete shard"):
                validate_shard_directory(directory)

    def test_shard_set_rejects_cross_manifest_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifests = self._write_train_shards(Path(directory))
            first = load_manifest(manifests[0])
            second_payload = json.loads(manifests[1].read_text(encoding="utf-8"))
            second_payload["source_document_hashes"] = [first.source_document_hashes[0]]
            manifests[1].write_text(json.dumps(second_payload), encoding="utf-8")

            with self.assertRaisesRegex(DataValidationError, "more than one shard"):
                validate_shard_directory(directory)

    def test_shard_set_rejects_mixed_tokenizers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifests = self._write_train_shards(Path(directory))
            second_payload = json.loads(manifests[1].read_text(encoding="utf-8"))
            second_payload["tokenizer_hash"] = "0" * 64
            manifests[1].write_text(json.dumps(second_payload), encoding="utf-8")

            with self.assertRaisesRegex(DataValidationError, "mixed tokenizer"):
                validate_shard_directory(directory)

    def test_manifest_rejects_duplicate_document_hashes(self) -> None:
        digest = document_hash(b"duplicate")
        with self.assertRaises(ValidationError):
            ShardManifest(
                data_file="train-00000.bin",
                split="train",
                token_count=1,
                byte_count=2,
                sha256="0" * 64,
                tokenizer_hash="1" * 64,
                source_name="synthetic",
                source_revision="fixture-v1",
                source_document_hashes=(digest, digest),
                split_seed=1337,
                cursor=ShardCursor(next_document_index=2),
            )

    def test_rank_cursors_cover_indices_without_overlap_and_resume(self) -> None:
        first = RankCursor(rank=0, world_size=2)
        second = RankCursor(rank=1, world_size=2)

        first_indices, first_resumed = first.take(3)
        second_indices, second_resumed = second.take(3)
        next_indices, _ = first_resumed.take(2)

        self.assertEqual(first_indices, (0, 2, 4))
        self.assertEqual(second_indices, (1, 3, 5))
        self.assertEqual(next_indices, (6, 8))
        self.assertEqual(
            first_resumed,
            RankCursor.model_validate_json(first_resumed.model_dump_json()),
        )
        self.assertEqual(second_resumed.next_local_index, 3)

    def test_rank_cursor_validation(self) -> None:
        with self.assertRaises(ValidationError):
            RankCursor(rank=2, world_size=2)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            RankCursor(rank=0, world_size=1).take(-1)

    def test_document_larger_than_shard_is_rejected(self) -> None:
        policy = SplitPolicy(validation_buckets=0, test_buckets=0)
        documents = split_documents([b"too long"], policy)["train"]
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(DataValidationError, "exceeds"),
        ):
            write_token_shards(
                documents,
                self.tokenizer,
                directory,
                split="train",
                max_tokens_per_shard=1,
                source_name="synthetic",
                source_revision="fixture-v1",
                split_seed=policy.seed,
            )

    def test_verify_shard_cli_outputs_checked_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = self._write_train_shards(Path(directory))[0]

            result = CliRunner().invoke(
                app,
                [
                    "verify-shard",
                    str(manifest),
                    "--tokenizer-hash",
                    self.tokenizer.model_hash,
                    "--vocab-size",
                    str(self.tokenizer.vocab_size),
                ],
            )

        self.assertEqual(result.exit_code, 0, cast(str, result.exception))
        payload = json.loads(result.stdout)
        self.assertEqual(payload["status"], "valid")
        self.assertEqual(payload["split"], "train")


if __name__ == "__main__":
    unittest.main()
