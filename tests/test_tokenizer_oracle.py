from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lm_from_zero.sampling import SamplingConfig, sample_text_records
from lm_from_zero.tokenizer.bpe import INITIAL_VOCAB_SIZE
from lm_from_zero.tokenizer.oracle import verify_hugging_face_oracle
from lm_from_zero.tokenizer.pipeline import (
    TokenizerTrainingConfig,
    train_tokenizer_from_sample,
)


class TokenizerOracleTests(unittest.TestCase):
    def test_oracle_matches_project_merges_and_encoding_ids(self) -> None:
        records = [
            {"text": "Hello world! I'm here."},
            {"text": "Hello there; you're here."},
            {"text": "Numbers 42 and 4242."},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = sample_text_records(
                records,
                root / "sample",
                SamplingConfig(target_text_bytes=50, max_storage_bytes=100_000),
            )
            output = root / "tokenizer"
            train_tokenizer_from_sample(
                sample_path,
                output,
                TokenizerTrainingConfig(
                    target_vocab_size=INITIAL_VOCAB_SIZE + 12,
                    min_frequency=1,
                ),
            )

            result = verify_hugging_face_oracle(
                output / "training.json",
                sample_path,
            )

        self.assertTrue(result.merge_order_equal)
        self.assertTrue(result.encoding_ids_equal)
        self.assertEqual(result.merge_count, 12)


if __name__ == "__main__":
    unittest.main()
