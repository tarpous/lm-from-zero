from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lm_from_zero.sampling import SamplingConfig, sample_text_records
from lm_from_zero.tokenizer.benchmark import append_benchmark, benchmark_encoding
from lm_from_zero.tokenizer.bpe import ByteBPE


class TokenizerBenchmarkTests(unittest.TestCase):
    def test_benchmark_measures_and_appends_canonical_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = sample_text_records(
                [{"text": "alpha beta"}, {"text": "gamma delta"}],
                root / "sample",
                SamplingConfig(target_text_bytes=20, max_storage_bytes=100_000),
            )
            tokenizer = ByteBPE.train(
                ["alpha beta", "gamma delta"],
                target_vocab_size=270,
                min_frequency=1,
                pretokenizer="gpt2",
            )
            tokenizer_path = root / "tokenizer.json"
            tokenizer.save(tokenizer_path)

            benchmark = benchmark_encoding(
                tokenizer_path,
                sample_path,
                max_text_bytes=5,
                warmup_text_bytes=0,
            )
            output = root / "benchmarks.jsonl"
            append_benchmark(output, benchmark)

            lines = output.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 1)
        self.assertGreater(benchmark.megabytes_per_second, 0)
        self.assertGreater(benchmark.tokens_per_second, 0)
        self.assertEqual(benchmark.document_count, 1)

    def test_benchmark_validates_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            benchmark_encoding("missing", "missing", max_text_bytes=0)
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            benchmark_encoding(
                "missing",
                "missing",
                max_text_bytes=1,
                warmup_text_bytes=-1,
            )


if __name__ == "__main__":
    unittest.main()
