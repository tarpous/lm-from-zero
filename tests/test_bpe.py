from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from itertools import pairwise
from pathlib import Path
from typing import cast

from hypothesis import given
from hypothesis import strategies as st
from tokenizers import (  # type: ignore[import-untyped]
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
)

from lm_from_zero.tokenizer.bpe import (
    BYTE_TO_TOKEN_ID,
    BYTE_TOKEN_OFFSET,
    INITIAL_VOCAB_SIZE,
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKENS,
    ByteBPE,
    _merge_pair,
)
from lm_from_zero.tokenizer.pretokenizer import pretokenize
from lm_from_zero.tokenizer.trainer import train_bpe

FUZZ_TOKENIZER = ByteBPE.train(
    [b"banana bandana", b"the quick brown fox", bytes(range(256))],
    target_vocab_size=INITIAL_VOCAB_SIZE + 32,
    min_frequency=1,
)


def _byte_level_decoder() -> dict[str, int]:
    visible_bytes = list(range(ord("!"), ord("~") + 1))
    visible_bytes.extend(range(ord("¡"), ord("¬") + 1))
    visible_bytes.extend(range(ord("®"), ord("ÿ") + 1))
    byte_values = list(visible_bytes)
    code_points = list(visible_bytes)
    extra_code_point = 256
    for byte_value in range(256):
        if byte_value not in visible_bytes:
            byte_values.append(byte_value)
            code_points.append(extra_code_point)
            extra_code_point += 1
    return {
        chr(code_point): byte_value
        for byte_value, code_point in zip(byte_values, code_points, strict=True)
    }


BYTE_LEVEL_DECODER = _byte_level_decoder()


def _decode_hf_symbol(symbol: str) -> bytes:
    return bytes(BYTE_LEVEL_DECODER[character] for character in symbol)


def _train_hf_oracle(
    documents: list[str], vocab_size: int, *, use_regex: bool = False
) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False,
        use_regex=use_regex,
    )
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        show_progress=False,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(documents, trainer=trainer)
    return tokenizer


def _hf_merge_bytes(tokenizer: Tokenizer) -> list[tuple[bytes, bytes]]:
    with tempfile.TemporaryDirectory() as directory:
        tokenizer.model.save(directory)
        merge_lines = (Path(directory) / "merges.txt").read_text(encoding="utf-8")
    result: list[tuple[bytes, bytes]] = []
    for line in merge_lines.splitlines()[1:]:
        left, right = line.split()
        result.append((_decode_hf_symbol(left), _decode_hf_symbol(right)))
    return result


def _naive_merges(
    documents: list[bytes], merge_count: int
) -> tuple[tuple[int, int], ...]:
    sequences = [
        [BYTE_TO_TOKEN_ID[value] for value in document] for document in documents
    ]
    merges: list[tuple[int, int]] = []
    for merge_index in range(merge_count):
        counts: Counter[tuple[int, int]] = Counter()
        for sequence in sequences:
            counts.update(pairwise(sequence))
        if not counts:
            break
        pair, _ = min(counts.items(), key=lambda item: (-item[1], item[0]))
        new_id = INITIAL_VOCAB_SIZE + merge_index
        sequences = [_merge_pair(sequence, pair, new_id) for sequence in sequences]
        merges.append(pair)
    return tuple(merges)


class ByteBPETests(unittest.TestCase):
    @given(
        st.lists(st.binary(max_size=12), min_size=1, max_size=8),
        st.integers(min_value=0, max_value=8),
    )
    def test_indexed_trainer_matches_naive_reference(
        self, documents: list[bytes], merge_count: int
    ) -> None:
        tokenizer = ByteBPE.train(
            documents,
            target_vocab_size=INITIAL_VOCAB_SIZE + merge_count,
            min_frequency=1,
        )

        self.assertEqual(tokenizer.merges, _naive_merges(documents, merge_count))

    @given(st.binary(max_size=2048))
    def test_fuzzed_arbitrary_bytes_round_trip(self, payload: bytes) -> None:
        encoded = FUZZ_TOKENIZER.encode_bytes(payload)

        self.assertEqual(FUZZ_TOKENIZER.decode_bytes(encoded), payload)

    @given(st.text(max_size=512))
    def test_fuzzed_unicode_text_round_trip(self, text: str) -> None:
        encoded = FUZZ_TOKENIZER.encode(text)

        self.assertEqual(FUZZ_TOKENIZER.decode(encoded), text)

    def test_all_byte_values_round_trip_without_unknown_token(self) -> None:
        tokenizer = ByteBPE.train(
            [bytes(range(256)) * 2],
            target_vocab_size=INITIAL_VOCAB_SIZE + 32,
            min_frequency=1,
        )
        payload = bytes(range(256)) * 3

        encoded = tokenizer.encode_bytes(payload)

        self.assertTrue(
            all(BYTE_TOKEN_OFFSET <= item < tokenizer.vocab_size for item in encoded)
        )
        self.assertEqual(tokenizer.decode_bytes(encoded), payload)

    def test_training_is_deterministic(self) -> None:
        documents = [b"banana bandana", b"ananas", b"banana"]

        first = ByteBPE.train(
            documents, target_vocab_size=INITIAL_VOCAB_SIZE + 20, min_frequency=1
        )
        second = ByteBPE.train(
            documents, target_vocab_size=INITIAL_VOCAB_SIZE + 20, min_frequency=1
        )

        self.assertEqual(first.merges, second.merges)
        self.assertEqual(first.model_hash, second.model_hash)

    def test_equal_frequency_pairs_use_lexicographic_tie_break(self) -> None:
        tokenizer = ByteBPE.train(
            [b"abac"], target_vocab_size=INITIAL_VOCAB_SIZE + 1, min_frequency=1
        )

        self.assertEqual(
            tokenizer.merges,
            ((ByteBPE().encode_bytes(b"a")[0], ByteBPE().encode_bytes(b"b")[0]),),
        )

    def test_hugging_face_oracle_matches_merges_and_segmentation(self) -> None:
        documents = ["banana bandana", "ananas", "banana"]
        target_vocab_size = INITIAL_VOCAB_SIZE + 6
        tokenizer = ByteBPE.train(
            documents,
            target_vocab_size=target_vocab_size,
            min_frequency=1,
        )
        oracle = _train_hf_oracle(documents, target_vocab_size)

        scratch_merges = [
            (tokenizer.token_bytes(left), tokenizer.token_bytes(right))
            for left, right in tokenizer.merges
        ]

        self.assertEqual(_hf_merge_bytes(oracle), scratch_merges)
        for token, token_id in SPECIAL_TOKEN_IDS.items():
            self.assertEqual(oracle.token_to_id(token), token_id)

        held_out = "a banana bandana"
        scratch_segments = [
            tokenizer.token_bytes(token_id) for token_id in tokenizer.encode(held_out)
        ]
        oracle_segments = [
            _decode_hf_symbol(symbol) for symbol in oracle.encode(held_out).tokens
        ]
        self.assertEqual(oracle_segments, scratch_segments)

    def test_gpt2_pretokenizer_matches_bytelevel_oracle(self) -> None:
        text = "Hello   world! I'm naïve 42.\nTabs\there Ⅷ"
        oracle = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        oracle_chunks = tuple(
            _decode_hf_symbol(symbol) for symbol, _ in oracle.pre_tokenize_str(text)
        )

        self.assertEqual(pretokenize(text, "gpt2"), oracle_chunks)

    def test_gpt2_training_and_encoding_match_oracle(self) -> None:
        documents = [
            "Hello world! I'm here.",
            "Hello there; you're here.",
            "Numbers 42 and 4242.",
        ]
        target_vocab_size = INITIAL_VOCAB_SIZE + 20
        tokenizer = train_bpe(
            documents,
            target_vocab_size=target_vocab_size,
            min_frequency=1,
            pretokenizer="gpt2",
        ).tokenizer
        oracle = _train_hf_oracle(
            documents,
            target_vocab_size,
            use_regex=True,
        )

        scratch_merges = [
            (tokenizer.token_bytes(left), tokenizer.token_bytes(right))
            for left, right in tokenizer.merges
        ]
        self.assertEqual(_hf_merge_bytes(oracle), scratch_merges)
        self.assertEqual(
            tokenizer.encode("Hello, you're here 42!"),
            oracle.encode("Hello, you're here 42!").ids,
        )

    def test_replayed_training_matches_one_shot_training(self) -> None:
        documents = [
            "one fish two fish",
            "red fish blue fish",
            "one blue bird",
        ]
        partial = train_bpe(
            documents,
            target_vocab_size=INITIAL_VOCAB_SIZE + 8,
            min_frequency=1,
            pretokenizer="gpt2",
        )
        resumed = train_bpe(
            documents,
            target_vocab_size=INITIAL_VOCAB_SIZE + 16,
            min_frequency=1,
            pretokenizer="gpt2",
            initial_merges=partial.tokenizer.merges,
        )
        one_shot = train_bpe(
            documents,
            target_vocab_size=INITIAL_VOCAB_SIZE + 16,
            min_frequency=1,
            pretokenizer="gpt2",
        )

        self.assertEqual(resumed.tokenizer.merges, one_shot.tokenizer.merges)
        self.assertEqual(resumed.stats.initial_merge_count, 8)
        self.assertEqual(resumed.stats.corpus_sha256, one_shot.stats.corpus_sha256)

    def test_gpt2_training_rejects_invalid_utf8_bytes(self) -> None:
        with self.assertRaisesRegex(ValueError, "valid UTF-8"):
            train_bpe(
                [b"\xff"],
                target_vocab_size=INITIAL_VOCAB_SIZE,
                pretokenizer="gpt2",
            )

    def test_merges_do_not_cross_document_boundaries(self) -> None:
        tokenizer = ByteBPE.train(
            [b"a", b"b"], target_vocab_size=INITIAL_VOCAB_SIZE + 1, min_frequency=1
        )

        self.assertEqual(tokenizer.merges, ())

    def test_special_token_parsing_is_opt_in(self) -> None:
        tokenizer = ByteBPE()
        text = "before<|bos|>after"

        ordinary = tokenizer.encode(text)
        controlled = tokenizer.encode(text, allowed_special={"<|bos|>"})

        self.assertNotIn(SPECIAL_TOKEN_IDS["<|bos|>"], ordinary)
        self.assertIn(SPECIAL_TOKEN_IDS["<|bos|>"], controlled)
        self.assertEqual(tokenizer.decode(ordinary), text)
        self.assertEqual(tokenizer.decode(controlled, render_special=True), text)
        with self.assertRaisesRegex(ValueError, "special token encountered"):
            tokenizer.decode(controlled)

    def test_unknown_allowed_special_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown special token"):
            ByteBPE().encode("text", allowed_special={"<|unknown|>"})

    def test_save_load_preserves_encoding_and_hash(self) -> None:
        tokenizer = ByteBPE.train(
            ["naïve café", "café café"],
            target_vocab_size=INITIAL_VOCAB_SIZE + 12,
            min_frequency=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            tokenizer.save(path)
            restored = ByteBPE.load(path)

        self.assertEqual(restored, tokenizer)
        self.assertEqual(restored.model_hash, tokenizer.model_hash)
        self.assertEqual(restored.encode("naïve café"), tokenizer.encode("naïve café"))

    def test_modified_special_mapping_is_rejected(self) -> None:
        payload = ByteBPE().to_dict()
        special_tokens = dict(cast(dict[str, int], payload["special_tokens"]))
        special_tokens["<|pad|>"] = 9
        payload["special_tokens"] = special_tokens

        with self.assertRaisesRegex(ValueError, "special-token mapping"):
            ByteBPE.from_dict(payload)

    def test_merge_cannot_reference_future_token(self) -> None:
        with self.assertRaisesRegex(ValueError, "earlier merges"):
            ByteBPE(merges=((INITIAL_VOCAB_SIZE, BYTE_TOKEN_OFFSET),))

    def test_invalid_token_id_is_rejected(self) -> None:
        tokenizer = ByteBPE()

        with self.assertRaisesRegex(ValueError, "outside the vocabulary"):
            tokenizer.decode_bytes([tokenizer.vocab_size])

    def test_invalid_training_limits_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_vocab_size"):
            ByteBPE.train([], target_vocab_size=INITIAL_VOCAB_SIZE - 1)
        with self.assertRaisesRegex(ValueError, "min_frequency"):
            ByteBPE.train([], target_vocab_size=INITIAL_VOCAB_SIZE, min_frequency=0)

    def test_corrupt_json_is_reported_as_model_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"
            path.write_text("{broken", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cannot load tokenizer model"):
                ByteBPE.load(path)

    def test_serialization_is_canonical_json(self) -> None:
        tokenizer = ByteBPE()

        parsed = json.loads(tokenizer.canonical_json())

        self.assertEqual(parsed, tokenizer.to_dict())
        self.assertNotIn(" ", tokenizer.canonical_json())


if __name__ == "__main__":
    unittest.main()
