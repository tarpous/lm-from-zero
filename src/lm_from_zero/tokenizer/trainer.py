"""Incremental indexed byte-BPE training."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from heapq import heapify, heappop, heappush
from itertools import pairwise
from time import perf_counter

from lm_from_zero.progress import ProgressReporter
from lm_from_zero.tokenizer.bpe import (
    BYTE_TO_TOKEN_ID,
    INITIAL_VOCAB_SIZE,
    MAX_UINT16_VOCAB_SIZE,
    ByteBPE,
    TokenPair,
    _merge_pair,
)
from lm_from_zero.tokenizer.pretokenizer import PretokenizerMode, pretokenize

MergeCallback = Callable[[tuple[TokenPair, ...], int], None]


@dataclass(frozen=True, slots=True)
class BPETrainingStats:
    """Measured corpus and trainer state for one completed invocation."""

    corpus_sha256: str
    document_count: int
    corpus_bytes: int
    segment_count: int
    unique_segment_count: int
    initial_merge_count: int
    final_merge_count: int
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class BPETrainingResult:
    """A trained tokenizer plus its measured training metadata."""

    tokenizer: ByteBPE
    stats: BPETrainingStats


@dataclass(slots=True)
class _SegmentState:
    token_ids: list[int]
    frequency: int


def _segment_frequencies(
    documents: Iterable[bytes | str], mode: PretokenizerMode
) -> tuple[Counter[bytes], str, int, int]:
    frequencies: Counter[bytes] = Counter()
    digest = sha256()
    document_count = 0
    corpus_bytes = 0
    for document in documents:
        data = document if isinstance(document, bytes) else document.encode("utf-8")
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
        document_count += 1
        corpus_bytes += len(data)
        if mode == "none":
            if data:
                frequencies[data] += 1
        else:
            try:
                text = data.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(
                    "gpt2 pre-tokenization requires valid UTF-8"
                ) from error
            frequencies.update(pretokenize(text, mode))
    return frequencies, digest.hexdigest(), document_count, corpus_bytes


class _IndexedTrainer:
    def __init__(self, frequencies: Counter[bytes]) -> None:
        self.segments = [
            _SegmentState(
                token_ids=[BYTE_TO_TOKEN_ID[value] for value in segment],
                frequency=frequency,
            )
            for segment, frequency in sorted(frequencies.items())
        ]
        self.pair_counts: Counter[TokenPair] = Counter()
        self.pair_segments: defaultdict[TokenPair, set[int]] = defaultdict(set)
        for segment_id, state in enumerate(self.segments):
            local_counts = Counter(pairwise(state.token_ids))
            for pair, occurrences in local_counts.items():
                self.pair_counts[pair] += occurrences * state.frequency
                self.pair_segments[pair].add(segment_id)
        self.heap: list[tuple[int, TokenPair]] = []

    def _set_count(self, pair: TokenPair, count: int) -> None:
        if count <= 0:
            self.pair_counts.pop(pair, None)
        else:
            self.pair_counts[pair] = count
            heappush(self.heap, (-count, pair))

    def _apply(self, pair: TokenPair, new_id: int, *, update_heap: bool) -> None:
        segment_ids = sorted(self.pair_segments.pop(pair, ()))
        for segment_id in segment_ids:
            state = self.segments[segment_id]
            old_local = Counter(pairwise(state.token_ids))
            for old_pair, occurrences in old_local.items():
                self.pair_segments[old_pair].discard(segment_id)
                new_count = (
                    self.pair_counts.get(old_pair, 0) - occurrences * state.frequency
                )
                if update_heap:
                    self._set_count(old_pair, new_count)
                elif new_count <= 0:
                    self.pair_counts.pop(old_pair, None)
                else:
                    self.pair_counts[old_pair] = new_count

            state.token_ids = _merge_pair(state.token_ids, pair, new_id)
            new_local = Counter(pairwise(state.token_ids))
            for new_pair, occurrences in new_local.items():
                self.pair_segments[new_pair].add(segment_id)
                new_count = (
                    self.pair_counts.get(new_pair, 0) + occurrences * state.frequency
                )
                if update_heap:
                    self._set_count(new_pair, new_count)
                else:
                    self.pair_counts[new_pair] = new_count

    def replay(self, merges: Sequence[TokenPair]) -> None:
        for merge_index, pair in enumerate(merges):
            if self.pair_counts.get(pair, 0) <= 0:
                raise ValueError(
                    f"replay merge {merge_index} is absent from the corpus"
                )
            self._apply(pair, INITIAL_VOCAB_SIZE + merge_index, update_heap=False)
        self.heap = [(-count, pair) for pair, count in self.pair_counts.items()]
        heapify(self.heap)

    def best_pair(self) -> tuple[TokenPair, int] | None:
        while self.heap:
            negative_count, pair = heappop(self.heap)
            count = -negative_count
            if self.pair_counts.get(pair) == count:
                return pair, count
        return None

    def merge(self, pair: TokenPair, new_id: int) -> None:
        self._apply(pair, new_id, update_heap=True)


def train_bpe(
    documents: Iterable[bytes | str],
    *,
    target_vocab_size: int,
    min_frequency: int = 2,
    pretokenizer: PretokenizerMode = "none",
    initial_merges: Sequence[TokenPair] = (),
    on_merge: MergeCallback | None = None,
) -> BPETrainingResult:
    """Train BPE with indexed pair updates and optional deterministic replay."""

    if not INITIAL_VOCAB_SIZE <= target_vocab_size <= MAX_UINT16_VOCAB_SIZE:
        raise ValueError(
            f"target_vocab_size must be in "
            f"[{INITIAL_VOCAB_SIZE}, {MAX_UINT16_VOCAB_SIZE}]"
        )
    if min_frequency < 1:
        raise ValueError("min_frequency must be at least 1")
    if INITIAL_VOCAB_SIZE + len(initial_merges) > target_vocab_size:
        raise ValueError("initial merges exceed the requested vocabulary size")
    ByteBPE(merges=tuple(initial_merges), pretokenizer=pretokenizer)

    started = perf_counter()
    progress = ProgressReporter("tokenizer training")
    progress.phase("counting corpus")
    frequencies, corpus_hash, document_count, corpus_bytes = _segment_frequencies(
        documents, pretokenizer
    )
    trainer = _IndexedTrainer(frequencies)
    trainer.replay(initial_merges)
    merges = list(initial_merges)
    merge_total = target_vocab_size - INITIAL_VOCAB_SIZE
    progress.phase(
        "merging vocabulary",
        total=merge_total if merge_total > 0 else None,
        current=len(merges),
        fields={"min_frequency": min_frequency},
    )

    while INITIAL_VOCAB_SIZE + len(merges) < target_vocab_size:
        selected = trainer.best_pair()
        if selected is None:
            break
        pair, frequency = selected
        if frequency < min_frequency:
            break
        trainer.merge(pair, INITIAL_VOCAB_SIZE + len(merges))
        merges.append(pair)
        progress.update(len(merges), fields={"pair_frequency": frequency})
        if on_merge is not None:
            on_merge(tuple(merges), frequency)

    tokenizer = ByteBPE(merges=tuple(merges), pretokenizer=pretokenizer)
    stats = BPETrainingStats(
        corpus_sha256=corpus_hash,
        document_count=document_count,
        corpus_bytes=corpus_bytes,
        segment_count=sum(frequencies.values()),
        unique_segment_count=len(frequencies),
        initial_merge_count=len(initial_merges),
        final_merge_count=len(merges),
        elapsed_seconds=perf_counter() - started,
    )
    progress.finish("complete")
    return BPETrainingResult(tokenizer=tokenizer, stats=stats)
