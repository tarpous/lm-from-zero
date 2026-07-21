"""Measured tokenizer encoding benchmarks."""

from __future__ import annotations

import gc
import json
import os
import platform
import tracemalloc
from pathlib import Path
from time import perf_counter
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lm_from_zero.sampling import load_sample
from lm_from_zero.tokenizer.bpe import ByteBPE


class EncodingBenchmark(BaseModel):
    """Machine-readable outcome of a local pure-Python encoding run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tokenizer_hash: str
    sample_sha256: str
    python_version: str
    platform: str
    document_count: Annotated[int, Field(gt=0)]
    text_bytes: Annotated[int, Field(gt=0)]
    token_count: Annotated[int, Field(gt=0)]
    elapsed_seconds: Annotated[float, Field(gt=0)]
    megabytes_per_second: Annotated[float, Field(gt=0)]
    tokens_per_second: Annotated[float, Field(gt=0)]
    peak_traced_bytes: Annotated[int | None, Field(ge=0)]

    @model_validator(mode="after")
    def validate_rates(self) -> Self:
        expected_mb = self.text_bytes / 1_000_000 / self.elapsed_seconds
        expected_tokens = self.token_count / self.elapsed_seconds
        if abs(self.megabytes_per_second - expected_mb) > 1e-9:
            raise ValueError("megabytes_per_second does not match measured fields")
        if abs(self.tokens_per_second - expected_tokens) > 1e-6:
            raise ValueError("tokens_per_second does not match measured fields")
        return self


def benchmark_encoding(
    tokenizer_path: str | Path,
    sample_manifest_path: str | Path,
    *,
    max_text_bytes: int,
    warmup_text_bytes: int = 100_000,
    trace_memory: bool = False,
) -> EncodingBenchmark:
    """Benchmark encoding whole documents after a bounded warm-up."""

    if max_text_bytes <= 0:
        raise ValueError("max_text_bytes must be positive")
    if warmup_text_bytes < 0:
        raise ValueError("warmup_text_bytes cannot be negative")
    tokenizer = ByteBPE.load(tokenizer_path)
    sample, texts = load_sample(sample_manifest_path)

    warmed_bytes = 0
    while warmed_bytes < warmup_text_bytes:
        try:
            text = next(texts)
        except StopIteration as error:
            raise ValueError("sample ended during benchmark warm-up") from error
        tokenizer.encode(text)
        warmed_bytes += len(text.encode("utf-8"))

    gc.collect()
    if trace_memory:
        tracemalloc.start()
    started = perf_counter()
    document_count = 0
    text_bytes = 0
    token_count = 0
    while text_bytes < max_text_bytes:
        try:
            text = next(texts)
        except StopIteration:
            break
        encoded = tokenizer.encode(text)
        text_bytes += len(text.encode("utf-8"))
        token_count += len(encoded)
        document_count += 1
    elapsed = perf_counter() - started
    if trace_memory:
        _, peak_traced_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    else:
        peak_traced_bytes = None

    if document_count == 0:
        raise ValueError("sample contains no benchmark documents")
    return EncodingBenchmark(
        tokenizer_hash=tokenizer.model_hash,
        sample_sha256=sample.sample_sha256,
        python_version=platform.python_version(),
        platform=platform.platform(),
        document_count=document_count,
        text_bytes=text_bytes,
        token_count=token_count,
        elapsed_seconds=elapsed,
        megabytes_per_second=text_bytes / 1_000_000 / elapsed,
        tokens_per_second=token_count / elapsed,
        peak_traced_bytes=peak_traced_bytes,
    )


def append_benchmark(path: str | Path, benchmark: EncodingBenchmark) -> None:
    """Append one canonical benchmark record and force it to stable storage."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            benchmark.model_dump(mode="json"),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    descriptor = os.open(
        destination,
        os.O_APPEND | os.O_CREAT | os.O_WRONLY,
        0o644,
    )
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
