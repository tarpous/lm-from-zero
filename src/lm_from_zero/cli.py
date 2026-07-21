"""Local command-line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from lm_from_zero.data import (
    DataValidationError,
    SplitPolicy,
    load_manifest,
    validate_shard,
)
from lm_from_zero.sampling import (
    SamplingConfig,
    load_sample,
    stream_hugging_face_sample,
)
from lm_from_zero.sharding import build_token_shards, validate_shard_build
from lm_from_zero.tokenizer.benchmark import append_benchmark, benchmark_encoding
from lm_from_zero.tokenizer.oracle import (
    append_oracle_result,
    verify_hugging_face_oracle,
)
from lm_from_zero.tokenizer.pipeline import (
    TokenizerTrainingConfig,
    load_training_manifest,
    train_tokenizer_from_sample,
)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False)


@app.callback()
def main() -> None:
    """Inspect and build local lm-from-zero artifacts."""


@app.command("sample-tinystories")
def sample_tinystories_command(
    output_directory: Annotated[Path, typer.Option()] = Path("data/tinystories"),
    cache_directory: Annotated[Path, typer.Option()] = Path(".cache/huggingface"),
    target_text_bytes: Annotated[int, typer.Option(min=1)] = 100_000_000,
    max_storage_bytes: Annotated[int, typer.Option(min=1)] = 1_000_000_000,
) -> None:
    """Stream the pinned public TinyStories prefix without authentication."""

    config = SamplingConfig(
        target_text_bytes=target_text_bytes,
        max_storage_bytes=max_storage_bytes,
    )
    manifest_path = stream_hugging_face_sample(
        output_directory,
        cache_directory,
        config,
    )
    typer.echo(manifest_path)


@app.command("verify-sample")
def verify_sample_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Verify every document and provenance hash in a text sample."""

    metadata, texts = load_sample(manifest)
    document_count = sum(1 for _ in texts)
    typer.echo(
        json.dumps(
            {
                "aggregate_content_sha256": metadata.aggregate_content_sha256,
                "document_count": document_count,
                "revision": metadata.revision,
                "sample_sha256": metadata.sample_sha256,
                "status": "valid",
                "text_bytes": metadata.actual_text_bytes,
            },
            sort_keys=True,
        )
    )


@app.command("train-tokenizer")
def train_tokenizer_command(
    sample_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_directory: Annotated[Path, typer.Option()],
    target_vocab_size: Annotated[int, typer.Option(min=264, max=65536)] = 16_000,
    min_frequency: Annotated[int, typer.Option(min=1)] = 2,
    checkpoint_every_merges: Annotated[int, typer.Option(min=1)] = 250,
    max_documents: Annotated[int | None, typer.Option(min=1)] = None,
    max_corpus_bytes: Annotated[int | None, typer.Option(min=1)] = None,
    stop_after_new_merges: Annotated[int | None, typer.Option(min=1)] = None,
) -> None:
    """Train or resume the project-owned GPT-2-pretokenized byte BPE."""

    config = TokenizerTrainingConfig(
        target_vocab_size=target_vocab_size,
        min_frequency=min_frequency,
        checkpoint_every_merges=checkpoint_every_merges,
        max_documents=max_documents,
        max_corpus_bytes=max_corpus_bytes,
    )
    manifest = train_tokenizer_from_sample(
        sample_manifest,
        output_directory,
        config,
        stop_after_new_merges=stop_after_new_merges,
    )
    typer.echo(manifest.model_dump_json())


@app.command("benchmark-tokenizer")
def benchmark_tokenizer_command(
    tokenizer: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    sample_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    max_text_bytes: Annotated[int, typer.Option(min=1)] = 10_000_000,
    warmup_text_bytes: Annotated[int, typer.Option(min=0)] = 100_000,
    trace_memory: Annotated[bool, typer.Option()] = False,
    jsonl_output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Measure pure-Python encoding throughput and peak traced memory."""

    benchmark = benchmark_encoding(
        tokenizer,
        sample_manifest,
        max_text_bytes=max_text_bytes,
        warmup_text_bytes=warmup_text_bytes,
        trace_memory=trace_memory,
    )
    if jsonl_output is not None:
        append_benchmark(jsonl_output, benchmark)
    typer.echo(benchmark.model_dump_json())


@app.command("verify-tokenizer-oracle")
def verify_tokenizer_oracle_command(
    training_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    sample_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    jsonl_output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Retrain the Rust BPE oracle and require exact merge/encoding parity."""

    result = verify_hugging_face_oracle(training_manifest, sample_manifest)
    if jsonl_output is not None:
        append_oracle_result(jsonl_output, result)
    typer.echo(result.model_dump_json())


@app.command("build-token-shards")
def build_token_shards_command(
    sample_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    training_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_directory: Annotated[Path, typer.Option()],
    max_tokens_per_shard: Annotated[
        int, typer.Option(min=1, max=100_000_000)
    ] = 100_000_000,
    split_seed: Annotated[int, typer.Option()] = 1337,
    bucket_count: Annotated[int, typer.Option(min=1)] = 10_000,
    validation_buckets: Annotated[int, typer.Option(min=0)] = 100,
    test_buckets: Annotated[int, typer.Option(min=0)] = 100,
) -> None:
    """Split a checked sample and atomically publish validated uint16 shards."""

    policy = SplitPolicy(
        seed=split_seed,
        bucket_count=bucket_count,
        validation_buckets=validation_buckets,
        test_buckets=test_buckets,
    )
    result = build_token_shards(
        sample_manifest,
        training_manifest,
        output_directory,
        split_policy=policy,
        max_tokens_per_shard=max_tokens_per_shard,
    )
    typer.echo(result.model_dump_json())


@app.command("verify-shard-build")
def verify_shard_build_command(
    build_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate a complete shard set and its sample/tokenizer provenance."""

    result = validate_shard_build(build_manifest)
    typer.echo(
        json.dumps(
            {
                "document_count": result.source_document_count,
                "status": "valid",
                "token_count": result.total_token_count,
                "tokenizer_hash": result.tokenizer_hash,
            },
            sort_keys=True,
        )
    )


@app.command("dense-model-summary")
def dense_model_summary_command(
    training_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate and summarize the pinned 20M TinyStories dense model."""

    from lm_from_zero.models import Olmo2Config, Olmo2ForCausalLM

    training = load_training_manifest(training_manifest)
    if training.status != "complete":
        raise DataValidationError("tokenizer training must be complete")
    if training.realized_vocab_size != 16_000:
        raise DataValidationError(
            "the TinyStories dense model requires a 16K tokenizer"
        )
    config = Olmo2Config(tokenizer_hash=training.tokenizer_hash)
    expected = config.parameter_breakdown()
    model = Olmo2ForCausalLM(config)
    actual = model.trainable_parameter_count()
    if actual != expected.total:
        raise DataValidationError("realized model parameters do not match the analysis")
    typer.echo(
        json.dumps(
            {
                "config_hash": config.config_hash,
                "context_length": config.max_position_embeddings,
                "flops": config.forward_flops(
                    config.max_position_embeddings
                ).model_dump(mode="json"),
                "model_name": config.model_name,
                "parameters": expected.model_dump(mode="json"),
                "tokenizer_hash": config.tokenizer_hash,
            },
            sort_keys=True,
        )
    )


@app.command("verify-shard")
def verify_shard_command(
    manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    tokenizer_hash: Annotated[str | None, typer.Option()] = None,
    vocab_size: Annotated[int | None, typer.Option(min=1, max=65536)] = None,
) -> None:
    """Validate one token shard and print its checked metadata."""

    tokens = validate_shard(
        manifest,
        expected_tokenizer_hash=tokenizer_hash,
        expected_vocab_size=vocab_size,
    )
    metadata = load_manifest(manifest)
    typer.echo(
        json.dumps(
            {
                "manifest": manifest.name,
                "split": metadata.split,
                "token_count": len(tokens),
                "sha256": metadata.sha256,
                "status": "valid",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    app()
