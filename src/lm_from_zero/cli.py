"""Local command-line entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

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


@app.command("pretrain-dense")
def pretrain_dense_command(
    build_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    checkpoint_directory: Annotated[Path, typer.Option()],
    jsonl_log: Annotated[Path, typer.Option()],
    target_tokens: Annotated[int, typer.Option(min=1)] = 500_000_000,
    sequence_length: Annotated[int, typer.Option(min=2)] = 1_024,
    micro_batch_size: Annotated[int, typer.Option(min=1)] = 8,
    gradient_accumulation_steps: Annotated[int, typer.Option(min=1)] = 1,
    learning_rate: Annotated[float, typer.Option(min=1e-12)] = 1e-3,
    device: Annotated[str, typer.Option()] = "cuda",
    precision: Annotated[str, typer.Option()] = "bf16",
    compile_model: Annotated[bool, typer.Option()] = True,
    estimated_tokens_per_second: Annotated[
        float | None, typer.Option(min=1e-12)
    ] = None,
    resume_from: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    stop_after_optimizer_step: Annotated[int | None, typer.Option(min=1)] = None,
    execute: Annotated[bool, typer.Option()] = False,
) -> None:
    """Plan dense pretraining; execute only after the separate approval gate."""

    from lm_from_zero.models import Olmo2Config, Olmo2ForCausalLM
    from lm_from_zero.training import (
        CausalBatchConfig,
        DenseTrainer,
        DenseTrainingConfig,
        OptimizationConfig,
        ShardBatchSource,
        create_dense_run_plan,
        distributed_session,
        optimizer_steps_for_token_budget,
        seed_training,
    )

    with distributed_session(device) as distributed:
        build = validate_shard_build(build_manifest)
        if build.tokenizer_vocab_size != 16_000:
            raise DataValidationError(
                "dense TinyStories pretraining requires 16K shards"
            )
        model_config = Olmo2Config(tokenizer_hash=build.tokenizer_hash)
        batch_config = CausalBatchConfig(
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            seed=1_337,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        optimizer_steps = optimizer_steps_for_token_budget(
            target_tokens,
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            world_size=distributed.world_size,
        )
        training_config = DenseTrainingConfig.model_validate(
            {
                "model": model_config,
                "batch": batch_config,
                "optimization": OptimizationConfig(
                    learning_rate=learning_rate,
                    total_steps=optimizer_steps,
                ),
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "device": device,
                "precision": precision,
                "compile_model": compile_model,
            }
        )
        source = ShardBatchSource(build_manifest, batch_config)
        plan = create_dense_run_plan(
            training_config,
            source,
            checkpoint_directory,
            estimated_tokens_per_second=estimated_tokens_per_second,
        )
        if distributed.is_primary:
            typer.echo(plan.model_dump_json())
        if not execute:
            return

        seed_training(training_config.seed, cuda=training_config.device == "cuda")
        model = Olmo2ForCausalLM(model_config)
        trainer = DenseTrainer(
            model=model,
            source=source,
            config=training_config,
            checkpoint_directory=checkpoint_directory,
            repository=Path.cwd(),
            jsonl_log=jsonl_log,
            distributed=distributed,
        )
        result = trainer.run(
            resume_from=resume_from,
            stop_after_optimizer_step=stop_after_optimizer_step,
        )
        if distributed.is_primary:
            typer.echo(result.model_dump_json())


@app.command("evaluate-dense")
def evaluate_dense_command(
    checkpoint: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    build_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    max_batches: Annotated[int, typer.Option(min=1)] = 32,
    sequence_length: Annotated[int, typer.Option(min=2)] = 1_024,
    batch_size: Annotated[int, typer.Option(min=1)] = 8,
    split: Annotated[str, typer.Option()] = "validation",
    device: Annotated[str, typer.Option()] = "cpu",
    precision: Annotated[str, typer.Option()] = "fp32",
    jsonl_output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Evaluate a validated dense checkpoint on fixed non-repeating shards."""

    from lm_from_zero.evaluation import (
        CausalEvaluationConfig,
        append_evaluation_result,
        evaluate_causal_loss,
    )
    from lm_from_zero.models import Olmo2Config, Olmo2ForCausalLM
    from lm_from_zero.training import (
        CausalBatchConfig,
        ShardBatchSource,
        load_checkpoint_model,
        validate_checkpoint,
    )

    manifest = validate_checkpoint(checkpoint)
    model_config = Olmo2Config.model_validate(manifest.binding.resolved_model_config)
    batch_config = CausalBatchConfig.model_validate(
        {
            "split": split,
            "sequence_length": sequence_length,
            "micro_batch_size": batch_size,
            "shuffle": False,
        }
    )
    evaluation_config = CausalEvaluationConfig.model_validate(
        {
            "max_batches": max_batches,
            "device": device,
            "precision": precision,
        }
    )
    source = ShardBatchSource(build_manifest, batch_config)
    model = Olmo2ForCausalLM(model_config)
    load_checkpoint_model(
        checkpoint,
        model=model,
        expected_binding=manifest.binding,
    )
    result = evaluate_causal_loss(model, source, evaluation_config)
    if jsonl_output is not None:
        append_evaluation_result(jsonl_output, result)
    typer.echo(result.canonical_json())


@app.command("export-dense-hf")
def export_dense_hf_command(
    checkpoint: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    tokenizer_training_manifest: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    output_directory: Annotated[Path, typer.Option()],
) -> None:
    """Export a validated dense checkpoint as a standard local OLMo2 package."""

    from lm_from_zero.export_hf import export_dense_to_hugging_face

    result = export_dense_to_hugging_face(
        checkpoint,
        tokenizer_training_manifest,
        output_directory,
    )
    typer.echo(result.model_dump_json())


@app.command("generate-dense")
def generate_dense_command(
    checkpoint: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    tokenizer_training_manifest: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    prompt: Annotated[str, typer.Argument()],
    max_new_tokens: Annotated[int, typer.Option(min=1)] = 64,
    strategy: Annotated[Literal["greedy", "sample"], typer.Option()] = "greedy",
    temperature: Annotated[float, typer.Option(min=1e-12)] = 1.0,
    top_k: Annotated[int | None, typer.Option(min=1)] = None,
    top_p: Annotated[float | None, typer.Option(min=1e-12, max=1)] = None,
    seed: Annotated[int, typer.Option()] = 1337,
    device: Annotated[str, typer.Option()] = "cpu",
    allow_raw_special_tokens: Annotated[bool, typer.Option()] = False,
    stream: Annotated[bool, typer.Option()] = False,
    jsonl_output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Generate locally from a validated dense checkpoint with native KV caching."""

    import torch

    from lm_from_zero.generation import (
        CausalGenerationConfig,
        CausalGenerationEvent,
        append_generation_record,
        create_generation_record,
        generate_causal,
    )
    from lm_from_zero.models import Olmo2Config, Olmo2ForCausalLM
    from lm_from_zero.tokenizer.bpe import ByteBPE
    from lm_from_zero.training import load_checkpoint_model, validate_checkpoint

    checkpoint_manifest = validate_checkpoint(checkpoint)
    if checkpoint_manifest.binding.architecture != "olmo2":
        raise DataValidationError(
            "native dense generation requires an OLMo2 checkpoint"
        )
    model_config = Olmo2Config.model_validate(
        checkpoint_manifest.binding.resolved_model_config
    )
    training = load_training_manifest(tokenizer_training_manifest)
    if (
        training.status != "complete"
        or training.tokenizer_hash != checkpoint_manifest.binding.tokenizer_sha256
    ):
        raise DataValidationError("tokenizer manifest does not match the checkpoint")
    tokenizer = ByteBPE.load(
        tokenizer_training_manifest.parent / training.tokenizer_file
    )
    if tokenizer.model_hash != training.tokenizer_hash:
        raise DataValidationError("tokenizer file does not match its manifest")
    model = Olmo2ForCausalLM(model_config)
    load_checkpoint_model(
        checkpoint,
        model=model,
        expected_binding=checkpoint_manifest.binding,
    )
    model.to(torch.device(device))
    generation_config = CausalGenerationConfig(
        strategy=strategy,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        allow_raw_special_tokens=allow_raw_special_tokens,
    )

    def emit(event: CausalGenerationEvent) -> None:
        if stream:
            typer.echo(
                json.dumps(
                    {"event": "token", **event.model_dump(mode="json")},
                    sort_keys=True,
                )
            )

    prompt_ids = tokenizer.encode(prompt)
    result = generate_causal(
        model,
        [prompt_ids],
        generation_config,
        on_token=emit,
    )
    if jsonl_output is not None:
        append_generation_record(
            jsonl_output,
            create_generation_record(
                result,
                [prompt_ids],
                model_config_sha256=model_config.config_hash,
                tokenizer_sha256=tokenizer.model_hash,
            ),
        )
    generated = result.generated_token_ids[0]
    typer.echo(
        json.dumps(
            {
                "event": "complete",
                "generated_text": tokenizer.decode(
                    generated,
                    render_special=True,
                    errors="replace",
                ),
                **result.model_dump(mode="json"),
            },
            sort_keys=True,
        )
    )


@app.command("build-dense-smoke-report")
def build_dense_smoke_report_command(
    training_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    checkpoint: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    evaluation_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    export_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    generation_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Generate portable smoke evidence from validated local artifacts."""

    from lm_from_zero.smoke_report import (
        build_dense_smoke_report,
        write_dense_smoke_report,
    )

    report = build_dense_smoke_report(
        training_jsonl=training_jsonl,
        checkpoint_directory=checkpoint,
        evaluation_jsonl=evaluation_jsonl,
        export_directory=export_directory,
        generation_jsonl=generation_jsonl,
    )
    write_dense_smoke_report(output, report)
    typer.echo(report.model_dump_json())


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
