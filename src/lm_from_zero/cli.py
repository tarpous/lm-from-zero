"""Local command-line entry points."""

from __future__ import annotations

import json
from hashlib import sha256
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


@app.command("verify-mamba2-oracle")
def verify_mamba2_oracle_command(
    output: Annotated[Path, typer.Option()],
    seed: Annotated[int, typer.Option()] = 1337,
) -> None:
    """Compare project-owned chunked SSD with the optional CUDA oracle."""

    from lm_from_zero.mamba2_oracle import (
        Mamba2OracleConfig,
        verify_mamba2_oracle,
        write_mamba2_oracle_report,
    )

    result = verify_mamba2_oracle(Mamba2OracleConfig(seed=seed))
    write_mamba2_oracle_report(output, result)
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


@app.command("mamba2-model-summary")
def mamba2_model_summary_command(
    training_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate and summarize the pinned 20M TinyStories Mamba-2 model."""

    from lm_from_zero.models import Mamba2Config, Mamba2ForCausalLM

    training = load_training_manifest(training_manifest)
    if training.status != "complete":
        raise DataValidationError("tokenizer training must be complete")
    if training.realized_vocab_size != 16_000:
        raise DataValidationError("the TinyStories Mamba-2 model requires 16K")
    config = Mamba2Config(tokenizer_hash=training.tokenizer_hash)
    expected = config.parameter_breakdown()
    model = Mamba2ForCausalLM(config)
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


@app.command("diffusion-model-summary")
def diffusion_model_summary_command(
    training_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
) -> None:
    """Validate and summarize the pinned 20M TinyStories diffusion model."""

    from lm_from_zero.models import (
        MaskedDiffusionConfig,
        MaskedDiffusionForMaskedLM,
    )

    training = load_training_manifest(training_manifest)
    if training.status != "complete":
        raise DataValidationError("tokenizer training must be complete")
    if training.realized_vocab_size != 16_000:
        raise DataValidationError("the TinyStories diffusion model requires 16K")
    config = MaskedDiffusionConfig(tokenizer_hash=training.tokenizer_hash)
    expected = config.parameter_breakdown()
    model = MaskedDiffusionForMaskedLM(config)
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
                "time_conditioning": config.time_conditioning,
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
    tensorboard_directory: Annotated[Path | None, typer.Option()] = None,
    parquet_log: Annotated[Path | None, typer.Option()] = None,
    target_tokens: Annotated[int, typer.Option(min=1)] = 500_000_000,
    sequence_length: Annotated[int, typer.Option(min=2)] = 1_024,
    micro_batch_size: Annotated[int, typer.Option(min=1)] = 8,
    gradient_accumulation_steps: Annotated[int, typer.Option(min=1)] = 1,
    learning_rate: Annotated[float, typer.Option(min=1e-12)] = 1e-3,
    seed: Annotated[int, typer.Option()] = 1_337,
    device: Annotated[str, typer.Option()] = "cuda",
    precision: Annotated[str, typer.Option()] = "bf16",
    compile_model: Annotated[bool, typer.Option()] = True,
    compile_mode: Annotated[str, typer.Option()] = "default",
    adamw_backend: Annotated[str, typer.Option()] = "auto",
    model_variant: Annotated[str, typer.Option()] = "baseline",
    optimizer_variant: Annotated[str, typer.Option()] = "adamw",
    loss_backend: Annotated[str, typer.Option()] = "full",
    sdpa_backend: Annotated[str, typer.Option()] = "auto",
    float32_matmul_precision: Annotated[str, typer.Option()] = "highest",
    telemetry_every_steps: Annotated[int, typer.Option(min=1)] = 50,
    checkpoint_every_steps: Annotated[int | None, typer.Option(min=1)] = None,
    checkpoint_every_seconds: Annotated[float, typer.Option(min=1e-6)] = 900.0,
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
        resolved_tensorboard_directory = (
            jsonl_log.parent / "tensorboard"
            if tensorboard_directory is None
            else tensorboard_directory
        )
        resolved_parquet_log = (
            jsonl_log.with_suffix(".parquet") if parquet_log is None else parquet_log
        )
        build = validate_shard_build(build_manifest)
        if build.tokenizer_vocab_size != 16_000:
            raise DataValidationError(
                "dense TinyStories pretraining requires 16K shards"
            )
        model_config = Olmo2Config(tokenizer_hash=build.tokenizer_hash)
        batch_config = CausalBatchConfig(
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            seed=seed,
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
                "compile_mode": compile_mode,
                "adamw_backend": adamw_backend,
                "model_variant": model_variant,
                "optimizer_variant": optimizer_variant,
                "loss_backend": loss_backend,
                "sdpa_backend": sdpa_backend,
                "float32_matmul_precision": float32_matmul_precision,
                "telemetry_every_steps": telemetry_every_steps,
                "checkpoint_every_steps": checkpoint_every_steps,
                "checkpoint_every_seconds": checkpoint_every_seconds,
                "seed": seed,
            }
        )
        source = ShardBatchSource(build_manifest, batch_config)
        plan = create_dense_run_plan(
            training_config,
            source,
            checkpoint_directory,
            estimated_tokens_per_second=estimated_tokens_per_second,
            jsonl_log=jsonl_log,
            tensorboard_directory=resolved_tensorboard_directory,
            parquet_log=resolved_parquet_log,
        )
        if distributed.is_primary:
            typer.echo(plan.model_dump_json())
        if not execute:
            return

        seed_training(training_config.seed, cuda=training_config.device == "cuda")
        model = Olmo2ForCausalLM(model_config, variant=training_config.model_variant)
        trainer = DenseTrainer(
            model=model,
            source=source,
            config=training_config,
            checkpoint_directory=checkpoint_directory,
            repository=Path.cwd(),
            jsonl_log=jsonl_log,
            tensorboard_directory=resolved_tensorboard_directory,
            parquet_log=resolved_parquet_log,
            distributed=distributed,
        )
        result = trainer.run(
            resume_from=resume_from,
            stop_after_optimizer_step=stop_after_optimizer_step,
        )
        if distributed.is_primary:
            typer.echo(result.model_dump_json())


@app.command("plan-architecture-study")
def plan_architecture_study_command(
    build_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option()] = None,
    screening_dense_reference_tokens: Annotated[int, typer.Option(min=1)] = 100_000_000,
    full_dense_reference_tokens: Annotated[int, typer.Option(min=1)] = 500_000_000,
    micro_batch_size: Annotated[int, typer.Option(min=1)] = 8,
    gradient_accumulation_steps: Annotated[int, typer.Option(min=1)] = 1,
    dense_tokens_per_second: Annotated[float | None, typer.Option(min=1e-12)] = None,
    mamba2_tokens_per_second: Annotated[float | None, typer.Option(min=1e-12)] = None,
    diffusion_tokens_per_second: Annotated[
        float | None, typer.Option(min=1e-12)
    ] = None,
    diffusion_adamw_backend: Literal["auto", "foreach", "fused"] = "auto",
) -> None:
    """Freeze all nine full-scheduler lineages before architecture-study runs."""

    from lm_from_zero.architecture_study import (
        create_architecture_study_plan,
        write_architecture_study_plan,
    )

    plan = create_architecture_study_plan(
        build_manifest,
        screening_dense_reference_tokens=screening_dense_reference_tokens,
        full_dense_reference_tokens=full_dense_reference_tokens,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        dense_tokens_per_second=dense_tokens_per_second,
        mamba2_tokens_per_second=mamba2_tokens_per_second,
        diffusion_tokens_per_second=diffusion_tokens_per_second,
        diffusion_adamw_backend=diffusion_adamw_backend,
    )
    if output is not None:
        write_architecture_study_plan(output, plan)
    typer.echo(plan.canonical_json())


@app.command("plan-dense-ablations")
def plan_dense_ablations_command(
    architecture_study_plan: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    output: Annotated[Path | None, typer.Option()] = None,
    artifact_root: Annotated[Path, typer.Option()] = Path("artifacts/dense-ablations"),
) -> None:
    """Freeze the CPU-only M8 dense baseline and seven variant jobs."""

    from lm_from_zero.dense_ablations import (
        create_dense_ablation_plan,
        write_dense_ablation_plan,
    )

    plan = create_dense_ablation_plan(
        architecture_study_plan,
        artifact_root=artifact_root,
    )
    if output is not None:
        write_dense_ablation_plan(output, plan)
    typer.echo(plan.canonical_json())


@app.command("pretrain-mamba2")
def pretrain_mamba2_command(
    build_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    checkpoint_directory: Annotated[Path, typer.Option()],
    jsonl_log: Annotated[Path, typer.Option()],
    tensorboard_directory: Annotated[Path | None, typer.Option()] = None,
    parquet_log: Annotated[Path | None, typer.Option()] = None,
    target_tokens: Annotated[int | None, typer.Option(min=1)] = None,
    dense_reference_tokens: Annotated[int, typer.Option(min=1)] = 500_000_000,
    sequence_length: Annotated[int, typer.Option(min=2)] = 1_024,
    micro_batch_size: Annotated[int, typer.Option(min=1)] = 8,
    gradient_accumulation_steps: Annotated[int, typer.Option(min=1)] = 1,
    learning_rate: Annotated[float, typer.Option(min=1e-12)] = 1e-3,
    seed: Annotated[int, typer.Option()] = 1_337,
    device: Annotated[str, typer.Option()] = "cuda",
    precision: Annotated[str, typer.Option()] = "bf16",
    compile_model: Annotated[bool, typer.Option()] = True,
    compile_mode: Annotated[str, typer.Option()] = "default",
    adamw_backend: Annotated[str, typer.Option()] = "auto",
    loss_backend: Annotated[str, typer.Option()] = "full",
    float32_matmul_precision: Annotated[str, typer.Option()] = "highest",
    telemetry_every_steps: Annotated[int, typer.Option(min=1)] = 50,
    checkpoint_every_steps: Annotated[int | None, typer.Option(min=1)] = None,
    checkpoint_every_seconds: Annotated[float, typer.Option(min=1e-6)] = 900.0,
    estimated_tokens_per_second: Annotated[
        float | None, typer.Option(min=1e-12)
    ] = None,
    resume_from: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    stop_after_optimizer_step: Annotated[int | None, typer.Option(min=1)] = None,
    execute: Annotated[bool, typer.Option()] = False,
) -> None:
    """Plan Mamba-2 pretraining; execute only after the approval gate."""

    from lm_from_zero.models import (
        Mamba2Config,
        Mamba2ForCausalLM,
        Olmo2Config,
    )
    from lm_from_zero.training import (
        CausalBatchConfig,
        Mamba2Trainer,
        Mamba2TrainingConfig,
        OptimizationConfig,
        ShardBatchSource,
        create_mamba2_run_plan,
        distributed_session,
        optimizer_steps_for_token_budget,
        seed_training,
    )

    with distributed_session(device) as distributed:
        resolved_tensorboard_directory = (
            jsonl_log.parent / "tensorboard"
            if tensorboard_directory is None
            else tensorboard_directory
        )
        resolved_parquet_log = (
            jsonl_log.with_suffix(".parquet") if parquet_log is None else parquet_log
        )
        build = validate_shard_build(build_manifest)
        if build.tokenizer_vocab_size != 16_000:
            raise DataValidationError(
                "Mamba-2 TinyStories pretraining requires 16K shards"
            )
        model_config = Mamba2Config(tokenizer_hash=build.tokenizer_hash)
        dense_reference = Olmo2Config(tokenizer_hash=build.tokenizer_hash)
        dense_forward_flops = dense_reference.forward_flops(
            sequence_length
        ).total_flops_per_token
        mamba_forward_flops = model_config.forward_flops(
            sequence_length
        ).total_flops_per_token
        reference_training_flops = 3 * dense_forward_flops * dense_reference_tokens
        matched_tokens = (reference_training_flops + 3 * mamba_forward_flops - 1) // (
            3 * mamba_forward_flops
        )
        resolved_target_tokens = (
            matched_tokens if target_tokens is None else target_tokens
        )
        batch_config = CausalBatchConfig(
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            seed=seed,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        optimizer_steps = optimizer_steps_for_token_budget(
            resolved_target_tokens,
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            world_size=distributed.world_size,
        )
        training_config = Mamba2TrainingConfig.model_validate(
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
                "compile_mode": compile_mode,
                "adamw_backend": adamw_backend,
                "loss_backend": loss_backend,
                "float32_matmul_precision": float32_matmul_precision,
                "telemetry_every_steps": telemetry_every_steps,
                "checkpoint_every_steps": checkpoint_every_steps,
                "checkpoint_every_seconds": checkpoint_every_seconds,
                "seed": seed,
            }
        )
        source = ShardBatchSource(build_manifest, batch_config)
        plan = create_mamba2_run_plan(
            training_config,
            source,
            checkpoint_directory,
            estimated_tokens_per_second=estimated_tokens_per_second,
            jsonl_log=jsonl_log,
            tensorboard_directory=resolved_tensorboard_directory,
            parquet_log=resolved_parquet_log,
            reference_training_flops=reference_training_flops,
        )
        if distributed.is_primary:
            typer.echo(plan.model_dump_json())
        if not execute:
            return

        seed_training(training_config.seed, cuda=training_config.device == "cuda")
        model = Mamba2ForCausalLM(model_config)
        trainer = Mamba2Trainer(
            model=model,
            source=source,
            config=training_config,
            checkpoint_directory=checkpoint_directory,
            repository=Path.cwd(),
            jsonl_log=jsonl_log,
            tensorboard_directory=resolved_tensorboard_directory,
            parquet_log=resolved_parquet_log,
            distributed=distributed,
        )
        result = trainer.run(
            resume_from=resume_from,
            stop_after_optimizer_step=stop_after_optimizer_step,
        )
        if distributed.is_primary:
            typer.echo(result.model_dump_json())


@app.command("plan-acceleration-calibration")
def plan_acceleration_calibration_command(
    build_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path | None, typer.Option()] = None,
    artifact_root: Annotated[Path, typer.Option()] = Path(
        "artifacts/acceleration-calibration/fixed-500-results"
    ),
    micro_batch_size: Annotated[int, typer.Option(min=1)] = 8,
    gradient_accumulation_steps: Annotated[int, typer.Option(min=1)] = 1,
    telemetry_interval_steps: Annotated[int, typer.Option(min=2)] = 50,
    expected_cuda_device_name: Annotated[str, typer.Option()] = (
        "NVIDIA GeForce RTX 4080 SUPER"
    ),
) -> None:
    """Freeze the synchronized Milestone 6A matrix without allocating a model."""

    from lm_from_zero.acceleration_calibration import (
        AccelerationCalibrationError,
        create_plan,
        write_plan,
    )
    from lm_from_zero.acceleration_execution import inspect_repository_state

    repository_state = inspect_repository_state(Path.cwd())
    if repository_state.dirty:
        raise AccelerationCalibrationError(
            "calibration plans require a clean Git worktree"
        )

    plan = create_plan(
        build_manifest,
        repository_revision=repository_state.revision,
        artifact_root=artifact_root,
        micro_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        telemetry_interval_steps=telemetry_interval_steps,
        expected_cuda_device_name=expected_cuda_device_name,
    )
    if output is not None:
        write_plan(output, plan)
    typer.echo(plan.canonical_json())


@app.command("inspect-acceleration-calibration-cell")
def inspect_acceleration_calibration_cell_command(
    plan_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    architecture: Annotated[str, typer.Argument()],
    cell_id: Annotated[str, typer.Argument()],
) -> None:
    """Resolve one calibration cell; this command never executes GPU work."""

    from lm_from_zero.acceleration_calibration import (
        ARCHITECTURES,
        AccelerationCalibrationError,
        load_plan,
        resolve_cell,
    )

    if architecture not in ARCHITECTURES:
        raise AccelerationCalibrationError("unknown calibration architecture")
    plan = load_plan(plan_path)
    cell = resolve_cell(plan, architecture, cell_id)
    typer.echo(cell.canonical_json())


@app.command("build-acceleration-calibration-report")
def build_acceleration_calibration_report_command(
    plan_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    results_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Validate complete calibration evidence and publish its derived report."""

    from lm_from_zero.acceleration_calibration import (
        build_report,
        load_plan,
        load_results,
        write_report,
    )

    report = build_report(load_plan(plan_path), load_results(results_directory))
    write_report(output, report)
    typer.echo(report.canonical_json())


@app.command("run-acceleration-calibration-cell")
def run_acceleration_calibration_cell_command(
    plan_path: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    build_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    architecture: Annotated[str, typer.Argument()],
    cell_id: Annotated[str, typer.Argument()],
    repetition: Annotated[int, typer.Option(min=1)] = 1,
    tokenizer_model: Annotated[Path | None, typer.Option()] = None,
    repository: Annotated[Path, typer.Option()] = Path("."),
    execute: Annotated[bool, typer.Option()] = False,
) -> None:
    """Resolve one fresh-process cell and optionally generate CUDA evidence."""

    from lm_from_zero.acceleration_calibration import (
        ARCHITECTURES,
        AccelerationCalibrationError,
    )
    from lm_from_zero.acceleration_execution import (
        AccelerationExecutionError,
        execute_calibration_cell,
        require_executable_calibration_cell,
        resolve_from_plan,
    )

    if architecture not in ARCHITECTURES:
        raise AccelerationCalibrationError("unknown calibration architecture")
    dry_run = resolve_from_plan(
        plan_path,
        build_manifest,
        architecture,
        cell_id,
        repetition,
    )
    typer.echo(dry_run.canonical_json())
    if execute:
        require_executable_calibration_cell(dry_run)
        if tokenizer_model is None:
            raise AccelerationExecutionError(
                "--tokenizer-model is required with --execute"
            )
        result = execute_calibration_cell(
            plan_path,
            build_manifest,
            tokenizer_model,
            repository,
            architecture,
            cell_id,
            repetition,
        )
        typer.echo(result.canonical_json())


@app.command("pretrain-diffusion")
def pretrain_diffusion_command(
    build_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    checkpoint_directory: Annotated[Path, typer.Option()],
    jsonl_log: Annotated[Path, typer.Option()],
    tensorboard_directory: Annotated[Path | None, typer.Option()] = None,
    parquet_log: Annotated[Path | None, typer.Option()] = None,
    target_tokens: Annotated[int | None, typer.Option(min=1)] = None,
    dense_reference_tokens: Annotated[int, typer.Option(min=1)] = 500_000_000,
    sequence_length: Annotated[int, typer.Option(min=2)] = 1_024,
    micro_batch_size: Annotated[int, typer.Option(min=1)] = 8,
    gradient_accumulation_steps: Annotated[int, typer.Option(min=1)] = 1,
    learning_rate: Annotated[float, typer.Option(min=1e-12)] = 1e-3,
    seed: Annotated[int, typer.Option()] = 1_337,
    device: Annotated[str, typer.Option()] = "cuda",
    precision: Annotated[str, typer.Option()] = "bf16",
    compile_model: Annotated[bool, typer.Option()] = True,
    compile_mode: Annotated[str, typer.Option()] = "default",
    adamw_backend: Annotated[str, typer.Option()] = "auto",
    loss_backend: Annotated[str, typer.Option()] = "full",
    sdpa_backend: Annotated[str, typer.Option()] = "auto",
    diffusion_padding_free_attention: Annotated[bool, typer.Option()] = False,
    float32_matmul_precision: Annotated[str, typer.Option()] = "highest",
    telemetry_every_steps: Annotated[int, typer.Option(min=1)] = 50,
    checkpoint_every_steps: Annotated[int | None, typer.Option(min=1)] = None,
    checkpoint_every_seconds: Annotated[float, typer.Option(min=1e-6)] = 900.0,
    estimated_tokens_per_second: Annotated[
        float | None, typer.Option(min=1e-12)
    ] = None,
    resume_from: Annotated[
        Path | None, typer.Option(exists=True, file_okay=False)
    ] = None,
    stop_after_optimizer_step: Annotated[int | None, typer.Option(min=1)] = None,
    execute: Annotated[bool, typer.Option()] = False,
) -> None:
    """Plan diffusion pretraining; execute only after the approval gate."""

    from lm_from_zero.models import (
        MaskedDiffusionConfig,
        MaskedDiffusionForMaskedLM,
        Olmo2Config,
    )
    from lm_from_zero.training import (
        CausalBatchConfig,
        DiffusionTrainer,
        DiffusionTrainingConfig,
        OptimizationConfig,
        ShardBatchSource,
        create_diffusion_run_plan,
        distributed_session,
        optimizer_steps_for_token_budget,
        seed_training,
    )

    with distributed_session(device) as distributed:
        resolved_tensorboard_directory = (
            jsonl_log.parent / "tensorboard"
            if tensorboard_directory is None
            else tensorboard_directory
        )
        resolved_parquet_log = (
            jsonl_log.with_suffix(".parquet") if parquet_log is None else parquet_log
        )
        build = validate_shard_build(build_manifest)
        if build.tokenizer_vocab_size != 16_000:
            raise DataValidationError(
                "diffusion TinyStories pretraining requires 16K shards"
            )
        model_config = MaskedDiffusionConfig(tokenizer_hash=build.tokenizer_hash)
        dense_reference = Olmo2Config(tokenizer_hash=build.tokenizer_hash)
        dense_forward_flops = dense_reference.forward_flops(
            sequence_length
        ).total_flops_per_token
        diffusion_forward_flops = model_config.forward_flops(
            sequence_length
        ).total_flops_per_token
        reference_training_flops = 3 * dense_forward_flops * dense_reference_tokens
        matched_tokens = (
            reference_training_flops + 3 * diffusion_forward_flops - 1
        ) // (3 * diffusion_forward_flops)
        resolved_target_tokens = (
            matched_tokens if target_tokens is None else target_tokens
        )
        batch_config = CausalBatchConfig(
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            seed=seed,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        optimizer_steps = optimizer_steps_for_token_budget(
            resolved_target_tokens,
            sequence_length=sequence_length,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            world_size=distributed.world_size,
        )
        training_config = DiffusionTrainingConfig.model_validate(
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
                "compile_mode": compile_mode,
                "adamw_backend": adamw_backend,
                "loss_backend": loss_backend,
                "sdpa_backend": sdpa_backend,
                "diffusion_padding_free_attention": (diffusion_padding_free_attention),
                "float32_matmul_precision": float32_matmul_precision,
                "telemetry_every_steps": telemetry_every_steps,
                "checkpoint_every_steps": checkpoint_every_steps,
                "checkpoint_every_seconds": checkpoint_every_seconds,
                "seed": seed,
            }
        )
        source = ShardBatchSource(build_manifest, batch_config)
        plan = create_diffusion_run_plan(
            training_config,
            source,
            checkpoint_directory,
            estimated_tokens_per_second=estimated_tokens_per_second,
            jsonl_log=jsonl_log,
            tensorboard_directory=resolved_tensorboard_directory,
            parquet_log=resolved_parquet_log,
            reference_training_flops=reference_training_flops,
        )
        if distributed.is_primary:
            typer.echo(plan.model_dump_json())
        if not execute:
            return

        seed_training(training_config.seed, cuda=training_config.device == "cuda")
        model = MaskedDiffusionForMaskedLM(model_config)
        trainer = DiffusionTrainer(
            model=model,
            source=source,
            config=training_config,
            checkpoint_directory=checkpoint_directory,
            repository=Path.cwd(),
            jsonl_log=jsonl_log,
            tensorboard_directory=resolved_tensorboard_directory,
            parquet_log=resolved_parquet_log,
            distributed=distributed,
        )
        result = trainer.run(
            resume_from=resume_from,
            stop_after_optimizer_step=stop_after_optimizer_step,
        )
        if distributed.is_primary:
            typer.echo(result.model_dump_json())


@app.command("materialize-training-metrics")
def materialize_training_metrics_command(
    jsonl_log: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Rebuild a typed Parquet metric table from canonical training JSONL."""

    from lm_from_zero.training.metrics import (
        load_optimizer_metrics,
        materialize_metrics_parquet,
    )

    destination = materialize_metrics_parquet(jsonl_log, output)
    records = load_optimizer_metrics(jsonl_log)
    typer.echo(
        json.dumps(
            {
                "optimizer_steps": len(records),
                "output": str(destination),
            },
            sort_keys=True,
        )
    )


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


@app.command("evaluate-mamba2")
def evaluate_mamba2_command(
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
    """Evaluate a validated Mamba-2 checkpoint on fixed shards."""

    from lm_from_zero.evaluation import (
        CausalEvaluationConfig,
        append_evaluation_result,
        evaluate_causal_loss,
    )
    from lm_from_zero.models import Mamba2Config, Mamba2ForCausalLM
    from lm_from_zero.training import (
        CausalBatchConfig,
        ShardBatchSource,
        load_checkpoint_model,
        validate_checkpoint,
    )

    manifest = validate_checkpoint(checkpoint)
    if manifest.binding.architecture != "mamba2":
        raise DataValidationError("Mamba-2 evaluation requires a Mamba-2 checkpoint")
    model_config = Mamba2Config.model_validate(manifest.binding.resolved_model_config)
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
    model = Mamba2ForCausalLM(model_config)
    load_checkpoint_model(
        checkpoint,
        model=model,
        expected_binding=manifest.binding,
    )
    result = evaluate_causal_loss(model, source, evaluation_config)
    if jsonl_output is not None:
        append_evaluation_result(jsonl_output, result)
    typer.echo(result.canonical_json())


@app.command("evaluate-diffusion")
def evaluate_diffusion_command(
    checkpoint: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    build_manifest: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    max_batches: Annotated[int, typer.Option(min=1)] = 32,
    corruption_samples_per_batch: Annotated[int, typer.Option(min=1)] = 1,
    sequence_length: Annotated[int, typer.Option(min=2)] = 1_024,
    batch_size: Annotated[int, typer.Option(min=1)] = 8,
    split: Annotated[str, typer.Option()] = "validation",
    seed: Annotated[int, typer.Option()] = 1_337,
    device: Annotated[str, typer.Option()] = "cpu",
    precision: Annotated[str, typer.Option()] = "fp32",
    jsonl_output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Evaluate diffusion with fixed seeded corruption, never perplexity."""

    from lm_from_zero.diffusion_evaluation import (
        DiffusionEvaluationConfig,
        append_diffusion_evaluation_result,
        evaluate_diffusion,
    )
    from lm_from_zero.models import (
        MaskedDiffusionConfig,
        MaskedDiffusionForMaskedLM,
    )
    from lm_from_zero.training import (
        CausalBatchConfig,
        ShardBatchSource,
        load_checkpoint_model,
        validate_checkpoint,
    )

    manifest = validate_checkpoint(checkpoint)
    if manifest.binding.architecture != "masked_diffusion":
        raise DataValidationError(
            "diffusion evaluation requires a masked-diffusion checkpoint"
        )
    model_config = MaskedDiffusionConfig.model_validate(
        manifest.binding.resolved_model_config
    )
    batch_config = CausalBatchConfig.model_validate(
        {
            "split": split,
            "sequence_length": sequence_length,
            "micro_batch_size": batch_size,
            "shuffle": False,
        }
    )
    evaluation_config = DiffusionEvaluationConfig.model_validate(
        {
            "max_batches": max_batches,
            "corruption_samples_per_batch": corruption_samples_per_batch,
            "seed": seed,
            "device": device,
            "precision": precision,
        }
    )
    source = ShardBatchSource(build_manifest, batch_config)
    model = MaskedDiffusionForMaskedLM(model_config)
    load_checkpoint_model(
        checkpoint,
        model=model,
        expected_binding=manifest.binding,
    )
    result = evaluate_diffusion(
        model,
        source,
        evaluation_config,
        source_checkpoint_id=manifest.lineage.checkpoint_id,
        source_checkpoint_manifest_sha256=sha256(
            manifest.canonical_bytes()
        ).hexdigest(),
    )
    if jsonl_output is not None:
        append_diffusion_evaluation_result(jsonl_output, result)
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


@app.command("export-mamba2-hf")
def export_mamba2_hf_command(
    checkpoint: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    tokenizer_training_manifest: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    output_directory: Annotated[Path, typer.Option()],
) -> None:
    """Export Mamba-2 with grouped-normalization Hugging Face compatibility."""

    from lm_from_zero.export_mamba2_hf import export_mamba2_to_hugging_face

    result = export_mamba2_to_hugging_face(
        checkpoint,
        tokenizer_training_manifest,
        output_directory,
    )
    typer.echo(result.model_dump_json())


@app.command("export-diffusion-hf")
def export_diffusion_hf_command(
    checkpoint: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    tokenizer_training_manifest: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    output_directory: Annotated[Path, typer.Option()],
) -> None:
    """Export diffusion with self-contained Transformers custom code."""

    from lm_from_zero.export_diffusion_hf import (
        export_diffusion_to_hugging_face,
    )

    result = export_diffusion_to_hugging_face(
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


@app.command("generate-mamba2")
def generate_mamba2_command(
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
    """Generate locally from Mamba-2 with constant-size recurrent state."""

    import torch

    from lm_from_zero.generation import (
        CausalGenerationConfig,
        CausalGenerationEvent,
        append_generation_record,
        create_generation_record,
        generate_causal,
    )
    from lm_from_zero.models import Mamba2Config, Mamba2ForCausalLM
    from lm_from_zero.tokenizer.bpe import ByteBPE
    from lm_from_zero.training import load_checkpoint_model, validate_checkpoint

    checkpoint_manifest = validate_checkpoint(checkpoint)
    if checkpoint_manifest.binding.architecture != "mamba2":
        raise DataValidationError(
            "native Mamba-2 generation requires a Mamba-2 checkpoint"
        )
    model_config = Mamba2Config.model_validate(
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
    model = Mamba2ForCausalLM(model_config)
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


@app.command("generate-diffusion")
def generate_diffusion_command(
    checkpoint: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    tokenizer_training_manifest: Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    prompt: Annotated[str, typer.Argument()],
    response_length: Annotated[int, typer.Option(min=1)] = 64,
    diffusion_steps: Annotated[int | None, typer.Option(min=1)] = None,
    strategy: Annotated[Literal["greedy", "sample"], typer.Option()] = "greedy",
    reveal_schedule: Annotated[Literal["linear", "cosine"], typer.Option()] = "linear",
    temperature: Annotated[float, typer.Option(min=1e-12)] = 1.0,
    remask_strategy: Annotated[
        Literal["none", "low_confidence"], typer.Option()
    ] = "none",
    remask_fraction: Annotated[float, typer.Option(min=0, max=0.999999)] = 0.0,
    seed: Annotated[int, typer.Option()] = 1337,
    device: Annotated[Literal["cpu", "cuda"], typer.Option()] = "cpu",
    allow_raw_special_tokens: Annotated[bool, typer.Option()] = False,
    stream: Annotated[bool, typer.Option()] = False,
    jsonl_output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Generate locally by iteratively denoising a fixed response canvas."""

    import torch

    from lm_from_zero.generation import (
        DiffusionGenerationConfig,
        DiffusionGenerationEvent,
        append_diffusion_generation_record,
        create_diffusion_generation_record,
        generate_diffusion,
    )
    from lm_from_zero.models import (
        MaskedDiffusionConfig,
        MaskedDiffusionForMaskedLM,
    )
    from lm_from_zero.tokenizer.bpe import ByteBPE
    from lm_from_zero.training import load_checkpoint_model, validate_checkpoint

    checkpoint_manifest = validate_checkpoint(checkpoint)
    if checkpoint_manifest.binding.architecture != "masked_diffusion":
        raise DataValidationError(
            "native diffusion generation requires a masked-diffusion checkpoint"
        )
    model_config = MaskedDiffusionConfig.model_validate(
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
    model = MaskedDiffusionForMaskedLM(model_config)
    load_checkpoint_model(
        checkpoint,
        model=model,
        expected_binding=checkpoint_manifest.binding,
    )
    model.to(torch.device(device))
    generation_config = DiffusionGenerationConfig(
        strategy=strategy,
        response_length=response_length,
        diffusion_steps=diffusion_steps,
        reveal_schedule=reveal_schedule,
        temperature=temperature,
        remask_strategy=remask_strategy,
        remask_fraction=remask_fraction,
        seed=seed,
        allow_raw_special_tokens=allow_raw_special_tokens,
    )

    def emit(event: DiffusionGenerationEvent) -> None:
        if stream:
            typer.echo(
                json.dumps(
                    {"event": "reveal", **event.model_dump(mode="json")},
                    sort_keys=True,
                )
            )

    prompt_ids = tokenizer.encode(prompt)
    result = generate_diffusion(
        model,
        [prompt_ids],
        generation_config,
        on_step=emit,
    )
    if jsonl_output is not None:
        append_diffusion_generation_record(
            jsonl_output,
            create_diffusion_generation_record(
                result,
                [prompt_ids],
                source_checkpoint_id=checkpoint_manifest.lineage.checkpoint_id,
                source_checkpoint_manifest_sha256=sha256(
                    checkpoint_manifest.canonical_bytes()
                ).hexdigest(),
                device=device,
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


@app.command("build-mamba2-smoke-report")
def build_mamba2_smoke_report_command(
    training_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    checkpoint: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    evaluation_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    export_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    generation_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Generate portable Mamba-2 smoke evidence from validated artifacts."""

    from lm_from_zero.smoke_report import (
        build_mamba2_smoke_report,
        write_dense_smoke_report,
    )

    report = build_mamba2_smoke_report(
        training_jsonl=training_jsonl,
        checkpoint_directory=checkpoint,
        evaluation_jsonl=evaluation_jsonl,
        export_directory=export_directory,
        generation_jsonl=generation_jsonl,
    )
    write_dense_smoke_report(output, report)
    typer.echo(report.model_dump_json())


@app.command("build-diffusion-smoke-report")
def build_diffusion_smoke_report_command(
    training_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    checkpoint: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    evaluation_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    export_directory: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    generation_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Generate portable masked-diffusion smoke evidence."""

    from lm_from_zero.smoke_report import (
        build_diffusion_smoke_report,
        write_dense_smoke_report,
    )

    report = build_diffusion_smoke_report(
        training_jsonl=training_jsonl,
        checkpoint_directory=checkpoint,
        evaluation_jsonl=evaluation_jsonl,
        export_directory=export_directory,
        generation_jsonl=generation_jsonl,
    )
    write_dense_smoke_report(output, report)
    typer.echo(report.model_dump_json())


@app.command("compare-dense-resume")
def compare_dense_resume_command(
    uninterrupted_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    uninterrupted_checkpoint: Annotated[
        Path, typer.Argument(exists=True, file_okay=False)
    ],
    resumed_jsonl: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    resumed_checkpoint: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option()],
    parameter_atol: Annotated[float, typer.Option(min=1e-12)] = 1e-5,
    parameter_rtol: Annotated[float, typer.Option(min=1e-12)] = 1e-4,
    loss_atol: Annotated[float, typer.Option(min=1e-12)] = 1e-4,
) -> None:
    """Compare compiled-CUDA resumed training with an uninterrupted run."""

    from lm_from_zero.resume_tolerance import (
        ResumeToleranceThresholds,
        build_dense_resume_tolerance_report,
        write_dense_resume_tolerance_report,
    )

    report = build_dense_resume_tolerance_report(
        uninterrupted_jsonl=uninterrupted_jsonl,
        uninterrupted_checkpoint=uninterrupted_checkpoint,
        resumed_jsonl=resumed_jsonl,
        resumed_checkpoint=resumed_checkpoint,
        thresholds=ResumeToleranceThresholds(
            parameter_atol=parameter_atol,
            parameter_rtol=parameter_rtol,
            loss_atol=loss_atol,
        ),
    )
    write_dense_resume_tolerance_report(output, report)
    typer.echo(report.model_dump_json())
    if not report.passed:
        raise typer.Exit(code=1)


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
