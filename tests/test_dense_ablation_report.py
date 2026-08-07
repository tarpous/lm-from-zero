from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from lm_from_zero.dense_ablation_report import (
    DenseAblationReportError,
    DenseAblationSeedResult,
    _load_events,
    _load_run_outputs,
    _load_seed_result,
    _metric_summary,
    build_dense_ablation_downstream_report,
    build_dense_ablation_report,
    write_dense_ablation_downstream_report,
    write_dense_ablation_report,
)
from lm_from_zero.dense_ablations import (
    DenseAblationPlan,
    DenseAblationSeedPlan,
    DenseAblationVariantSpec,
    DenseModelVariant,
    OptimizerVariant,
    VariantChange,
    VariantName,
)
from lm_from_zero.evaluation import CausalEvaluationResult
from lm_from_zero.generation.causal import (
    CausalGenerationRecord,
    CausalGenerationResult,
)
from lm_from_zero.training import (
    BatchCursor,
    DenseRunPlan,
    DenseTrainingResult,
    OptimizerStepMetrics,
)

SEEDS = (1337, 2027, 3407)


def _plan() -> DenseAblationPlan:
    variant_data: tuple[
        tuple[VariantName, VariantChange, str, DenseModelVariant, OptimizerVariant], ...
    ] = (
        ("baseline", "none", "Canonical baseline.", "baseline", "adamw"),
        (
            "learned_absolute_positions",
            "rope_to_learned_absolute_positions",
            "Learned positions.",
            "learned_absolute_positions",
            "adamw",
        ),
        ("layer_norm", "rmsnorm_to_layernorm", "LayerNorm.", "layer_norm", "adamw"),
        ("gelu", "swiglu_to_gelu", "GELU.", "gelu", "adamw"),
        ("mha", "gqa_to_mha", "MHA.", "mha", "adamw"),
        (
            "without_qk_norm",
            "remove_qk_normalization",
            "No QK norm.",
            "without_qk_norm",
            "adamw",
        ),
        (
            "tied_embeddings",
            "tie_input_output_embeddings",
            "Tied embeddings.",
            "tied_embeddings",
            "adamw",
        ),
        (
            "hybrid_muon",
            "adamw_to_hybrid_muon",
            "Hybrid Muon.",
            "baseline",
            "hybrid_muon",
        ),
    )
    variants = tuple(
        DenseAblationVariantSpec(
            name=name,
            change=change,
            description=description,
            execution_status=(
                "reuse_m7_screening"
                if name == "baseline"
                else "ready_for_bounded_gpu_smoke"
            ),
            model_variant=model_variant,
            optimizer_variant=optimizer_variant,
        )
        for name, change, description, model_variant, optimizer_variant in variant_data
    )
    jobs = tuple(
        DenseAblationSeedPlan(
            variant=variant.name,
            variant_spec_sha256=variant.spec_sha256,
            seed=seed,
            execution_status=(
                "requires_m7_checkpoint_recovery"
                if variant.name == "baseline"
                else "ready_for_bounded_gpu_smoke"
            ),
            model_variant=variant.model_variant,
            optimizer_variant=variant.optimizer_variant,
            artifact_directory=f"artifacts/{variant.name}/seed-{seed}",
            checkpoint_directory=f"artifacts/{variant.name}/seed-{seed}/checkpoints",
            jsonl_log=f"artifacts/{variant.name}/seed-{seed}/events.jsonl",
            expected_optimizer_step=4,
            expected_tokens=32_768,
        )
        for variant in variants
        for seed in SEEDS
    )
    return DenseAblationPlan(
        source_architecture_study_plan_sha256="a" * 64,
        shard_manifest_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        sequence_length=1_024,
        micro_batch_size=8,
        gradient_accumulation_steps=1,
        dense_reference_tokens=100_000_000,
        expected_screening_optimizer_step=4,
        expected_screening_tokens=32_768,
        seeds=SEEDS,
        variants=variants,
        jobs=jobs,
    )


def _run_outputs() -> tuple[DenseRunPlan, DenseTrainingResult]:
    cursor = BatchCursor(
        build_manifest_sha256="b" * 64,
        tokenizer_hash="c" * 64,
        split="train",
        sequence_length=1_024,
        seed=1_337,
        rank=0,
        world_size=1,
        shuffle=True,
        next_local_window=32,
        sequences_consumed=32,
        tokens_consumed=32_768,
    )
    metric = OptimizerStepMetrics(
        optimizer_step=4,
        measurement_optimizer_steps=4,
        loss=1.5,
        learning_rate=0.001,
        gradient_norm=0.8,
        tokens_consumed=32_768,
        elapsed_seconds=1.0,
        tokens_per_second=32_768.0,
        peak_cuda_memory_reserved_bytes=2_000,
    )
    plan = DenseRunPlan(
        training_config_sha256="d" * 64,
        model_config_sha256="e" * 64,
        shard_manifest_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        seed=1_337,
        parameter_count=20_000_000,
        optimizer_steps=4,
        micro_batches_per_step=1,
        world_size=1,
        local_tokens_per_optimizer_step=8_192,
        tokens_per_optimizer_step=8_192,
        total_training_tokens=32_768,
        estimated_training_flops=1,
        reference_training_flops=None,
        training_flop_ratio=None,
        estimated_checkpoint_bytes=1,
        estimated_retained_checkpoint_bytes_upper_bound=1,
        estimated_seconds=None,
        checkpoint_directory="checkpoints",
        jsonl_log="events.jsonl",
        tensorboard_directory=None,
        parquet_log=None,
        device="cpu",
        precision="fp32",
        compile_model=False,
        compile_mode="default",
        adamw_backend="auto",
        model_variant="gelu",
        optimizer_variant="adamw",
        loss_backend="full",
        sdpa_backend="auto",
        float32_matmul_precision="highest",
        telemetry_every_steps=1,
        checkpoint_every_steps=None,
        checkpoint_every_seconds=900.0,
    )
    return plan, DenseTrainingResult(
        optimizer_step=4,
        cursor=cursor,
        last_checkpoint=Path("checkpoints/step-000000000004"),
        metrics=(metric,),
    )


def _events() -> tuple[dict[str, Any], ...]:
    recorded_at = datetime(2026, 8, 7, tzinfo=UTC).isoformat()
    return (
        {
            "event": "run_start",
            "optimizer_step": 0,
            "recorded_at_utc": recorded_at,
            "training_config": {
                "seed": 1_337,
                "model_variant": "gelu",
                "optimizer_variant": "adamw",
            },
            "training_config_sha256": "d" * 64,
        },
        {
            "event": "optimizer_step",
            "recorded_at_utc": recorded_at,
            "optimizer_step": 4,
            "measurement_optimizer_steps": 4,
            "loss": 1.5,
            "learning_rate": 0.001,
            "gradient_norm": 0.8,
            "tokens_consumed": 32_768,
            "elapsed_seconds": 1.0,
            "tokens_per_second": 32_768.0,
            "peak_cuda_memory_reserved_bytes": 2_000,
        },
        {
            "event": "run_complete",
            "optimizer_step": 4,
            "tokens_consumed": 32_768,
            "checkpoint": "step-000000000004",
            "recorded_at_utc": recorded_at,
        },
    )


def _checkpoint() -> Any:
    return SimpleNamespace(
        progress=SimpleNamespace(optimizer_step=4, tokens_consumed=32_768),
        binding=SimpleNamespace(
            model_config_sha256="e" * 64,
            shard_manifest_sha256="b" * 64,
            tokenizer_sha256="c" * 64,
            git=SimpleNamespace(revision="f" * 40, dirty=False),
        ),
        lineage=SimpleNamespace(checkpoint_id="step-000000000004"),
        model_artifact=SimpleNamespace(sha256="2" * 64),
        recovery_artifact=SimpleNamespace(sha256="3" * 64),
    )


def _validation_cursor(*, next_window: int, sequences: int, tokens: int) -> BatchCursor:
    return BatchCursor(
        build_manifest_sha256="b" * 64,
        tokenizer_hash="c" * 64,
        split="validation",
        sequence_length=1_024,
        seed=1_337,
        rank=0,
        world_size=1,
        shuffle=False,
        next_local_window=next_window,
        sequences_consumed=sequences,
        tokens_consumed=tokens,
    )


def _downstream_evaluation() -> CausalEvaluationResult:
    return CausalEvaluationResult(
        evaluated_at_utc=datetime(2026, 8, 7, tzinfo=UTC),
        split="validation",
        model_config_sha256="e" * 64,
        shard_manifest_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        batch_count=24,
        sequence_count=192,
        predicted_token_count=196_416,
        mean_loss=1.7,
        perplexity=5.5,
        elapsed_seconds=1.0,
        predicted_tokens_per_second=196_416.0,
        cursor_before=_validation_cursor(next_window=0, sequences=0, tokens=0),
        cursor_after=_validation_cursor(
            next_window=192,
            sequences=192,
            tokens=196_608,
        ),
    )


def _downstream_generation() -> CausalGenerationRecord:
    return CausalGenerationRecord(
        generated_at_utc=datetime(2026, 8, 7, tzinfo=UTC),
        model_config_sha256="e" * 64,
        tokenizer_sha256="c" * 64,
        prompt_token_sha256="1" * 64,
        result=CausalGenerationResult(
            prompt_token_counts=(4,),
            generated_token_ids=(tuple(range(16)),),
            stop_reasons=("max_new_tokens",),
            model_forwards=16,
            generated_token_count=16,
            elapsed_seconds=1.0,
            tokens_per_second=16.0,
        ),
    )


def _seed_result(job: DenseAblationSeedPlan) -> DenseAblationSeedResult:
    base_loss = {
        "learned_absolute_positions": 4.0,
        "layer_norm": 2.0,
        "gelu": 3.0,
        "mha": 1.0,
        "without_qk_norm": 5.0,
        "tied_embeddings": 2.5,
        "hybrid_muon": 0.5,
    }[job.variant]
    throughput = {
        "learned_absolute_positions": 20.0,
        "layer_norm": 30.0,
        "gelu": 40.0,
        "mha": 10.0,
        "without_qk_norm": 25.0,
        "tied_embeddings": 60.0,
        "hybrid_muon": 50.0,
    }[job.variant]
    return DenseAblationSeedResult(
        variant=job.variant,
        change="none" if job.variant == "baseline" else "swiglu_to_gelu",
        model_variant=job.model_variant,
        optimizer_variant=job.optimizer_variant,
        variant_spec_sha256=job.variant_spec_sha256,
        seed=job.seed,
        parameter_count=20_000_000,
        optimizer_step=4,
        tokens_consumed=32_768,
        terminal_loss=base_loss + job.seed / 10_000_000,
        terminal_gradient_norm=1.0,
        terminal_tokens_per_second=throughput,
        peak_cuda_memory_reserved_bytes=2_000,
        training_config_sha256="d" * 64,
        model_config_sha256="e" * 64,
        shard_manifest_sha256="b" * 64,
        tokenizer_sha256="c" * 64,
        git_revision="f" * 40,
        git_dirty=False,
        checkpoint_id="step-000000000004",
        checkpoint_manifest_sha256="1" * 64,
        model_artifact_sha256="2" * 64,
        recovery_artifact_sha256="3" * 64,
        events_sha256="4" * 64,
        artifact_directory=f"artifacts/{job.variant}/seed-{job.seed}",
        checkpoint_directory=(
            f"artifacts/{job.variant}/seed-{job.seed}/checkpoints/step-000000000004"
        ),
        events_jsonl=f"artifacts/{job.variant}/seed-{job.seed}/events.jsonl",
    )


class DenseAblationReportTests(unittest.TestCase):
    def test_loaders_validate_canonical_events_and_utf16_run_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            events_path = root / "events.jsonl"
            events_path.write_text(
                "".join(
                    f"{json.dumps(item, sort_keys=True, separators=(',', ':'))}\n"
                    for item in _events()
                ),
                encoding="utf-8",
            )
            self.assertEqual(len(_load_events(events_path)), 3)

            plan, result = _run_outputs()
            output_path = root / "result.jsonl"
            output_path.write_bytes(
                (
                    plan.model_dump_json() + "\n" + result.model_dump_json() + "\n"
                ).encode("utf-16")
            )
            loaded_plan, loaded_result = _load_run_outputs(output_path)
            self.assertEqual(loaded_plan.optimizer_steps, 4)
            self.assertEqual(loaded_result.optimizer_step, 4)

            with self.assertRaisesRegex(DenseAblationReportError, "not canonical"):
                events_path.write_text('{"event": "run_start"}\n', encoding="utf-8")
                _load_events(events_path)

            with self.assertRaisesRegex(DenseAblationReportError, "event log"):
                _load_events(root / "missing-events.jsonl")
            events_path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(DenseAblationReportError, "empty"):
                _load_events(events_path)
            events_path.write_text("not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(DenseAblationReportError, "invalid"):
                _load_events(events_path)
            with self.assertRaisesRegex(DenseAblationReportError, "empty metric"):
                _metric_summary([])

            output_path.write_text("one-line\n", encoding="utf-8")
            with self.assertRaisesRegex(DenseAblationReportError, "one plan"):
                _load_run_outputs(output_path)

    def test_load_seed_result_binds_run_and_checkpoint_metadata(self) -> None:
        plan = _plan()
        job = next(
            item for item in plan.jobs if item.variant == "gelu" and item.seed == 1_337
        )
        variant_spec = next(item for item in plan.variants if item.name == "gelu")
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "lm_from_zero.dense_ablation_report._load_events",
                return_value=_events(),
            ),
            patch(
                "lm_from_zero.dense_ablation_report._load_run_outputs",
                return_value=_run_outputs(),
            ),
            patch(
                "lm_from_zero.dense_ablation_report.validate_checkpoint",
                return_value=_checkpoint(),
            ),
            patch(
                "lm_from_zero.dense_ablation_report._sha256_file",
                return_value="1" * 64,
            ),
            patch(
                "lm_from_zero.dense_ablation_report._path_text",
                side_effect=lambda path: path.as_posix(),
            ),
        ):
            result = _load_seed_result(
                plan=plan,
                job=job,
                variant_spec=variant_spec,
                artifact_root=Path(directory),
            )
        self.assertEqual(result.variant, "gelu")
        self.assertEqual(result.optimizer_step, 4)
        self.assertEqual(result.terminal_loss, 1.5)

    def test_aggregates_and_selects_deterministically(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text(plan.canonical_json() + "\n", encoding="utf-8")
            with patch(
                "lm_from_zero.dense_ablation_report._load_seed_result",
                side_effect=lambda **kwargs: _seed_result(kwargs["job"]),
            ):
                report = build_dense_ablation_report(plan_path, root)

            self.assertEqual(report.job_count, 21)
            self.assertEqual(
                report.finalists_by_terminal_loss, ("hybrid_muon", "mha", "layer_norm")
            )
            self.assertEqual(report.fastest_variant, "tied_embeddings")
            self.assertEqual(
                report.recommended_variants,
                ("hybrid_muon", "mha", "layer_norm", "tied_embeddings"),
            )
            output = root / "report.json"
            write_dense_ablation_report(output, report)
            self.assertEqual(output.read_bytes(), report.canonical_bytes() + b"\n")
            temporary = output.with_name(f".{output.name}.tmp")
            temporary.write_text("incomplete", encoding="utf-8")
            with self.assertRaisesRegex(DenseAblationReportError, "incomplete"):
                write_dense_ablation_report(output, report)

    def test_builds_downstream_report_from_canonical_records(self) -> None:
        plan = _plan()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path = root / "plan.json"
            plan_path.write_text(plan.canonical_json() + "\n", encoding="utf-8")
            with patch(
                "lm_from_zero.dense_ablation_report._load_seed_result",
                side_effect=lambda **kwargs: _seed_result(kwargs["job"]),
            ):
                ablation_report = build_dense_ablation_report(plan_path, root)

            ablation_report_path = root / "m8-report.json"
            write_dense_ablation_report(ablation_report_path, ablation_report)
            evaluation_root = root / "evaluations"
            generation_root = root / "generations"
            evaluation_root.mkdir()
            generation_root.mkdir()
            for variant in ablation_report.recommended_variants:
                slug = variant.replace("_", "-")
                (evaluation_root / f"m8-{slug}-1337-v24.jsonl").write_text(
                    _downstream_evaluation().canonical_json() + "\n", encoding="utf-8"
                )
                (generation_root / f"m8-{slug}-1337.jsonl").write_text(
                    _downstream_generation().canonical_json() + "\n", encoding="utf-8"
                )

            report = build_dense_ablation_downstream_report(
                ablation_report_path,
                evaluation_root,
                generation_root,
                seed=1_337,
            )
            self.assertEqual(
                tuple(item.variant for item in report.records),
                ("hybrid_muon", "mha", "layer_norm", "tied_embeddings"),
            )
            self.assertTrue(
                all(
                    item.evaluation.predicted_token_count == 196_416
                    and item.generation.result.generated_token_count == 16
                    for item in report.records
                )
            )
            output = root / "downstream.json"
            write_dense_ablation_downstream_report(output, report)
            self.assertEqual(output.read_bytes(), report.canonical_bytes() + b"\n")
            temporary = output.with_name(f".{output.name}.tmp")
            temporary.write_text("incomplete", encoding="utf-8")
            with self.assertRaisesRegex(DenseAblationReportError, "incomplete"):
                write_dense_ablation_downstream_report(output, report)


if __name__ == "__main__":
    unittest.main()
