from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError
from typer.testing import CliRunner

from lm_from_zero.acceleration_calibration import (
    ARCHITECTURES,
    AccelerationCalibrationError,
    AccelerationCalibrationPlan,
    CalibrationNumericalTrace,
    CalibrationResult,
    ParameterGroupMeasurement,
    build_report,
    compare_numerical_traces,
    create_plan,
    load_numerical_trace,
    load_plan,
    load_results,
    resolve_cell,
    write_artifact,
    write_numerical_trace,
)
from lm_from_zero.cli import app
from lm_from_zero.training.checkpointing import GitMetadata

SHARD_HASH = "a" * 64
TOKENIZER_HASH = "b" * 64
REVISION = "c" * 40
TRACE_HASH = "d" * 64


def _build() -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer_hash=TOKENIZER_HASH,
        tokenizer_vocab_size=16_000,
    )


class AccelerationCalibrationTests(unittest.TestCase):
    def _plan(self) -> AccelerationCalibrationPlan:
        with (
            patch(
                "lm_from_zero.acceleration_calibration.validate_shard_build",
                return_value=_build(),
            ),
            patch(
                "lm_from_zero.acceleration_calibration._file_sha256",
                return_value=SHARD_HASH,
            ),
        ):
            return create_plan("build.json", repository_revision=REVISION)

    def _results(
        self,
        plan: AccelerationCalibrationPlan,
    ) -> list[CalibrationResult]:
        results: list[CalibrationResult] = []
        baselines: dict[tuple[str, int], CalibrationResult] = {}
        for cell in plan.cells:
            for repetition in range(1, plan.repetitions + 1):
                base = 100.0 + (repetition - 2)
                if cell.cell_id == "sampled-telemetry":
                    throughput = base * 1.15
                else:
                    throughput = base if cell.cell_id == "baseline" else base * 1.05
                measured_tokens = (
                    plan.measured_optimizer_steps
                    * plan.sequence_length
                    * plan.micro_batch_size
                    * plan.gradient_accumulation_steps
                    * plan.world_size
                )
                result = CalibrationResult(
                    plan_sha256=plan.artifact_sha256,
                    cell_sha256=cell.artifact_sha256,
                    repository_revision=plan.repository_revision,
                    cuda_device_name=plan.expected_cuda_device_name,
                    cuda_compute_capability=(8, 9),
                    torch_version="2.13.0",
                    cuda_version="13.0",
                    architecture=cell.architecture,
                    cell_id=cell.cell_id,
                    repetition=repetition,
                    shard_manifest_sha256=plan.shard_manifest_sha256,
                    model_config_sha256=cell.model_config_sha256,
                    training_config_sha256="e" * 64,
                    seed=plan.seed,
                    sequence_length=plan.sequence_length,
                    micro_batch_size=plan.micro_batch_size,
                    gradient_accumulation_steps=plan.gradient_accumulation_steps,
                    world_size=plan.world_size,
                    warmup_optimizer_steps=plan.warmup_optimizer_steps,
                    measured_optimizer_steps=plan.measured_optimizer_steps,
                    measured_end_to_end_seconds=measured_tokens / throughput,
                    measured_cuda_compute_seconds=measured_tokens / throughput * 0.8,
                    tokens_per_second=throughput,
                    cold_compiled_step_seconds=1.0,
                    warm_compiled_step_seconds=0.1,
                    estimated_compile_overhead_seconds=0.9,
                    optimizer_seconds=5.0,
                    evaluation_seconds=0.0,
                    checkpoint_seconds=0.0,
                    metric_fsync_seconds=0.1,
                    cpu_data_seconds=1.0,
                    process_cpu_utilization=0.5,
                    peak_cuda_allocated_bytes=1_000,
                    peak_cuda_reserved_bytes=2_000,
                    graph_breaks=0,
                    observed_sdpa_backend=(
                        "flash" if cell.settings.sdpa_backend == "flash" else None
                    ),
                    baseline_result_sha256=(
                        None
                        if cell.cell_id == "baseline"
                        else baselines[(cell.architecture, repetition)].artifact_sha256
                    ),
                    numerical_trace_sha256=TRACE_HASH,
                    checkpoint_manifest_sha256=TRACE_HASH,
                    event_log_sha256=TRACE_HASH,
                    parameter_groups=(
                        ParameterGroupMeasurement(
                            name="decay",
                            parameter_count=1,
                            weight_rms=1.0,
                            gradient_rms=0.1,
                            update_rms=0.01,
                            angular_learning_rate=0.01,
                            effective_learning_rate=0.1,
                        ),
                    ),
                    maximum_loss_absolute_delta=(
                        None if cell.cell_id == "baseline" else 1e-5
                    ),
                    maximum_gradient_absolute_delta=(
                        None if cell.cell_id == "baseline" else 1e-4
                    ),
                    maximum_update_absolute_delta=(
                        None if cell.cell_id == "baseline" else 1e-4
                    ),
                )
                if cell.cell_id == "baseline":
                    baselines[(cell.architecture, repetition)] = result
                results.append(result)
        return results

    def test_plan_is_synchronized_staged_and_non_cartesian(self) -> None:
        plan = self._plan()

        self.assertEqual(plan.format_version, 3)
        self.assertEqual(len(plan.cells), 26)
        self.assertEqual(plan.sequence_length, 1_024)
        self.assertEqual(plan.warmup_optimizer_steps, 50)
        self.assertEqual(plan.measurement_mode, "fixed-steps")
        self.assertEqual(plan.measured_optimizer_steps, 500)
        self.assertEqual(plan.maximum_throughput_relative_spread, 0.05)
        self.assertEqual(plan.repetitions, 3)
        self.assertEqual(plan.repository_revision, REVISION)
        self.assertEqual(
            plan.artifact_root,
            "artifacts/acceleration-calibration/fixed-500-results",
        )
        expected_counts = {"dense": 9, "mamba2": 8, "diffusion": 9}
        for architecture in ARCHITECTURES:
            cells = [cell for cell in plan.cells if cell.architecture == architecture]
            self.assertEqual(len(cells), expected_counts[architecture])
            self.assertEqual(cells[0].cell_id, "baseline")
            self.assertEqual(cells[1].cell_id, "sampled-telemetry")
            for cell in cells[2:]:
                self.assertEqual(cell.parent_cell_id, "sampled-telemetry")
                differences = sum(
                    left != right
                    for left, right in zip(
                        cell.settings.model_dump().values(),
                        cells[1].settings.model_dump().values(),
                        strict=True,
                    )
                )
                expected_differences = (
                    2
                    if architecture == "diffusion" and cell.cell_id == "flash-sdpa"
                    else 1
                )
                self.assertEqual(differences, expected_differences)

        dense_ids = {
            cell.cell_id for cell in plan.cells if cell.architecture == "dense"
        }
        mamba_ids = {
            cell.cell_id for cell in plan.cells if cell.architecture == "mamba2"
        }
        self.assertIn("flash-sdpa", dense_ids)
        self.assertIn("linear-cross-entropy", dense_ids)
        self.assertNotIn("tf32-high", dense_ids)
        self.assertIn("tf32-high", mamba_ids)
        self.assertNotIn("flash-sdpa", mamba_ids)
        diffusion_flash = next(
            cell
            for cell in plan.cells
            if cell.architecture == "diffusion" and cell.cell_id == "flash-sdpa"
        )
        self.assertTrue(diffusion_flash.settings.diffusion_padding_free_attention)
        self.assertEqual(
            plan.canonical_json(),
            json.dumps(
                plan.model_dump(mode="json"),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def test_report_uses_architecture_medians_and_ten_percent_gate(self) -> None:
        plan = self._plan()
        report = build_report(plan, self._results(plan))

        self.assertEqual(report.plan_sha256, plan.artifact_sha256)
        self.assertEqual(report.result_count, 26 * 3)
        self.assertEqual(len(report.summaries), 26)
        for architecture in ARCHITECTURES:
            baseline = next(
                item
                for item in report.summaries
                if item.architecture == architecture and item.cell_id == "baseline"
            )
            sampled = next(
                item
                for item in report.summaries
                if item.architecture == architecture
                and item.cell_id == "sampled-telemetry"
            )
            fused = next(
                item
                for item in report.summaries
                if item.architecture == architecture and item.cell_id == "fused-adamw"
            )
            self.assertEqual(baseline.median_tokens_per_second, 100.0)
            self.assertAlmostEqual(sampled.median_tokens_per_second, 115.0)
            self.assertAlmostEqual(sampled.speedup_over_baseline, 1.15)
            self.assertTrue(sampled.numerical_parity_passed)
            self.assertTrue(sampled.throughput_stability_passed)
            self.assertTrue(sampled.promoted)
            self.assertFalse(fused.promoted)

    def test_promotion_rejects_unstable_repetition_throughput(self) -> None:
        plan = self._plan()
        results = self._results(plan)
        index = next(
            index
            for index, result in enumerate(results)
            if result.architecture == "dense"
            and result.cell_id == "sampled-telemetry"
            and result.repetition == 1
        )
        result = results[index]
        results[index] = result.model_copy(
            update={
                "tokens_per_second": 150.0,
                "measured_end_to_end_seconds": (
                    result.measured_end_to_end_seconds
                    * result.tokens_per_second
                    / 150.0
                ),
            }
        )

        report = build_report(plan, results)
        sampled = next(
            summary
            for summary in report.summaries
            if summary.architecture == "dense"
            and summary.cell_id == "sampled-telemetry"
        )

        self.assertFalse(sampled.throughput_stability_passed)
        self.assertFalse(sampled.promoted)

        results = self._results(plan)
        baseline_index = next(
            index
            for index, result in enumerate(results)
            if result.architecture == "dense"
            and result.cell_id == "baseline"
            and result.repetition == 1
        )
        baseline = results[baseline_index]
        unstable_baseline = baseline.model_copy(
            update={
                "tokens_per_second": 150.0,
                "measured_end_to_end_seconds": (
                    baseline.measured_end_to_end_seconds
                    * baseline.tokens_per_second
                    / 150.0
                ),
            }
        )
        results[baseline_index] = unstable_baseline
        for result_index, candidate in enumerate(results):
            if (
                candidate.architecture == "dense"
                and candidate.cell_id != "baseline"
                and candidate.repetition == 1
            ):
                results[result_index] = candidate.model_copy(
                    update={"baseline_result_sha256": unstable_baseline.artifact_sha256}
                )

        report = build_report(plan, results)
        sampled = next(
            summary
            for summary in report.summaries
            if summary.architecture == "dense"
            and summary.cell_id == "sampled-telemetry"
        )
        self.assertFalse(sampled.throughput_stability_passed)
        self.assertFalse(sampled.promoted)

    def test_rejects_missing_duplicate_mismatch_and_missing_parity(self) -> None:
        plan = self._plan()
        results = self._results(plan)

        with self.assertRaisesRegex(AccelerationCalibrationError, "incomplete"):
            build_report(plan, results[:-1])
        with self.assertRaisesRegex(AccelerationCalibrationError, "duplicate"):
            build_report(plan, [*results, results[0]])

        mismatched = results.copy()
        mismatched[0] = mismatched[0].model_copy(update={"seed": plan.seed + 1})
        with self.assertRaisesRegex(AccelerationCalibrationError, "binding mismatch"):
            build_report(plan, mismatched)

        wrong_device = results.copy()
        wrong_device[0] = wrong_device[0].model_copy(
            update={"cuda_device_name": "NVIDIA H100 80GB HBM3"}
        )
        with self.assertRaisesRegex(AccelerationCalibrationError, "binding mismatch"):
            build_report(plan, wrong_device)

        missing_flash = self._results(plan)
        flash_index = next(
            index
            for index, item in enumerate(missing_flash)
            if item.cell_id == "flash-sdpa"
        )
        missing_flash[flash_index] = missing_flash[flash_index].model_copy(
            update={"observed_sdpa_backend": "math"}
        )
        with self.assertRaisesRegex(AccelerationCalibrationError, "profiler-confirmed"):
            build_report(plan, missing_flash)

        no_parity = results.copy()
        candidate_index = next(
            index for index, item in enumerate(no_parity) if item.cell_id != "baseline"
        )
        no_parity[candidate_index] = no_parity[candidate_index].model_copy(
            update={"maximum_loss_absolute_delta": None}
        )
        with self.assertRaisesRegex(AccelerationCalibrationError, "parity evidence"):
            build_report(plan, no_parity)

        wrong_baseline = self._results(plan)
        candidate_index = next(
            index
            for index, item in enumerate(wrong_baseline)
            if item.cell_id != "baseline"
        )
        wrong_baseline[candidate_index] = wrong_baseline[candidate_index].model_copy(
            update={"baseline_result_sha256": "f" * 64}
        )
        with self.assertRaisesRegex(AccelerationCalibrationError, "wrong baseline"):
            build_report(plan, wrong_baseline)

    def test_rejects_nonmatching_measurement_and_noncanonical_matrix(self) -> None:
        plan = self._plan()
        results = self._results(plan)
        results[0] = results[0].model_copy(
            update={
                "measured_optimizer_steps": plan.measured_optimizer_steps - 1,
            }
        )
        with self.assertRaisesRegex(AccelerationCalibrationError, "fixed-step"):
            build_report(plan, results)

        results = self._results(plan)
        results[0] = results[0].model_copy(
            update={"measured_optimizer_steps": plan.measured_optimizer_steps + 1}
        )
        with self.assertRaisesRegex(AccelerationCalibrationError, "fixed-step"):
            build_report(plan, results)

        payload = plan.model_dump(mode="json")
        payload["cells"][0]["settings"]["compile_mode"] = "disabled"
        with self.assertRaisesRegex(ValidationError, "staged settings"):
            AccelerationCalibrationPlan.model_validate(payload)

        for field, value in (
            ("warmup_optimizer_steps", 49),
            ("measured_optimizer_steps", 499),
            ("repetitions", 2),
        ):
            payload = plan.model_dump(mode="json")
            payload[field] = value
            with self.subTest(field=field), self.assertRaises(ValidationError):
                AccelerationCalibrationPlan.model_validate(payload)

        result_payload = self._results(plan)[0].model_dump(mode="json")
        result_payload["unexpected"] = True
        with self.assertRaises(ValidationError):
            CalibrationResult.model_validate(result_payload)

    def test_atomic_write_and_incomplete_artifact_rejection(self) -> None:
        plan = self._plan()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "plan.json"
            write_artifact(output, plan)
            self.assertEqual(
                output.read_text(encoding="utf-8"), plan.canonical_json() + "\n"
            )
            self.assertEqual(load_plan(output), plan)
            self.assertEqual(resolve_cell(plan, "dense", "baseline").stage, "baseline")
            with self.assertRaisesRegex(AccelerationCalibrationError, "not planned"):
                resolve_cell(plan, "dense", "missing")

            results = Path(directory) / "results"
            result = self._results(plan)[0]
            repetition = results / "dense" / "baseline" / "repetition-01"
            write_artifact(repetition / "result.json", result)
            (repetition / "numerical-trace.json").write_text("{}", encoding="utf-8")
            checkpoint = repetition / "checkpoints" / "step-000000000150"
            checkpoint.mkdir(parents=True)
            (checkpoint / "manifest.json").write_text("{}", encoding="utf-8")
            self.assertEqual(load_results(results), (result,))
            (Path(directory) / "empty").mkdir()
            with self.assertRaisesRegex(AccelerationCalibrationError, "empty"):
                load_results(Path(directory) / "empty")

            temporary = output.with_name(f".{output.name}.tmp")
            temporary.write_text("incomplete", encoding="utf-8")
            with self.assertRaisesRegex(AccelerationCalibrationError, "incomplete"):
                write_artifact(output, plan)

    def test_numerical_traces_are_bound_canonical_and_compared(self) -> None:
        plan = self._plan()
        baseline_cell = resolve_cell(plan, "dense", "baseline")
        candidate_cell = resolve_cell(plan, "dense", "sampled-telemetry")
        baseline = CalibrationNumericalTrace(
            plan_sha256=plan.artifact_sha256,
            cell_sha256=baseline_cell.artifact_sha256,
            repository_revision=plan.repository_revision,
            architecture="dense",
            cell_id="baseline",
            repetition=1,
            loss_values=(2.0, 1.0),
            gradient_norm_values=(0.5, 0.25),
            update_rms_values=(0.01, 0.02),
        )
        candidate = CalibrationNumericalTrace(
            plan_sha256=plan.artifact_sha256,
            cell_sha256=candidate_cell.artifact_sha256,
            repository_revision=plan.repository_revision,
            architecture="dense",
            cell_id="sampled-telemetry",
            repetition=1,
            loss_values=(2.001, 0.999),
            gradient_norm_values=(0.499, 0.252),
            update_rms_values=(0.011, 0.018),
        )
        loss, gradient, update = compare_numerical_traces(baseline, candidate)
        self.assertAlmostEqual(loss, 0.001)
        self.assertAlmostEqual(gradient, 0.002)
        self.assertAlmostEqual(update, 0.002)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            write_numerical_trace(path, baseline)
            self.assertEqual(load_numerical_trace(path), baseline)

        with self.assertRaisesRegex(AccelerationCalibrationError, "bindings"):
            compare_numerical_traces(
                baseline,
                candidate.model_copy(update={"repository_revision": "f" * 40}),
            )

    def test_cpu_only_plan_and_inspect_cli_commands(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build = root / "build.json"
            build.write_text("{}", encoding="utf-8")
            output = root / "plan.json"
            with (
                patch(
                    "lm_from_zero.acceleration_calibration.validate_shard_build",
                    return_value=_build(),
                ),
                patch(
                    "lm_from_zero.acceleration_calibration._file_sha256",
                    return_value=SHARD_HASH,
                ),
                patch(
                    "lm_from_zero.acceleration_execution.inspect_repository_state",
                    return_value=GitMetadata(revision=REVISION, dirty=False),
                ),
            ):
                planned = CliRunner().invoke(
                    app,
                    [
                        "plan-acceleration-calibration",
                        str(build),
                        "--output",
                        str(output),
                    ],
                )
            self.assertEqual(planned.exit_code, 0, planned.output)
            self.assertEqual(json.loads(planned.stdout)["format_version"], 3)

            inspected = CliRunner().invoke(
                app,
                [
                    "inspect-acceleration-calibration-cell",
                    str(output),
                    "diffusion",
                    "flash-sdpa",
                ],
            )
            self.assertEqual(inspected.exit_code, 0, inspected.output)
            self.assertEqual(json.loads(inspected.stdout)["stage"], "attention")


if __name__ == "__main__":
    unittest.main()
