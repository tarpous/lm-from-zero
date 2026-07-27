import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from safetensors.torch import save_file
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.models import Olmo2Config
from lm_from_zero.resume_tolerance import (
    ResumeToleranceError,
    ResumeToleranceThresholds,
    build_dense_resume_tolerance_report,
    write_dense_resume_tolerance_report,
)
from lm_from_zero.training import (
    CausalBatchConfig,
    DenseTrainingConfig,
    OptimizationConfig,
)
from lm_from_zero.training.metrics import append_training_event

MODEL_HASH = "2" * 64
TOKENIZER_HASH = "3" * 64
SHARD_HASH = "4" * 64
GIT_REVISION = "5" * 40
RECORDED_AT = datetime(2026, 7, 27, 3, 0, tzinfo=UTC)
TRAINING_CONFIG = DenseTrainingConfig(
    model=Olmo2Config(
        model_name="resume-tolerance-test",
        tokenizer_hash=TOKENIZER_HASH,
        vocab_size=272,
        num_hidden_layers=1,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=32,
        max_position_embeddings=8,
    ),
    batch=CausalBatchConfig(sequence_length=4, micro_batch_size=1),
    optimization=OptimizationConfig(total_steps=2),
    device="cuda",
    precision="bf16",
    compile_model=True,
)
TRAINING_HASH = TRAINING_CONFIG.config_hash


def _training_config() -> dict[str, object]:
    return TRAINING_CONFIG.model_dump(mode="json")


def _step(step: int, loss: float) -> dict[str, object]:
    return {
        "elapsed_seconds": 0.5,
        "event": "optimizer_step",
        "gradient_norm": 1.0,
        "learning_rate": 0.001,
        "loss": loss,
        "optimizer_step": step,
        "peak_cuda_memory_allocated_bytes": 100,
        "peak_cuda_memory_reserved_bytes": 200,
        "tokens_consumed": step * 8,
        "tokens_per_second": 16.0,
    }


def _write_histories(root: Path) -> tuple[Path, Path]:
    uninterrupted = root / "uninterrupted.jsonl"
    resumed = root / "resumed.jsonl"
    start = {
        "event": "run_start",
        "optimizer_step": 0,
        "training_config": _training_config(),
        "training_config_sha256": TRAINING_HASH,
    }
    append_training_event(uninterrupted, start, recorded_at_utc=RECORDED_AT)
    append_training_event(
        uninterrupted,
        _step(1, 2.0),
        recorded_at_utc=RECORDED_AT,
    )
    append_training_event(
        uninterrupted,
        _step(2, 1.0),
        recorded_at_utc=RECORDED_AT,
    )

    append_training_event(resumed, start, recorded_at_utc=RECORDED_AT)
    append_training_event(resumed, _step(1, 2.0), recorded_at_utc=RECORDED_AT)
    append_training_event(
        resumed,
        {"event": "run_stopped", "optimizer_step": 1},
        recorded_at_utc=RECORDED_AT,
    )
    append_training_event(
        resumed,
        {
            "event": "run_resume",
            "optimizer_step": 1,
            "training_config": _training_config(),
            "training_config_sha256": TRAINING_HASH,
        },
        recorded_at_utc=RECORDED_AT,
    )
    append_training_event(resumed, _step(2, 1.0), recorded_at_utc=RECORDED_AT)
    return uninterrupted, resumed


def _manifest(*, resumed: bool, dirty: bool = False) -> MagicMock:
    manifest = MagicMock()
    manifest.canonical_bytes.return_value = (
        b"resumed manifest" if resumed else b"uninterrupted manifest"
    )
    manifest.created_at_utc = RECORDED_AT + timedelta(minutes=int(resumed))
    manifest.binding.git.dirty = dirty
    manifest.binding.git.revision = GIT_REVISION
    manifest.binding.model_config_sha256 = MODEL_HASH
    manifest.binding.tokenizer_sha256 = TOKENIZER_HASH
    manifest.binding.shard_manifest_sha256 = SHARD_HASH
    manifest.binding.rank = 0
    manifest.binding.world_size = 1
    manifest.binding.runtime.cuda_available = True
    manifest.binding.runtime.cuda_version = "13.0"
    manifest.binding.runtime.cuda_device_names = ("Test CUDA Device",)
    manifest.binding.runtime.torch_version = "2.9.0"
    manifest.binding.runtime.python_version = "3.12.3"
    manifest.binding.runtime.operating_system = "Linux"
    manifest.binding.runtime.machine = "x86_64"
    manifest.binding.runtime.dependency_versions = {
        "numpy": "2.4.3",
        "safetensors": "0.8.0",
        "torch": "2.9.0",
    }
    manifest.progress.optimizer_step = 2
    manifest.progress.tokens_consumed = 16
    manifest.lineage.checkpoint_id = "step-000000000002"
    manifest.lineage.parent_checkpoint_id = "step-000000000001" if resumed else None
    manifest.model_artifact.filename = "model.safetensors"
    manifest.model_artifact.sha256 = ("7" if resumed else "6") * 64
    return manifest


def _checkpoint(root: Path, name: str, values: torch.Tensor) -> Path:
    directory = root / name
    directory.mkdir()
    save_file(
        {
            "bias": torch.tensor([0.0]),
            "weight": values,
        },
        directory / "model.safetensors",
    )
    return directory


class DenseResumeToleranceTests(unittest.TestCase):
    def test_exact_comparison_writes_canonical_report_and_cli_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted_log, resumed_log = _write_histories(root)
            values = torch.tensor([1.0, 2.0])
            uninterrupted = _checkpoint(root, "uninterrupted", values)
            resumed = _checkpoint(root, "resumed", values.clone())
            manifests = (_manifest(resumed=False), _manifest(resumed=True))
            with patch(
                "lm_from_zero.resume_tolerance.validate_checkpoint",
                side_effect=manifests,
            ):
                report = build_dense_resume_tolerance_report(
                    uninterrupted_jsonl=uninterrupted_log,
                    uninterrupted_checkpoint=uninterrupted,
                    resumed_jsonl=resumed_log,
                    resumed_checkpoint=resumed,
                )

            self.assertTrue(report.passed)
            self.assertEqual(report.compared_tensors, 2)
            self.assertEqual(report.compared_values, 3)
            self.assertEqual(report.exact_equal_values, 3)
            self.assertEqual(report.tolerance_violation_values, 0)
            self.assertEqual(report.parameter_max_absolute_error, 0)
            self.assertEqual(report.loss_max_absolute_error, 0)

            output = root / "report.json"
            write_dense_resume_tolerance_report(output, report)
            self.assertEqual(output.read_bytes(), report.canonical_bytes() + b"\n")
            partial = output.with_name(f".{output.name}.tmp")
            partial.write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(ResumeToleranceError, "incomplete"):
                write_dense_resume_tolerance_report(output, report)

            cli_output = root / "cli-report.json"
            with patch(
                "lm_from_zero.resume_tolerance.validate_checkpoint",
                side_effect=(_manifest(resumed=False), _manifest(resumed=True)),
            ):
                result = CliRunner().invoke(
                    app,
                    [
                        "compare-dense-resume",
                        str(uninterrupted_log),
                        str(uninterrupted),
                        str(resumed_log),
                        str(resumed),
                        "--output",
                        str(cli_output),
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertTrue(json.loads(result.stdout)["passed"])
            self.assertTrue(cli_output.is_file())

    def test_reports_within_tolerance_and_fails_outside_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted_log, resumed_log = _write_histories(root)
            uninterrupted = _checkpoint(
                root,
                "uninterrupted",
                torch.tensor([1.0, 2.0]),
            )
            resumed = _checkpoint(
                root,
                "resumed",
                torch.tensor([1.00005, 2.0]),
            )
            thresholds = ResumeToleranceThresholds(
                parameter_atol=1e-4,
                parameter_rtol=1e-4,
                loss_atol=1e-4,
            )
            with patch(
                "lm_from_zero.resume_tolerance.validate_checkpoint",
                side_effect=(_manifest(resumed=False), _manifest(resumed=True)),
            ):
                accepted = build_dense_resume_tolerance_report(
                    uninterrupted_jsonl=uninterrupted_log,
                    uninterrupted_checkpoint=uninterrupted,
                    resumed_jsonl=resumed_log,
                    resumed_checkpoint=resumed,
                    thresholds=thresholds,
                )
            self.assertTrue(accepted.passed)
            self.assertLess(accepted.exact_equal_values, accepted.compared_values)

            save_file(
                {
                    "bias": torch.tensor([0.0]),
                    "weight": torch.tensor([1.1, 2.0]),
                },
                resumed / "model.safetensors",
            )
            with patch(
                "lm_from_zero.resume_tolerance.validate_checkpoint",
                side_effect=(_manifest(resumed=False), _manifest(resumed=True)),
            ):
                rejected = build_dense_resume_tolerance_report(
                    uninterrupted_jsonl=uninterrupted_log,
                    uninterrupted_checkpoint=uninterrupted,
                    resumed_jsonl=resumed_log,
                    resumed_checkpoint=resumed,
                    thresholds=thresholds,
                )
            self.assertFalse(rejected.passed)
            self.assertEqual(rejected.tolerance_violation_values, 1)
            self.assertEqual(
                rejected.tensors_with_tolerance_violations,
                ("weight",),
            )
            cli_output = root / "failed-report.json"
            with patch(
                "lm_from_zero.resume_tolerance.validate_checkpoint",
                side_effect=(_manifest(resumed=False), _manifest(resumed=True)),
            ):
                result = CliRunner().invoke(
                    app,
                    [
                        "compare-dense-resume",
                        str(uninterrupted_log),
                        str(uninterrupted),
                        str(resumed_log),
                        str(resumed),
                        "--output",
                        str(cli_output),
                        "--parameter-atol",
                        "0.0001",
                        "--parameter-rtol",
                        "0.0001",
                    ],
                )
            self.assertEqual(result.exit_code, 1, result.output)
            self.assertFalse(json.loads(cli_output.read_text())["passed"])

    def test_rejects_runtime_and_lineage_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted_log, resumed_log = _write_histories(root)
            uninterrupted = _checkpoint(
                root,
                "uninterrupted",
                torch.tensor([1.0]),
            )
            resumed = _checkpoint(root, "resumed", torch.tensor([1.0]))

            runtime_mismatch = _manifest(resumed=True)
            runtime_mismatch.binding.runtime.cuda_version = "12.8"
            with (
                patch(
                    "lm_from_zero.resume_tolerance.validate_checkpoint",
                    side_effect=(_manifest(resumed=False), runtime_mismatch),
                ),
                self.assertRaisesRegex(ResumeToleranceError, "runtimes"),
            ):
                build_dense_resume_tolerance_report(
                    uninterrupted_jsonl=uninterrupted_log,
                    uninterrupted_checkpoint=uninterrupted,
                    resumed_jsonl=resumed_log,
                    resumed_checkpoint=resumed,
                )

            lineage_mismatch = _manifest(resumed=True)
            lineage_mismatch.lineage.parent_checkpoint_id = "step-000000000000"
            with (
                patch(
                    "lm_from_zero.resume_tolerance.validate_checkpoint",
                    side_effect=(_manifest(resumed=False), lineage_mismatch),
                ),
                self.assertRaisesRegex(ResumeToleranceError, "lineage"),
            ):
                build_dense_resume_tolerance_report(
                    uninterrupted_jsonl=uninterrupted_log,
                    uninterrupted_checkpoint=uninterrupted,
                    resumed_jsonl=resumed_log,
                    resumed_checkpoint=resumed,
                )

    def test_rejects_dirty_checkpoint_and_invalid_run_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            uninterrupted_log, resumed_log = _write_histories(root)
            uninterrupted = _checkpoint(
                root,
                "uninterrupted",
                torch.tensor([1.0]),
            )
            resumed = _checkpoint(root, "resumed", torch.tensor([1.0]))
            with (
                patch(
                    "lm_from_zero.resume_tolerance.validate_checkpoint",
                    side_effect=(
                        _manifest(resumed=False, dirty=True),
                        _manifest(resumed=True),
                    ),
                ),
                self.assertRaisesRegex(ResumeToleranceError, "dirty"),
            ):
                build_dense_resume_tolerance_report(
                    uninterrupted_jsonl=uninterrupted_log,
                    uninterrupted_checkpoint=uninterrupted,
                    resumed_jsonl=resumed_log,
                    resumed_checkpoint=resumed,
                )

            resumed_log.write_text('{"event": "run_start"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ResumeToleranceError, "canonical"):
                build_dense_resume_tolerance_report(
                    uninterrupted_jsonl=uninterrupted_log,
                    uninterrupted_checkpoint=uninterrupted,
                    resumed_jsonl=resumed_log,
                    resumed_checkpoint=resumed,
                )


if __name__ == "__main__":
    unittest.main()
