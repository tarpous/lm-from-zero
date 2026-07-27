import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from typer.testing import CliRunner

from lm_from_zero.cli import app
from lm_from_zero.training.metrics import (
    ScalarWriter,
    TrainingMetricsError,
    TrainingMetricSinks,
    append_training_event,
    load_optimizer_metrics,
    materialize_metrics_parquet,
)


def _step_payload(step: int, *, loss: float = 2.0) -> dict[str, object]:
    return {
        "elapsed_seconds": 0.5,
        "event": "optimizer_step",
        "gradient_norm": 1.0,
        "learning_rate": 0.001,
        "loss": loss,
        "optimizer_step": step,
        "peak_cuda_memory_allocated_bytes": None,
        "peak_cuda_memory_reserved_bytes": None,
        "tokens_consumed": step * 8,
        "tokens_per_second": 16.0,
    }


class _FakeWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.flush_count = 0
        self.close_count = 0

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int,
    ) -> None:
        self.scalars.append((tag, scalar_value, global_step))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.close_count += 1


class TrainingMetricsTests(unittest.TestCase):
    def test_jsonl_is_canonical_and_parquet_keeps_latest_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "events.jsonl"
            recorded_at = datetime(2026, 7, 27, 1, 2, 3, tzinfo=UTC)
            append_training_event(
                jsonl,
                {"event": "run_start", "optimizer_step": 0},
                recorded_at_utc=recorded_at,
            )
            append_training_event(
                jsonl,
                _step_payload(1),
                recorded_at_utc=recorded_at,
            )
            append_training_event(
                jsonl,
                _step_payload(1, loss=1.5),
                recorded_at_utc=recorded_at,
            )
            append_training_event(
                jsonl,
                _step_payload(2, loss=1.0),
                recorded_at_utc=recorded_at,
            )

            lines = jsonl.read_text(encoding="utf-8").splitlines()
            self.assertTrue(
                all(
                    line
                    == json.dumps(
                        json.loads(line),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for line in lines
                )
            )
            records = load_optimizer_metrics(jsonl)
            self.assertEqual([record.optimizer_step for record in records], [1, 2])
            self.assertEqual(records[0].loss, 1.5)

            parquet = materialize_metrics_parquet(jsonl, root / "metrics.parquet")
            frame = pl.read_parquet(parquet)
            self.assertEqual(frame["optimizer_step"].to_list(), [1, 2])
            self.assertEqual(frame["loss"].to_list(), [1.5, 1.0])
            self.assertIn("recorded_at_utc", frame.columns)

    def test_rejects_noncanonical_corrupt_empty_and_partial_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "events.jsonl"
            jsonl.write_text('{"event": "run_start"}\n', encoding="utf-8")
            with self.assertRaisesRegex(TrainingMetricsError, "canonical"):
                load_optimizer_metrics(jsonl)

            jsonl.write_text("{broken\n", encoding="utf-8")
            with self.assertRaisesRegex(TrainingMetricsError, "invalid"):
                load_optimizer_metrics(jsonl)

            jsonl.write_text("", encoding="utf-8")
            append_training_event(jsonl, {"event": "run_start"})
            with self.assertRaisesRegex(TrainingMetricsError, "no optimizer"):
                load_optimizer_metrics(jsonl)

            jsonl.write_text("", encoding="utf-8")
            append_training_event(jsonl, _step_payload(1))
            parquet = root / "metrics.parquet"
            partial = parquet.with_name(f".{parquet.name}.tmp")
            partial.write_text("partial", encoding="utf-8")
            with self.assertRaisesRegex(TrainingMetricsError, "incomplete"):
                materialize_metrics_parquet(jsonl, parquet)

    def test_sinks_mirror_scalars_snapshot_and_resume_purge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = _FakeWriter()
            factory_calls: list[tuple[Path, int | None]] = []

            def factory(directory: Path, purge_step: int | None) -> ScalarWriter:
                factory_calls.append((directory, purge_step))
                return writer

            sinks = TrainingMetricSinks(
                jsonl_path=root / "events.jsonl",
                tensorboard_directory=root / "tensorboard",
                parquet_path=root / "metrics.parquet",
                resume_optimizer_step=4,
                writer_factory=factory,
            )
            sinks.append_event({"event": "run_resume", "optimizer_step": 4})
            sinks.log_optimizer_step(
                {
                    **_step_payload(5),
                    "peak_cuda_memory_allocated_bytes": 100,
                    "peak_cuda_memory_reserved_bytes": 200,
                }
            )
            sinks.snapshot()
            sinks.close()
            sinks.close()

            self.assertEqual(factory_calls, [(root / "tensorboard", 5)])
            self.assertEqual(
                {tag for tag, _, _ in writer.scalars},
                {
                    "train/elapsed_seconds",
                    "train/gradient_norm",
                    "train/learning_rate",
                    "train/loss",
                    "train/peak_cuda_memory_allocated_bytes",
                    "train/peak_cuda_memory_reserved_bytes",
                    "train/tokens_consumed",
                    "train/tokens_per_second",
                },
            )
            self.assertTrue(all(step == 5 for _, _, step in writer.scalars))
            self.assertEqual(writer.flush_count, 2)
            self.assertEqual(writer.close_count, 1)
            self.assertTrue((root / "metrics.parquet").is_file())
            with self.assertRaisesRegex(TrainingMetricsError, "closed"):
                sinks.append_event({"event": "run_complete"})

    def test_sinks_can_run_without_optional_mirrors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sinks = TrainingMetricSinks(
                jsonl_path=root / "events.jsonl",
                tensorboard_directory=None,
                parquet_path=None,
                resume_optimizer_step=0,
            )
            sinks.log_optimizer_step(_step_payload(1))
            sinks.snapshot()
            sinks.close()
            self.assertEqual(len(load_optimizer_metrics(root / "events.jsonl")), 1)

            output = root / "rebuilt.parquet"
            result = CliRunner().invoke(
                app,
                [
                    "materialize-training-metrics",
                    str(root / "events.jsonl"),
                    "--output",
                    str(output),
                ],
            )
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(json.loads(result.stdout)["optimizer_steps"], 1)
            self.assertTrue(output.is_file())

    def test_abort_closes_live_writer_without_materializing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = _FakeWriter()
            sinks = TrainingMetricSinks(
                jsonl_path=root / "events.jsonl",
                tensorboard_directory=root / "tensorboard",
                parquet_path=root / "metrics.parquet",
                resume_optimizer_step=0,
                writer_factory=lambda _directory, _purge_step: writer,
            )
            sinks.log_optimizer_step(_step_payload(1))
            sinks.abort()
            sinks.abort()
            self.assertEqual(writer.flush_count, 1)
            self.assertEqual(writer.close_count, 1)
            self.assertFalse((root / "metrics.parquet").exists())


if __name__ == "__main__":
    unittest.main()
