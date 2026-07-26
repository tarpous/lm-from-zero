import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lm_from_zero.evaluation import CausalEvaluationResult
from lm_from_zero.generation import (
    CausalGenerationRecord,
    CausalGenerationResult,
)
from lm_from_zero.smoke_report import (
    DenseSmokeReport,
    SmokeReportError,
    build_dense_smoke_report,
    write_dense_smoke_report,
)
from lm_from_zero.training import BatchCursor

TRAINING_HASH = "1" * 64
MODEL_HASH = "2" * 64
SHARD_HASH = "3" * 64
TOKENIZER_HASH = "4" * 64
PROMPT_HASH = "5" * 64
ARTIFACT_HASH = "6" * 64
GIT_REVISION = "7" * 40
RECORDED_AT = datetime(2026, 7, 26, 18, 0, tzinfo=UTC)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            f"{json.dumps(record, sort_keys=True, separators=(',', ':'))}\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _training_events() -> list[dict[str, object]]:
    config = {"compile_model": True, "device": "cuda", "precision": "bf16"}
    records: list[dict[str, object]] = [
        {
            "event": "run_start",
            "training_config": config,
            "training_config_sha256": TRAINING_HASH,
        }
    ]
    for step in range(1, 3):
        records.append(_training_step(step))
    records.append(
        {
            "event": "run_resume",
            "training_config_sha256": TRAINING_HASH,
        }
    )
    for step in range(3, 5):
        records.append(_training_step(step))
    return records


def _training_step(step: int) -> dict[str, object]:
    return {
        "elapsed_seconds": 0.25,
        "event": "optimizer_step",
        "gradient_norm": 1.0,
        "learning_rate": 0.001,
        "loss": 9.0,
        "optimizer_step": step,
        "peak_cuda_memory_allocated_bytes": 1_000,
        "peak_cuda_memory_reserved_bytes": 2_000,
        "recorded_at_utc": (RECORDED_AT + timedelta(seconds=step)).isoformat(),
        "tokens_consumed": step * 8_192,
        "tokens_per_second": 32_768.0,
    }


def _cursor(tokens: int) -> BatchCursor:
    return BatchCursor(
        build_manifest_sha256=SHARD_HASH,
        tokenizer_hash=TOKENIZER_HASH,
        split="validation",
        sequence_length=1_024,
        seed=1_337,
        rank=0,
        world_size=1,
        shuffle=False,
        next_local_window=tokens // 1_024,
        sequences_consumed=tokens // 1_024,
        tokens_consumed=tokens,
    )


def _write_evaluation(path: Path, *, model_hash: str = MODEL_HASH) -> None:
    result = CausalEvaluationResult(
        evaluated_at_utc=RECORDED_AT + timedelta(minutes=1),
        split="validation",
        model_config_sha256=model_hash,
        shard_manifest_sha256=SHARD_HASH,
        tokenizer_sha256=TOKENIZER_HASH,
        batch_count=1,
        sequence_count=2,
        predicted_token_count=2_046,
        mean_loss=9.5,
        perplexity=13_359.7,
        elapsed_seconds=0.5,
        predicted_tokens_per_second=4_092.0,
        cursor_before=_cursor(0),
        cursor_after=_cursor(2_048),
    )
    path.write_text(f"{result.canonical_json()}\n", encoding="utf-8")


def _write_generation(path: Path) -> None:
    record = CausalGenerationRecord(
        generated_at_utc=RECORDED_AT + timedelta(minutes=2),
        model_config_sha256=MODEL_HASH,
        tokenizer_sha256=TOKENIZER_HASH,
        prompt_token_sha256=PROMPT_HASH,
        result=CausalGenerationResult(
            prompt_token_counts=(4,),
            generated_token_ids=((8, 9),),
            stop_reasons=("max_new_tokens",),
            model_forwards=2,
            generated_token_count=2,
            elapsed_seconds=0.1,
            tokens_per_second=20.0,
        ),
    )
    path.write_text(f"{record.canonical_json()}\n", encoding="utf-8")


def _checkpoint() -> MagicMock:
    checkpoint = MagicMock()
    checkpoint.created_at_utc = RECORDED_AT
    checkpoint.canonical_bytes.return_value = b"canonical checkpoint"
    checkpoint.binding.git.dirty = False
    checkpoint.binding.git.revision = GIT_REVISION
    checkpoint.binding.model_config_sha256 = MODEL_HASH
    checkpoint.binding.shard_manifest_sha256 = SHARD_HASH
    checkpoint.binding.tokenizer_sha256 = TOKENIZER_HASH
    checkpoint.binding.runtime.cuda_available = True
    checkpoint.binding.runtime.cuda_version = "13.0"
    checkpoint.binding.runtime.cuda_device_names = ("Test GPU",)
    checkpoint.binding.runtime.torch_version = "2.13.0"
    checkpoint.progress.optimizer_step = 4
    checkpoint.progress.tokens_consumed = 32_768
    checkpoint.lineage.checkpoint_id = "step-000000000004"
    checkpoint.lineage.parent_checkpoint_id = "step-000000000002"
    return checkpoint


def _export(checkpoint: MagicMock) -> SimpleNamespace:
    from hashlib import sha256

    return SimpleNamespace(
        source_checkpoint_id="step-000000000004",
        source_checkpoint_manifest_sha256=sha256(
            checkpoint.canonical_bytes()
        ).hexdigest(),
        model_config_sha256=MODEL_HASH,
        tokenizer_sha256=TOKENIZER_HASH,
        transformers_version="5.14.1",
        fp32_max_abs_error=0.0,
        artifacts=(
            SimpleNamespace(filename="model.safetensors", sha256=ARTIFACT_HASH),
        ),
    )


class DenseSmokeReportTests(unittest.TestCase):
    def _paths(
        self,
        root: Path,
        *,
        model_hash: str = MODEL_HASH,
    ) -> tuple[Path, Path, Path, Path, Path]:
        training = root / "training.jsonl"
        evaluation = root / "evaluation.jsonl"
        generation = root / "generation.jsonl"
        checkpoint_directory = root / "checkpoint"
        export_directory = root / "export"
        checkpoint_directory.mkdir()
        export_directory.mkdir()
        _write_jsonl(training, _training_events())
        _write_evaluation(evaluation, model_hash=model_hash)
        _write_generation(generation)
        return (
            training,
            checkpoint_directory,
            evaluation,
            export_directory,
            generation,
        )

    def _build(
        self,
        paths: tuple[Path, Path, Path, Path, Path],
        checkpoint: MagicMock,
        exported: SimpleNamespace,
    ) -> DenseSmokeReport:
        with (
            patch(
                "lm_from_zero.smoke_report.validate_checkpoint",
                return_value=checkpoint,
            ),
            patch(
                "lm_from_zero.smoke_report.load_export_manifest",
                return_value=exported,
            ),
        ):
            return build_dense_smoke_report(
                training_jsonl=paths[0],
                checkpoint_directory=paths[1],
                evaluation_jsonl=paths[2],
                export_directory=paths[3],
                generation_jsonl=paths[4],
            )

    def test_builds_and_atomically_writes_canonical_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._paths(root)
            checkpoint = _checkpoint()
            report = self._build(paths, checkpoint, _export(checkpoint))

            self.assertEqual(report.checkpoint_id, "step-000000000004")
            self.assertEqual(report.parent_checkpoint_id, "step-000000000002")
            self.assertEqual(len(report.training_steps), 4)
            self.assertEqual(report.final_tokens_consumed, 32_768)
            self.assertEqual(report.generation_token_count, 2)

            output = root / "reports" / "smoke.json"
            write_dense_smoke_report(output, report)
            self.assertEqual(output.read_bytes(), report.canonical_bytes() + b"\n")

            temporary = output.with_name(f".{output.name}.tmp")
            temporary.write_text("incomplete", encoding="utf-8")
            with self.assertRaisesRegex(SmokeReportError, "incomplete"):
                write_dense_smoke_report(output, report)

    def test_rejects_noncanonical_and_incomplete_training_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._paths(root)
            checkpoint = _checkpoint()
            exported = _export(checkpoint)

            paths[0].write_text('{"event": "run_start"}\n', encoding="utf-8")
            with self.assertRaisesRegex(SmokeReportError, "not canonical"):
                self._build(paths, checkpoint, exported)

            _write_jsonl(paths[0], [_training_events()[0]])
            with self.assertRaisesRegex(SmokeReportError, "one start and one resume"):
                self._build(paths, checkpoint, exported)

    def test_rejects_training_configuration_and_step_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._paths(root)
            checkpoint = _checkpoint()
            exported = _export(checkpoint)
            events = _training_events()

            events[3]["training_config_sha256"] = "9" * 64
            _write_jsonl(paths[0], events)
            with self.assertRaisesRegex(SmokeReportError, "changed across resume"):
                self._build(paths, checkpoint, exported)

            events = _training_events()
            events[0]["training_config"] = {"device": "cpu"}
            _write_jsonl(paths[0], events)
            with self.assertRaisesRegex(SmokeReportError, "compiled bf16 CUDA"):
                self._build(paths, checkpoint, exported)

            events = _training_events()
            events[2]["optimizer_step"] = 4
            _write_jsonl(paths[0], events)
            with self.assertRaisesRegex(SmokeReportError, "not contiguous"):
                self._build(paths, checkpoint, exported)

    def test_rejects_checkpoint_and_cross_artifact_disagreement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._paths(root, model_hash="8" * 64)
            checkpoint = _checkpoint()
            exported = _export(checkpoint)
            with self.assertRaisesRegex(SmokeReportError, "model configuration"):
                self._build(paths, checkpoint, exported)

            _write_evaluation(paths[2])
            checkpoint.binding.git.dirty = True
            with self.assertRaisesRegex(SmokeReportError, "dirty worktree"):
                self._build(paths, checkpoint, exported)

            checkpoint.binding.git.dirty = False
            exported.source_checkpoint_manifest_sha256 = "9" * 64
            with self.assertRaisesRegex(SmokeReportError, "manifest hash"):
                self._build(paths, checkpoint, exported)

            exported = _export(checkpoint)
            checkpoint.binding.runtime.cuda_available = False
            checkpoint.binding.runtime.cuda_version = None
            with self.assertRaisesRegex(SmokeReportError, "CUDA runtime"):
                self._build(paths, checkpoint, exported)


if __name__ == "__main__":
    unittest.main()
