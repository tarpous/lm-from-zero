import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from lm_from_zero.diffusion_evaluation import DiffusionEvaluationResult
from lm_from_zero.evaluation import CausalEvaluationResult
from lm_from_zero.generation import (
    CausalGenerationRecord,
    CausalGenerationResult,
    DiffusionGenerationRecord,
    DiffusionGenerationResult,
)
from lm_from_zero.smoke_report import (
    DenseSmokeReport,
    DiffusionSmokeReport,
    SmokeReportError,
    build_dense_smoke_report,
    build_diffusion_smoke_report,
    build_mamba2_smoke_report,
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
CHECKPOINT_ID = "step-000000000004"
CHECKPOINT_HASH = sha256(b"canonical checkpoint").hexdigest()


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


def _write_diffusion_evaluation(path: Path) -> None:
    result = DiffusionEvaluationResult(
        evaluated_at_utc=RECORDED_AT + timedelta(minutes=1),
        source_checkpoint_id=CHECKPOINT_ID,
        source_checkpoint_manifest_sha256=CHECKPOINT_HASH,
        device="cuda",
        split="validation",
        model_config_sha256=MODEL_HASH,
        shard_manifest_sha256=SHARD_HASH,
        tokenizer_sha256=TOKENIZER_HASH,
        seed=1_337,
        batch_count=1,
        source_sequence_count=2,
        corruption_samples_per_batch=1,
        evaluated_example_count=2,
        model_forwards=1,
        eligible_token_count=2_046,
        masked_token_count=1_024,
        mean_mask_rate=1_024 / 2_046,
        masked_reconstruction_loss_nats=9.5,
        variational_upper_bound_nats=10.25,
        elapsed_seconds=0.5,
        masked_tokens_per_second=2_048.0,
        cursor_before=_cursor(0),
        cursor_after=_cursor(2_048),
    )
    path.write_text(f"{result.canonical_json()}\n", encoding="utf-8")


def _write_diffusion_generation(path: Path) -> None:
    record = DiffusionGenerationRecord(
        generated_at_utc=RECORDED_AT + timedelta(minutes=2),
        source_checkpoint_id=CHECKPOINT_ID,
        source_checkpoint_manifest_sha256=CHECKPOINT_HASH,
        device="cuda",
        model_config_sha256=MODEL_HASH,
        tokenizer_sha256=TOKENIZER_HASH,
        prompt_token_sha256=PROMPT_HASH,
        result=DiffusionGenerationResult(
            prompt_token_counts=(4,),
            response_canvas_length=2,
            generated_token_ids=((8, 9),),
            stop_reasons=("canvas_complete",),
            diffusion_steps=2,
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
    checkpoint.binding.architecture = "olmo2"
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


def _mamba2_export(checkpoint: MagicMock) -> SimpleNamespace:
    exported = _export(checkpoint)
    exported.cached_fp32_max_abs_error = 0.0
    exported.requires_trust_remote_code = True
    return exported


def _diffusion_export(checkpoint: MagicMock) -> SimpleNamespace:
    exported = _export(checkpoint)
    exported.fp32_loss_abs_error = 0.0
    exported.deterministic_trajectory_matches = True
    exported.requires_trust_remote_code = True
    return exported


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

    def test_builds_mamba2_report_with_export_compatibility_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._paths(root)
            checkpoint = _checkpoint()
            checkpoint.binding.architecture = "mamba2"
            with (
                patch(
                    "lm_from_zero.smoke_report.validate_checkpoint",
                    return_value=checkpoint,
                ),
                patch(
                    "lm_from_zero.smoke_report.load_mamba2_export_manifest",
                    return_value=_mamba2_export(checkpoint),
                ),
            ):
                report = build_mamba2_smoke_report(
                    training_jsonl=paths[0],
                    checkpoint_directory=paths[1],
                    evaluation_jsonl=paths[2],
                    export_directory=paths[3],
                    generation_jsonl=paths[4],
                )

            self.assertEqual(report.format, "lm-from-zero-mamba2-smoke-report")
            self.assertEqual(report.architecture, "mamba2")
            self.assertEqual(report.export_cached_fp32_max_abs_error, 0.0)
            self.assertTrue(report.export_requires_trust_remote_code)

    def test_builds_diffusion_report_without_causal_perplexity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._paths(root)
            _write_diffusion_evaluation(paths[2])
            _write_diffusion_generation(paths[4])
            checkpoint = _checkpoint()
            checkpoint.binding.architecture = "masked_diffusion"
            with (
                patch(
                    "lm_from_zero.smoke_report.validate_checkpoint",
                    return_value=checkpoint,
                ),
                patch(
                    "lm_from_zero.smoke_report.load_diffusion_export_manifest",
                    return_value=_diffusion_export(checkpoint),
                ),
            ):
                report = build_diffusion_smoke_report(
                    training_jsonl=paths[0],
                    checkpoint_directory=paths[1],
                    evaluation_jsonl=paths[2],
                    export_directory=paths[3],
                    generation_jsonl=paths[4],
                )

            self.assertIsInstance(report, DiffusionSmokeReport)
            self.assertEqual(report.format, "lm-from-zero-diffusion-smoke-report")
            self.assertEqual(report.architecture, "masked_diffusion")
            self.assertFalse(report.causal_perplexity_applicable)
            self.assertNotIn("validation_perplexity", report.model_dump())
            self.assertEqual(report.validation_model_forwards, 1)
            self.assertEqual(report.generation_diffusion_steps, 2)
            self.assertTrue(report.export_deterministic_trajectory_matches)
            self.assertEqual(report.format_version, 2)
            self.assertEqual(report.evaluation_checkpoint_id, CHECKPOINT_ID)
            self.assertEqual(report.generation_checkpoint_id, CHECKPOINT_ID)
            self.assertEqual(report.generation_device, "cuda")

    def test_rejects_unbound_diffusion_evaluation_and_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = self._paths(root)
            checkpoint = _checkpoint()
            checkpoint.binding.architecture = "masked_diffusion"

            def build() -> DiffusionSmokeReport:
                with (
                    patch(
                        "lm_from_zero.smoke_report.validate_checkpoint",
                        return_value=checkpoint,
                    ),
                    patch(
                        "lm_from_zero.smoke_report.load_diffusion_export_manifest",
                        return_value=_diffusion_export(checkpoint),
                    ),
                ):
                    return build_diffusion_smoke_report(
                        training_jsonl=paths[0],
                        checkpoint_directory=paths[1],
                        evaluation_jsonl=paths[2],
                        export_directory=paths[3],
                        generation_jsonl=paths[4],
                    )

            evaluation_changes: tuple[tuple[str, object, str], ...] = (
                ("source_checkpoint_id", "step-000000000003", "checkpoint ID"),
                (
                    "source_checkpoint_manifest_sha256",
                    "9" * 64,
                    "manifest hash",
                ),
                ("device", "cpu", "did not use CUDA"),
            )
            for field, value, error in evaluation_changes:
                with self.subTest(artifact="evaluation", field=field):
                    _write_diffusion_evaluation(paths[2])
                    _write_diffusion_generation(paths[4])
                    payload = json.loads(paths[2].read_text())
                    payload[field] = value
                    _write_jsonl(paths[2], [payload])
                    with self.assertRaisesRegex(SmokeReportError, error):
                        build()

            generation_changes: tuple[tuple[str, object, str], ...] = (
                ("source_checkpoint_id", "step-000000000003", "checkpoint ID"),
                (
                    "source_checkpoint_manifest_sha256",
                    "9" * 64,
                    "manifest hash",
                ),
                ("device", "cpu", "did not use CUDA"),
            )
            for field, value, error in generation_changes:
                with self.subTest(artifact="generation", field=field):
                    _write_diffusion_evaluation(paths[2])
                    _write_diffusion_generation(paths[4])
                    payload = json.loads(paths[4].read_text())
                    payload[field] = value
                    _write_jsonl(paths[4], [payload])
                    with self.assertRaisesRegex(SmokeReportError, error):
                        build()

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

            checkpoint = _checkpoint()
            checkpoint.binding.architecture = "mamba2"
            with self.assertRaisesRegex(SmokeReportError, "architecture"):
                self._build(paths, checkpoint, _export(checkpoint))


if __name__ == "__main__":
    unittest.main()
