from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError
from typer.testing import CliRunner

from lm_from_zero.architecture_study import (
    ArchitectureStudyPlan,
    create_architecture_study_plan,
)
from lm_from_zero.cli import app
from lm_from_zero.dense_ablations import (
    DenseAblationError,
    DenseAblationPlan,
    create_dense_ablation_plan,
    write_dense_ablation_plan,
)

SHARD_HASH = "a" * 64
TOKENIZER_HASH = "b" * 64


def _build() -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer_hash=TOKENIZER_HASH,
        tokenizer_vocab_size=16_000,
    )


def _architecture_plan() -> ArchitectureStudyPlan:
    with (
        patch(
            "lm_from_zero.architecture_study.validate_shard_build",
            return_value=_build(),
        ),
        patch(
            "lm_from_zero.architecture_study._file_sha256",
            return_value=SHARD_HASH,
        ),
    ):
        return create_architecture_study_plan("build.json")


class DenseAblationPlanTests(unittest.TestCase):
    def test_freezes_eight_variants_and_reuses_only_m7_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "architecture-study.json"
            source.write_text(
                _architecture_plan().canonical_json() + "\n", encoding="utf-8"
            )
            with patch(
                "lm_from_zero.dense_ablations._sha256_file",
                return_value="c" * 64,
            ):
                plan = create_dense_ablation_plan(source)

        self.assertEqual(len(plan.variants), 8)
        self.assertEqual(len(plan.jobs), 24)
        self.assertFalse(plan.execution_ready)
        self.assertEqual(
            {variant.execution_status for variant in plan.variants},
            {"reuse_m7_screening", "ready_for_bounded_gpu_smoke"},
        )
        baseline = [job for job in plan.jobs if job.variant == "baseline"]
        self.assertEqual(len(baseline), 3)
        self.assertTrue(
            all(
                job.execution_status
                in {"reuse_m7_screening", "requires_m7_checkpoint_recovery"}
                for job in baseline
            )
        )
        self.assertTrue(
            all(
                job.reused_m7_checkpoint
                if job.execution_status == "reuse_m7_screening"
                else job.reused_m7_checkpoint is None
                for job in baseline
            )
        )
        pending = [job for job in plan.jobs if job.variant != "baseline"]
        self.assertEqual(len(pending), 21)
        self.assertTrue(
            all(
                job.execution_status == "ready_for_bounded_gpu_smoke"
                and job.model_config_sha256 is None
                for job in pending
            )
        )
        self.assertEqual(
            plan.canonical_json(),
            json.dumps(
                plan.model_dump(mode="json"),
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    def test_rejects_non_100m_source_screen(self) -> None:
        source_plan = _architecture_plan().model_dump(mode="json")
        source_plan["screening_dense_reference_tokens"] = 99
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "architecture-study.json"
            source.write_text(json.dumps(source_plan), encoding="utf-8")
            with self.assertRaisesRegex(DenseAblationError, "100M-token"):
                create_dense_ablation_plan(source)

    def test_cli_writes_atomic_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "architecture-study.json"
            source.write_text(
                _architecture_plan().canonical_json() + "\n", encoding="utf-8"
            )
            output = root / "dense-ablations.json"
            with patch(
                "lm_from_zero.dense_ablations._sha256_file",
                return_value="c" * 64,
            ):
                result = CliRunner().invoke(
                    app,
                    [
                        "plan-dense-ablations",
                        str(source),
                        "--output",
                        str(output),
                        "--artifact-root",
                        str(root / "artifacts"),
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            plan = DenseAblationPlan.model_validate_json(result.stdout)
            self.assertEqual(
                plan.jobs[0].artifact_directory,
                (root / "artifacts" / "baseline" / "seed-1337").as_posix(),
            )
            self.assertEqual(output.read_text(encoding="utf-8"), result.stdout)
            temporary = output.with_name(f".{output.name}.tmp")
            temporary.write_text("incomplete", encoding="utf-8")
            with self.assertRaisesRegex(DenseAblationError, "incomplete"):
                write_dense_ablation_plan(output, plan)

    def test_rejects_duplicate_or_missing_variant_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "architecture-study.json"
            source.write_text(
                _architecture_plan().canonical_json() + "\n", encoding="utf-8"
            )
            with patch(
                "lm_from_zero.dense_ablations._sha256_file",
                return_value="c" * 64,
            ):
                plan = create_dense_ablation_plan(source)
        payload = plan.model_dump(mode="json")
        payload["jobs"] = payload["jobs"][:-1]
        with self.assertRaisesRegex(ValidationError, "exactly eight variants"):
            DenseAblationPlan.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
