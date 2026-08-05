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
    STUDY_SEEDS,
    ArchitectureStudyError,
    ArchitectureStudyPlan,
    create_architecture_study_plan,
    write_architecture_study_plan,
)
from lm_from_zero.cli import app

SHARD_HASH = "a" * 64
TOKENIZER_HASH = "b" * 64


def _build() -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer_hash=TOKENIZER_HASH,
        tokenizer_vocab_size=16_000,
    )


class ArchitectureStudyPlanTests(unittest.TestCase):
    def _plan(self) -> ArchitectureStudyPlan:
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

    def test_exact_matrix_budgets_and_full_scheduler_semantics(self) -> None:
        plan = self._plan()
        self.assertEqual(plan.format_version, 1)
        self.assertEqual(plan.diffusion_adamw_backend, "auto")
        self.assertEqual(len(plan.lineages), 9)
        self.assertEqual(
            tuple((item.architecture, item.seed) for item in plan.lineages),
            tuple(
                (architecture, seed)
                for architecture in ("dense", "mamba2", "diffusion")
                for seed in STUDY_SEEDS
            ),
        )
        expected = {
            "dense": (12_208, 100_007_936, 61_036, 500_006_912),
            "mamba2": (14_784, 121_110_528, 73_919, 605_544_448),
            "diffusion": (12_915, 105_799_680, 64_574, 528_990_208),
        }
        for lineage in plan.lineages:
            screen_step, screen_tokens, full_step, full_tokens = expected[
                lineage.architecture
            ]
            self.assertEqual(lineage.screening.optimizer_step, screen_step)
            self.assertEqual(lineage.screening.tokens_consumed, screen_tokens)
            self.assertEqual(lineage.full.optimizer_step, full_step)
            self.assertEqual(lineage.full.tokens_consumed, full_tokens)
            self.assertEqual(lineage.scheduler_total_steps, full_step)
            self.assertEqual(lineage.continues_to_full, lineage.seed == 1_337)
            self.assertLess(abs(lineage.screening.training_flop_ratio - 1), 0.03)
            self.assertLess(abs(lineage.full.training_flop_ratio - 1), 0.03)
            self.assertEqual(
                lineage.estimated_retained_checkpoint_bytes_upper_bound,
                lineage.estimated_checkpoint_bytes * 4,
            )
            self.assertEqual(
                lineage.parquet_log,
                str(Path(lineage.jsonl_log).with_suffix(".parquet")).replace("\\", "/"),
            )
            self.assertEqual(lineage.adamw_backend, "auto")
        self.assertEqual(
            len({item.checkpoint_directory for item in plan.lineages}),
            9,
        )
        self.assertEqual(
            len({item.training_config_sha256 for item in plan.lineages}),
            9,
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

    def test_atomic_write_cli_and_optional_runtime_estimates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_path = root / "build.json"
            build_path.write_text("{}", encoding="utf-8")
            output = root / "plans" / "architecture-study.json"
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
                result = CliRunner().invoke(
                    app,
                    [
                        "plan-architecture-study",
                        str(build_path),
                        "--output",
                        str(output),
                        "--dense-tokens-per-second",
                        "1000",
                        "--mamba2-tokens-per-second",
                        "500",
                        "--diffusion-tokens-per-second",
                        "2000",
                    ],
                )
            self.assertEqual(result.exit_code, 0, result.output)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["format"], "lm-from-zero-architecture-study-plan")
            self.assertEqual(output.read_text(encoding="utf-8"), result.stdout)
            plan = ArchitectureStudyPlan.model_validate(payload)
            dense = plan.lineages[0]
            self.assertEqual(
                dense.screening.estimated_seconds,
                dense.screening.tokens_consumed / 1_000,
            )

            self.assertEqual(
                plan.lineages[-1].adamw_backend,
                "auto",
            )

            temporary = output.with_name(f".{output.name}.tmp")
            temporary.write_text("incomplete", encoding="utf-8")
            with self.assertRaisesRegex(ArchitectureStudyError, "incomplete"):
                write_architecture_study_plan(output, plan)

    def test_rejects_invalid_budgets_throughput_and_matrix(self) -> None:
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
            with self.assertRaisesRegex(ArchitectureStudyError, "precede"):
                create_architecture_study_plan(
                    "build.json",
                    screening_dense_reference_tokens=500,
                    full_dense_reference_tokens=100,
                )
            with self.assertRaisesRegex(ArchitectureStudyError, "throughput"):
                create_architecture_study_plan(
                    "build.json",
                    dense_tokens_per_second=0,
                )

            fused = create_architecture_study_plan(
                "build.json",
                diffusion_adamw_backend="fused",
            )
            self.assertEqual(fused.diffusion_adamw_backend, "fused")
            self.assertTrue(
                all(
                    lineage.adamw_backend == "fused"
                    for lineage in fused.lineages
                    if lineage.architecture == "diffusion"
                )
            )

        plan = self._plan()
        payload = plan.model_dump(mode="json")
        payload["lineages"] = payload["lineages"][:-1]
        with self.assertRaisesRegex(ValidationError, "three architectures"):
            ArchitectureStudyPlan.model_validate(payload)

        payload = plan.model_dump(mode="json")
        payload["lineages"][0]["scheduler_total_steps"] += 1
        with self.assertRaisesRegex(ValidationError, "full-budget scheduler"):
            ArchitectureStudyPlan.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
