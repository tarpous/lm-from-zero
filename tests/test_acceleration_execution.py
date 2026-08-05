from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from lm_from_zero.acceleration_calibration import create_plan, write_plan
from lm_from_zero.acceleration_execution import (
    AccelerationExecutionError,
    CalibrationCellDryRun,
    CudaDeviceMetadata,
    capture_cuda_device_metadata,
    execute_calibration_cell,
    inspect_repository_state,
    require_executable_calibration_cell,
    resolve_from_plan,
    validate_calibration_execution_preflight,
)
from lm_from_zero.training.checkpointing import GitMetadata

SHARD_HASH = "a" * 64
TOKENIZER_HASH = "b" * 64
GIT_REVISION = "c" * 40
DEVICE_NAME = "NVIDIA GeForce RTX 4080 SUPER"


def _build() -> SimpleNamespace:
    return SimpleNamespace(
        tokenizer_hash=TOKENIZER_HASH,
        tokenizer_vocab_size=16_000,
    )


def _cuda(*, available: bool = True, name: str = DEVICE_NAME) -> CudaDeviceMetadata:
    if not available:
        return CudaDeviceMetadata(available=False, device_count=0)
    return CudaDeviceMetadata(
        available=True,
        device_count=1,
        selected_device_index=0,
        device_name=name,
        compute_capability=(8, 9),
        bf16_supported=True,
    )


class AccelerationExecutionTests(unittest.TestCase):
    def test_cuda_metadata_probe_covers_unavailable_invalid_and_valid_devices(
        self,
    ) -> None:
        with patch("torch.cuda.is_available", return_value=False):
            unavailable = capture_cuda_device_metadata()
        self.assertFalse(unavailable.available)

        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
        ):
            invalid = capture_cuda_device_metadata(1)
        self.assertIsNone(invalid.selected_device_index)

        properties = SimpleNamespace(name=DEVICE_NAME, major=8, minor=9)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.device_count", return_value=1),
            patch("torch.cuda.get_device_properties", return_value=properties),
            patch("torch.cuda.is_bf16_supported", return_value=True),
        ):
            valid = capture_cuda_device_metadata()
        self.assertEqual(valid.compute_capability, (8, 9))
        self.assertTrue(valid.bf16_supported)

    def _plan_and_dry_run(
        self, root: Path
    ) -> tuple[Path, Path, Path, CalibrationCellDryRun]:
        build_path = root / "build.json"
        build_path.write_text("{}", encoding="utf-8")
        tokenizer_path = root / "tokenizer.json"
        tokenizer_path.write_text("{}", encoding="utf-8")
        build = _build()
        with (
            patch(
                "lm_from_zero.acceleration_calibration.validate_shard_build",
                return_value=build,
            ),
            patch(
                "lm_from_zero.acceleration_calibration._file_sha256",
                return_value=SHARD_HASH,
            ),
        ):
            plan = create_plan(
                build_path,
                artifact_root=root / "results",
                repository_revision=GIT_REVISION,
            )
        plan_path = root / "plan.json"
        write_plan(plan_path, plan)
        with (
            patch(
                "lm_from_zero.acceleration_execution.validate_shard_build",
                return_value=build,
            ),
            patch(
                "lm_from_zero.acceleration_execution._file_sha256",
                return_value=SHARD_HASH,
            ),
        ):
            dry_run = resolve_from_plan(plan_path, build_path, "dense", "baseline", 1)
        return plan_path, build_path, tokenizer_path, dry_run

    def test_inspects_repository_state_for_plan_binding(self) -> None:
        expected = GitMetadata(revision=GIT_REVISION, dirty=False)
        with patch(
            "lm_from_zero.acceleration_execution.capture_git_metadata",
            return_value=expected,
        ) as capture:
            observed = inspect_repository_state(".")
        self.assertEqual(observed, expected)
        capture.assert_called_once_with(".")

    def test_resolves_all_architectures_without_model_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_path = root / "build.json"
            build_path.write_text("{}", encoding="utf-8")
            build = SimpleNamespace(
                tokenizer_hash=TOKENIZER_HASH,
                tokenizer_vocab_size=16_000,
            )
            with (
                patch(
                    "lm_from_zero.acceleration_calibration.validate_shard_build",
                    return_value=build,
                ),
                patch(
                    "lm_from_zero.acceleration_calibration._file_sha256",
                    return_value=SHARD_HASH,
                ),
            ):
                plan = create_plan(
                    build_path,
                    artifact_root=root / "results",
                    repository_revision=GIT_REVISION,
                )
            plan_path = root / "plan.json"
            write_plan(plan_path, plan)

            with (
                patch(
                    "lm_from_zero.acceleration_execution.validate_shard_build",
                    return_value=build,
                ),
                patch(
                    "lm_from_zero.acceleration_execution._file_sha256",
                    return_value=SHARD_HASH,
                ),
                patch("torch.nn.Module.to") as allocate,
            ):
                dense = resolve_from_plan(plan_path, build_path, "dense", "baseline", 1)
                mamba = resolve_from_plan(
                    plan_path, build_path, "mamba2", "tf32-high", 2
                )
                diffusion = resolve_from_plan(
                    plan_path, build_path, "diffusion", "flash-sdpa", 3
                )

            allocate.assert_not_called()
            self.assertTrue(dense.training_config["compile_model"])
            self.assertEqual(dense.training_config["compile_mode"], "default")
            self.assertEqual(mamba.training_config["float32_matmul_precision"], "high")
            self.assertEqual(diffusion.training_config["sdpa_backend"], "flash")
            self.assertTrue(
                diffusion.training_config["diffusion_padding_free_attention"]
            )
            self.assertEqual(diffusion.total_optimizer_steps, 150)
            self.assertTrue(diffusion.result_path.endswith("repetition-03/result.json"))
            require_executable_calibration_cell(diffusion)
            self.assertTrue(diffusion.execution_ready)
            self.assertEqual(diffusion.execution_blockers, ())

    def test_rejects_repetition_and_manifest_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_path = root / "build.json"
            build_path.write_text("{}", encoding="utf-8")
            build = SimpleNamespace(
                tokenizer_hash=TOKENIZER_HASH,
                tokenizer_vocab_size=16_000,
            )
            with (
                patch(
                    "lm_from_zero.acceleration_calibration.validate_shard_build",
                    return_value=build,
                ),
                patch(
                    "lm_from_zero.acceleration_calibration._file_sha256",
                    return_value=SHARD_HASH,
                ),
            ):
                plan = create_plan(build_path, repository_revision=GIT_REVISION)
            plan_path = root / "plan.json"
            write_plan(plan_path, plan)
            with patch(
                "lm_from_zero.acceleration_execution.validate_shard_build",
                return_value=build,
            ):
                with self.assertRaisesRegex(AccelerationExecutionError, "repetition"):
                    resolve_from_plan(plan_path, build_path, "dense", "baseline", 4)
                with self.assertRaisesRegex(
                    AccelerationExecutionError, "manifest hash"
                ):
                    resolve_from_plan(plan_path, build_path, "dense", "baseline", 1)

    def test_preflight_validates_exact_inputs_before_allocation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, build_path, tokenizer_path, raw_dry_run = self._plan_and_dry_run(
                root
            )
            dry_run = raw_dry_run
            build = _build()
            tokenizer = SimpleNamespace(
                model_hash=TOKENIZER_HASH,
                vocab_size=16_000,
            )
            with (
                patch(
                    "lm_from_zero.acceleration_execution.validate_shard_build",
                    return_value=build,
                ),
                patch(
                    "lm_from_zero.acceleration_execution._file_sha256",
                    return_value=SHARD_HASH,
                ),
                patch(
                    "lm_from_zero.acceleration_execution.ByteBPE.load",
                    return_value=tokenizer,
                ),
                patch("torch.nn.Module.to") as allocate,
            ):
                preflight = validate_calibration_execution_preflight(
                    dry_run,
                    plan_path=plan_path,
                    build_manifest=build_path,
                    tokenizer_model=tokenizer_path,
                    repository=root,
                    result_path=dry_run.result_path,
                    expected_cuda_device_name=DEVICE_NAME,
                    repository_state=GitMetadata(
                        revision=GIT_REVISION,
                        dirty=False,
                    ),
                    cuda_state=_cuda(),
                )

            allocate.assert_not_called()
            self.assertEqual(preflight.repository_revision, GIT_REVISION)
            self.assertEqual(preflight.cuda_compute_capability, (8, 9))
            self.assertFalse(preflight.allocation_performed)

    def test_preflight_rejects_missing_exact_inputs_and_wrong_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, build_path, tokenizer_path, raw_dry_run = self._plan_and_dry_run(
                root
            )
            dry_run = raw_dry_run
            common: dict[str, Any] = {
                "plan_path": plan_path,
                "build_manifest": build_path,
                "tokenizer_model": tokenizer_path,
                "repository": root,
                "result_path": dry_run.result_path,
                "expected_cuda_device_name": DEVICE_NAME,
                "repository_state": GitMetadata(
                    revision=GIT_REVISION,
                    dirty=False,
                ),
                "cuda_state": _cuda(),
            }
            with self.assertRaisesRegex(AccelerationExecutionError, "model.*missing"):
                validate_calibration_execution_preflight(
                    dry_run.model_copy(
                        update={
                            "training_config": {
                                key: value
                                for key, value in dry_run.training_config.items()
                                if key != "model"
                            }
                        }
                    ),
                    **common,
                )

            common["build_manifest"] = root / "missing-build.json"
            with self.assertRaisesRegex(
                AccelerationExecutionError, "shard build manifest is missing"
            ):
                validate_calibration_execution_preflight(dry_run, **common)
            common["build_manifest"] = build_path
            common["tokenizer_model"] = root / "missing-tokenizer.json"
            with self.assertRaisesRegex(
                AccelerationExecutionError, "tokenizer model is missing"
            ):
                validate_calibration_execution_preflight(dry_run, **common)
            common["tokenizer_model"] = tokenizer_path
            common["result_path"] = root / "wrong-result.json"
            with (
                patch(
                    "lm_from_zero.acceleration_execution.validate_shard_build",
                    return_value=_build(),
                ),
                patch(
                    "lm_from_zero.acceleration_execution._file_sha256",
                    return_value=SHARD_HASH,
                ),
                patch(
                    "lm_from_zero.acceleration_execution.ByteBPE.load",
                    return_value=SimpleNamespace(
                        model_hash=TOKENIZER_HASH,
                        vocab_size=16_000,
                    ),
                ),
                self.assertRaisesRegex(AccelerationExecutionError, "result path"),
            ):
                validate_calibration_execution_preflight(dry_run, **common)

    def test_preflight_rejects_plan_and_repository_revision_mismatches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, build_path, tokenizer_path, raw_dry_run = self._plan_and_dry_run(
                root
            )
            dry_run = raw_dry_run
            common: dict[str, Any] = {
                "plan_path": plan_path,
                "build_manifest": build_path,
                "tokenizer_model": tokenizer_path,
                "repository": root,
                "result_path": dry_run.result_path,
                "expected_cuda_device_name": DEVICE_NAME,
                "repository_state": GitMetadata(
                    revision=GIT_REVISION,
                    dirty=False,
                ),
                "cuda_state": _cuda(),
            }
            with self.assertRaisesRegex(AccelerationExecutionError, "plan artifact"):
                validate_calibration_execution_preflight(
                    dry_run.model_copy(update={"plan_sha256": "d" * 64}),
                    **common,
                )

            common["repository_state"] = GitMetadata(
                revision="e" * 40,
                dirty=False,
            )
            with (
                patch(
                    "lm_from_zero.acceleration_execution.validate_shard_build",
                    return_value=_build(),
                ),
                patch(
                    "lm_from_zero.acceleration_execution._file_sha256",
                    return_value=SHARD_HASH,
                ),
                patch(
                    "lm_from_zero.acceleration_execution.ByteBPE.load",
                    return_value=SimpleNamespace(
                        model_hash=TOKENIZER_HASH,
                        vocab_size=16_000,
                    ),
                ),
                self.assertRaisesRegex(AccelerationExecutionError, "planned revision"),
            ):
                validate_calibration_execution_preflight(dry_run, **common)

            common["repository_state"] = GitMetadata(
                revision=GIT_REVISION,
                dirty=True,
            )
            with (
                patch(
                    "lm_from_zero.acceleration_execution.validate_shard_build",
                    return_value=_build(),
                ),
                patch(
                    "lm_from_zero.acceleration_execution._file_sha256",
                    return_value=SHARD_HASH,
                ),
                patch(
                    "lm_from_zero.acceleration_execution.ByteBPE.load",
                    return_value=SimpleNamespace(
                        model_hash=TOKENIZER_HASH,
                        vocab_size=16_000,
                    ),
                ),
                self.assertRaisesRegex(
                    AccelerationExecutionError, "clean Git worktree"
                ),
            ):
                validate_calibration_execution_preflight(dry_run, **common)

    def test_preflight_rejects_non_cuda_and_unsupported_hardware(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, build_path, tokenizer_path, raw_dry_run = self._plan_and_dry_run(
                root
            )
            dry_run = raw_dry_run
            common: dict[str, Any] = {
                "plan_path": plan_path,
                "build_manifest": build_path,
                "tokenizer_model": tokenizer_path,
                "repository": root,
                "result_path": dry_run.result_path,
                "expected_cuda_device_name": DEVICE_NAME,
                "repository_state": GitMetadata(
                    revision=GIT_REVISION,
                    dirty=False,
                ),
            }
            tokenizer = SimpleNamespace(
                model_hash=TOKENIZER_HASH,
                vocab_size=16_000,
            )
            with (
                patch(
                    "lm_from_zero.acceleration_execution.validate_shard_build",
                    return_value=_build(),
                ),
                patch(
                    "lm_from_zero.acceleration_execution._file_sha256",
                    return_value=SHARD_HASH,
                ),
                patch(
                    "lm_from_zero.acceleration_execution.ByteBPE.load",
                    return_value=tokenizer,
                ),
                self.assertRaisesRegex(AccelerationExecutionError, "not available"),
            ):
                validate_calibration_execution_preflight(
                    dry_run,
                    cuda_state=_cuda(available=False),
                    **common,
                )

            with (
                patch(
                    "lm_from_zero.acceleration_execution.validate_shard_build",
                    return_value=_build(),
                ),
                patch(
                    "lm_from_zero.acceleration_execution._file_sha256",
                    return_value=SHARD_HASH,
                ),
                patch(
                    "lm_from_zero.acceleration_execution.ByteBPE.load",
                    return_value=tokenizer,
                ),
                self.assertRaisesRegex(AccelerationExecutionError, "hardware"),
            ):
                validate_calibration_execution_preflight(
                    dry_run,
                    cuda_state=_cuda(name="NVIDIA H100 80GB HBM3"),
                    **common,
                )

    def test_candidate_execution_requires_matching_baseline_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_path, build_path, tokenizer_path, _ = self._plan_and_dry_run(root)
            with (
                patch(
                    "lm_from_zero.acceleration_execution.validate_shard_build",
                    return_value=_build(),
                ),
                patch(
                    "lm_from_zero.acceleration_execution._file_sha256",
                    return_value=SHARD_HASH,
                ),
                patch(
                    "lm_from_zero.acceleration_execution.validate_calibration_execution_preflight"
                ),
                self.assertRaisesRegex(
                    AccelerationExecutionError, "baseline repetition"
                ),
            ):
                execute_calibration_cell(
                    plan_path,
                    build_path,
                    tokenizer_path,
                    root,
                    "dense",
                    "sampled-telemetry",
                    1,
                )


if __name__ == "__main__":
    unittest.main()
