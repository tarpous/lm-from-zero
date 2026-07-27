from __future__ import annotations

import tempfile
import unittest
from importlib.metadata import PackageNotFoundError
from pathlib import Path
from unittest.mock import patch

import torch
from torch.nn import functional as F

from lm_from_zero.mamba2_oracle import (
    Mamba2OracleConfig,
    Mamba2OracleError,
    _load_oracle_scan,
    verify_mamba2_oracle,
    write_mamba2_oracle_report,
)
from lm_from_zero.models.mamba2 import ssd_chunked


def _reference_scan(
    x: torch.Tensor,
    raw_dt: torch.Tensor,
    negative_a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    *,
    chunk_size: int,
    D: torch.Tensor,
    dt_bias: torch.Tensor,
    dt_softplus: bool,
    initial_states: torch.Tensor,
    return_final_states: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not dt_softplus or not return_final_states:
        raise AssertionError("test oracle requires the complete comparison contract")
    dt = F.softplus(raw_dt + dt_bias)
    output, state = ssd_chunked(
        x * dt.unsqueeze(-1),
        dt * negative_a,
        b,
        c,
        chunk_size=chunk_size,
        initial_state=initial_states,
    )
    return output + D.view(1, 1, -1, 1) * x, state


class Mamba2OracleTests(unittest.TestCase):
    def test_reference_injection_matches_and_writes_canonical_report(self) -> None:
        config = Mamba2OracleConfig(
            batch_size=1,
            sequence_length=9,
            num_heads=4,
            head_dim=3,
            num_groups=2,
            state_size=5,
            chunk_size=4,
            device="cpu",
        )
        with (
            patch(
                "lm_from_zero.mamba2_oracle._load_oracle_scan",
                return_value=_reference_scan,
            ),
            patch(
                "lm_from_zero.mamba2_oracle._oracle_version",
                return_value="2.3.2.post1",
            ),
        ):
            result = verify_mamba2_oracle(config)

        self.assertEqual(result.output_max_abs_error, 0.0)
        self.assertEqual(result.state_max_abs_error, 0.0)
        self.assertTrue(result.output_close)
        self.assertTrue(result.state_close)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "oracle.json"
            write_mamba2_oracle_report(output, result)
            self.assertEqual(output.read_bytes(), result.canonical_bytes() + b"\n")
            temporary = output.with_name(f".{output.name}.tmp")
            temporary.write_text("incomplete", encoding="utf-8")
            with self.assertRaisesRegex(Mamba2OracleError, "incomplete"):
                write_mamba2_oracle_report(output, result)

    def test_rejects_missing_package_invalid_results_and_numerical_mismatch(
        self,
    ) -> None:
        with (
            patch(
                "lm_from_zero.mamba2_oracle.import_module",
                side_effect=ImportError("missing"),
            ),
            self.assertRaisesRegex(Mamba2OracleError, "working CUDA extensions"),
        ):
            _load_oracle_scan()

        config = Mamba2OracleConfig(
            batch_size=1,
            sequence_length=3,
            num_heads=2,
            head_dim=2,
            num_groups=1,
            state_size=2,
            chunk_size=2,
            device="cpu",
        )
        with (
            patch(
                "lm_from_zero.mamba2_oracle._load_oracle_scan",
                return_value=lambda *args, **kwargs: torch.zeros(1),
            ),
            self.assertRaisesRegex(Mamba2OracleError, "output and final state"),
        ):
            verify_mamba2_oracle(config)

        def wrong_scan(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
            reference = _reference_scan(*args, **kwargs)  # type: ignore[arg-type]
            return reference[0] + 1, reference[1]

        with (
            patch(
                "lm_from_zero.mamba2_oracle._load_oracle_scan",
                return_value=wrong_scan,
            ),
            self.assertRaisesRegex(Mamba2OracleError, "exceeds oracle tolerance"),
        ):
            verify_mamba2_oracle(config)

    def test_rejects_invalid_config_and_missing_package_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            Mamba2OracleConfig(num_heads=3, num_groups=2)
        with self.assertRaisesRegex(ValueError, "CPU or CUDA"):
            Mamba2OracleConfig(device="mps")
        with (
            patch(
                "lm_from_zero.mamba2_oracle.version",
                side_effect=PackageNotFoundError,
            ),
            patch(
                "lm_from_zero.mamba2_oracle._load_oracle_scan",
                return_value=_reference_scan,
            ),
            self.assertRaisesRegex(Mamba2OracleError, "metadata"),
        ):
            verify_mamba2_oracle(
                Mamba2OracleConfig(
                    sequence_length=2,
                    num_heads=2,
                    head_dim=2,
                    num_groups=1,
                    state_size=2,
                    chunk_size=2,
                    device="cpu",
                )
            )


if __name__ == "__main__":
    unittest.main()
