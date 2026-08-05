from __future__ import annotations

import unittest
from collections import Counter

from lm_from_zero.acceleration_runtime import (
    ProcessCpuTimes,
    capture_process_cpu_times,
    classify_native_sdpa_backend,
    process_cpu_utilization,
    profile_cuda_sdpa_backend,
    read_compile_graph_break_counters,
    reset_compile_graph_break_counters,
)


class AccelerationRuntimeTests(unittest.TestCase):
    def test_graph_break_summary_is_counted_normalized_and_sorted(self) -> None:
        counter: Counter[object] = Counter(
            {
                "z reason": 2,
                " a\n reason ": 1,
                "ignored": 0,
            }
        )

        summary = read_compile_graph_break_counters(counter)

        self.assertEqual(summary.graph_breaks, 3)
        self.assertEqual(summary.reasons, ("a reason", "z reason"))

    def test_graph_break_reset_clears_injected_counter(self) -> None:
        counter: Counter[object] = Counter({"reason": 2})

        reset_compile_graph_break_counters(counter)

        self.assertEqual(counter, Counter())

    def test_sdpa_operator_classification(self) -> None:
        cases = {
            "flash": [
                "aten::scaled_dot_product_attention",
                "aten::_scaled_dot_product_flash_attention_cuda",
            ],
            "efficient": [
                "aten::scaled_dot_product_attention",
                "aten::_scaled_dot_product_efficient_attention",
            ],
            "math": [
                "aten::scaled_dot_product_attention",
                "aten::_scaled_dot_product_attention_math",
            ],
            "unknown": ["aten::scaled_dot_product_attention"],
            None: ["aten::linear", "aten::softmax"],
        }
        for expected, operators in cases.items():
            with self.subTest(expected=expected):
                self.assertEqual(classify_native_sdpa_backend(operators), expected)

    def test_mixed_sdpa_implementations_are_unknown(self) -> None:
        self.assertEqual(
            classify_native_sdpa_backend(
                [
                    "aten::_scaled_dot_product_flash_attention_cuda",
                    "aten::_scaled_dot_product_attention_math",
                ]
            ),
            "unknown",
        )

    def test_profile_wrapper_invokes_callback_via_injected_profiler(self) -> None:
        calls: list[str] = []

        def forward() -> object:
            calls.append("forward")
            return object()

        def profiler(callback: object) -> tuple[str, ...]:
            self.assertIs(callback, forward)
            assert callable(callback)
            callback()
            return ("aten::_scaled_dot_product_flash_attention_cuda",)

        backend = profile_cuda_sdpa_backend(forward, profiler=profiler)

        self.assertEqual(backend, "flash")
        self.assertEqual(calls, ["forward"])

    def test_capture_process_times_uses_injected_clocks(self) -> None:
        snapshot = capture_process_cpu_times(
            wall_clock=lambda: 12.5,
            process_clock=lambda: 3.25,
        )

        self.assertEqual(snapshot, ProcessCpuTimes(12.5, 3.25))

    def test_process_cpu_utilization_is_process_to_wall_ratio(self) -> None:
        utilization = process_cpu_utilization(
            ProcessCpuTimes(wall_seconds=10.0, process_seconds=2.0),
            ProcessCpuTimes(wall_seconds=14.0, process_seconds=4.0),
        )

        self.assertEqual(utilization, 0.5)

    def test_process_cpu_utilization_rejects_invalid_deltas(self) -> None:
        with self.assertRaisesRegex(ValueError, "wall-clock"):
            process_cpu_utilization(
                ProcessCpuTimes(1.0, 1.0), ProcessCpuTimes(1.0, 2.0)
            )
        with self.assertRaisesRegex(ValueError, "process-time"):
            process_cpu_utilization(
                ProcessCpuTimes(1.0, 2.0), ProcessCpuTimes(2.0, 1.0)
            )


if __name__ == "__main__":
    unittest.main()
