from __future__ import annotations

import math
import unittest

import torch
from torch import nn

from lm_from_zero.acceleration_statistics import (
    resolve_parameter_group_measurements,
    snapshot_parameter_groups,
    update_rms_values,
)
from lm_from_zero.training.optimization import ParameterPartition, partition_parameters


class ParameterStatisticsTests(unittest.TestCase):
    def test_resolves_known_group_statistics_in_partition_order(self) -> None:
        decay = nn.Parameter(torch.tensor([3.0, 4.0]))
        no_decay = nn.Parameter(torch.tensor([1.0, -1.0]))
        partition = ParameterPartition(
            decay_names=("decay",),
            no_decay_names=("bias",),
            decay_parameters=(decay,),
            no_decay_parameters=(no_decay,),
        )
        snapshot = snapshot_parameter_groups(partition)

        with torch.no_grad():
            decay.copy_(torch.tensor([4.0, 6.0]))
            no_decay.copy_(torch.tensor([1.0, 0.0]))
        decay.grad = torch.tensor([2.0, -2.0])
        no_decay.grad = torch.tensor([0.0, 3.0])

        measurements = resolve_parameter_group_measurements(snapshot)

        self.assertEqual(
            tuple(item.name for item in measurements), ("decay", "no_decay")
        )
        self.assertEqual(tuple(item.parameter_count for item in measurements), (2, 2))
        self.assertAlmostEqual(measurements[0].weight_rms, math.sqrt(25.0 / 2.0))
        self.assertAlmostEqual(measurements[0].gradient_rms, 2.0)
        self.assertAlmostEqual(measurements[0].update_rms, math.sqrt(5.0 / 2.0))
        self.assertAlmostEqual(
            measurements[0].effective_learning_rate, math.sqrt(5.0 / 25.0)
        )
        self.assertAlmostEqual(
            measurements[0].angular_learning_rate,
            math.acos(36.0 / math.sqrt(25.0 * 52.0)),
        )
        self.assertEqual(
            update_rms_values(measurements),
            (measurements[0].update_rms, measurements[1].update_rms),
        )

    def test_aggregates_parameters_and_treats_missing_gradients_as_zero(self) -> None:
        first = nn.Parameter(torch.tensor([1.0, 2.0]))
        second = nn.Parameter(torch.tensor([2.0]))
        partition = ParameterPartition(
            decay_names=("first", "second"),
            no_decay_names=(),
            decay_parameters=(first, second),
            no_decay_parameters=(),
        )
        snapshot = snapshot_parameter_groups(partition)
        first.grad = torch.tensor([3.0, 4.0])
        with torch.no_grad():
            first.add_(1.0)
            second.sub_(2.0)

        (measurement,) = resolve_parameter_group_measurements(snapshot)

        self.assertEqual(measurement.name, "decay")
        self.assertEqual(measurement.parameter_count, 3)
        self.assertAlmostEqual(measurement.weight_rms, math.sqrt(9.0 / 3.0))
        self.assertAlmostEqual(measurement.gradient_rms, math.sqrt(25.0 / 3.0))
        self.assertAlmostEqual(measurement.update_rms, math.sqrt(6.0 / 3.0))

    def test_snapshot_remains_stable_across_real_optimizer_update(self) -> None:
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(3, 2), nn.LayerNorm(2))
        partition = partition_parameters(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        snapshot = snapshot_parameter_groups(partition)

        loss = model(torch.ones(4, 3)).square().mean()
        loss.backward()
        optimizer.step()
        measurements = resolve_parameter_group_measurements(snapshot)

        self.assertEqual(
            tuple(item.name for item in measurements), ("decay", "no_decay")
        )
        self.assertTrue(all(item.parameter_count > 0 for item in measurements))
        self.assertTrue(any(item.update_rms > 0.0 for item in measurements))
        self.assertTrue(all(math.isfinite(item.gradient_rms) for item in measurements))

    def test_zero_norm_group_has_finite_relative_metrics(self) -> None:
        parameter = nn.Parameter(torch.zeros(2))
        partition = ParameterPartition(
            decay_names=(),
            no_decay_names=("zero",),
            decay_parameters=(),
            no_decay_parameters=(parameter,),
        )
        snapshot = snapshot_parameter_groups(partition)
        with torch.no_grad():
            parameter.fill_(1.0)

        (measurement,) = resolve_parameter_group_measurements(snapshot)

        self.assertEqual(measurement.weight_rms, 0.0)
        self.assertEqual(measurement.effective_learning_rate, 0.0)
        self.assertEqual(measurement.angular_learning_rate, 0.0)

    def test_empty_snapshot_is_rejected(self) -> None:
        partition = ParameterPartition(
            decay_names=(),
            no_decay_names=(),
            decay_parameters=(),
            no_decay_parameters=(),
        )

        with self.assertRaisesRegex(ValueError, "at least one parameter"):
            snapshot_parameter_groups(partition)


if __name__ == "__main__":
    unittest.main()
