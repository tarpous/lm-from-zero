"""Opt-in parameter-scale measurements for acceleration calibration."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from lm_from_zero.acceleration_calibration import ParameterGroupMeasurement
from lm_from_zero.training.optimization import ParameterPartition


@dataclass(frozen=True, slots=True)
class _ParameterGroupSnapshot:
    """Device-resident starting values for one optimizer partition."""

    name: str
    parameters: tuple[nn.Parameter, ...]
    initial_values: tuple[Tensor, ...]
    parameter_count: int


@dataclass(frozen=True, slots=True)
class ParameterStatisticsSnapshot:
    """Device-resident decay/no-decay snapshot for a calibration window."""

    groups: tuple[_ParameterGroupSnapshot, ...]
    device: torch.device


def snapshot_parameter_groups(
    partition: ParameterPartition,
) -> ParameterStatisticsSnapshot:
    """Clone both optimizer partitions without copying parameter data to the CPU."""

    named_groups = (
        ("decay", partition.decay_parameters),
        ("no_decay", partition.no_decay_parameters),
    )
    devices = {parameter.device for _, group in named_groups for parameter in group}
    if not devices:
        raise ValueError("parameter statistics require at least one parameter")
    if len(devices) != 1:
        raise ValueError("parameter statistics require one shared device")
    device = next(iter(devices))

    snapshots: list[_ParameterGroupSnapshot] = []
    with torch.no_grad():
        for name, parameters in named_groups:
            if not parameters:
                continue
            count = sum(parameter.numel() for parameter in parameters)
            snapshots.append(
                _ParameterGroupSnapshot(
                    name=name,
                    parameters=parameters,
                    initial_values=tuple(
                        parameter.detach().clone() for parameter in parameters
                    ),
                    parameter_count=count,
                )
            )
    return ParameterStatisticsSnapshot(groups=tuple(snapshots), device=device)


def _sum_of_squares(value: Tensor) -> Tensor:
    if value.is_sparse:
        value = value.coalesce().values()
    return value.detach().float().square().sum()


def _group_reductions(snapshot: _ParameterGroupSnapshot) -> Tensor:
    device = snapshot.initial_values[0].device
    totals = torch.zeros(5, dtype=torch.float32, device=device)
    with torch.no_grad():
        for parameter, initial in zip(
            snapshot.parameters, snapshot.initial_values, strict=True
        ):
            current = parameter.detach().float()
            initial_float = initial.float()
            update = current - initial_float
            totals[0] += initial_float.square().sum()
            if parameter.grad is not None:
                totals[1] += _sum_of_squares(parameter.grad)
            totals[2] += update.square().sum()
            totals[3] += current.square().sum()
            totals[4] += (initial_float * update).sum()
    return totals


def resolve_parameter_group_measurements(
    snapshot: ParameterStatisticsSnapshot,
) -> tuple[ParameterGroupMeasurement, ...]:
    """Resolve group RMS and relative-update metrics with one device sync.

    Weight RMS is measured at the beginning of the window, while gradient RMS
    uses gradients retained after the final backward/update. Update RMS compares
    the current parameters with the snapshot. Effective learning rate is the
    update-to-weight norm ratio, and angular learning rate is the angle in
    radians between the initial and current flattened parameter vectors. The two
    relative metrics are defined as zero for a zero-norm starting group.
    """

    reductions = torch.stack([_group_reductions(group) for group in snapshot.groups])
    if snapshot.device.type == "cuda":
        torch.cuda.synchronize(snapshot.device)
    values = reductions.cpu().tolist()

    measurements: list[ParameterGroupMeasurement] = []
    for group, (weight_sq, gradient_sq, update_sq, current_sq, update_cross) in zip(
        snapshot.groups, values, strict=True
    ):
        count = group.parameter_count
        weight_rms = math.sqrt(weight_sq / count)
        gradient_rms = math.sqrt(gradient_sq / count)
        update_rms = math.sqrt(update_sq / count)
        if weight_sq == 0.0:
            effective_learning_rate = 0.0
            angular_learning_rate = 0.0
        else:
            effective_learning_rate = math.sqrt(update_sq / weight_sq)
            if current_sq == 0.0:
                angular_learning_rate = 0.0
            else:
                perpendicular_sq = max(
                    0.0, update_sq - update_cross * update_cross / weight_sq
                )
                updated_parallel = math.sqrt(weight_sq) + (
                    update_cross / math.sqrt(weight_sq)
                )
                angular_learning_rate = math.atan2(
                    math.sqrt(perpendicular_sq), updated_parallel
                )
        measurements.append(
            ParameterGroupMeasurement(
                name=group.name,
                parameter_count=count,
                weight_rms=weight_rms,
                gradient_rms=gradient_rms,
                update_rms=update_rms,
                angular_learning_rate=angular_learning_rate,
                effective_learning_rate=effective_learning_rate,
            )
        )
    return tuple(measurements)


def update_rms_values(
    measurements: tuple[ParameterGroupMeasurement, ...],
) -> tuple[float, ...]:
    """Return update RMS values in the deterministic measurement order."""

    return tuple(measurement.update_rms for measurement in measurements)
