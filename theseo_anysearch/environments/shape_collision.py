"""Swept collision checks for explicit shapes against occupied voxel cubes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real

import numpy as np

Coordinate = tuple[int, int, int]


class AxisHeading(str, Enum):
    """Six travel headings; these do not encode roll about the heading axis."""

    POS_X = "+x"
    NEG_X = "-x"
    POS_Y = "+y"
    NEG_Y = "-y"
    POS_Z = "+z"
    NEG_Z = "-z"


@dataclass(frozen=True)
class Sphere:
    radius_voxels: float

    def __post_init__(self) -> None:
        radius = self.radius_voxels
        if (
            isinstance(radius, bool)
            or not isinstance(radius, Real)
            or not math.isfinite(radius)
            or radius < 0
        ):
            raise ValueError("sphere radius must be finite and nonnegative")


def _check_coordinate(point: Coordinate) -> None:
    if len(point) != 3 or any(
        isinstance(value, bool) or not isinstance(value, Integral) for value in point
    ):
        raise ValueError("voxel coordinate must contain three integers")


def swept_voxel_segment_clear(
    truth: np.ndarray,
    start: Coordinate,
    endpoint: Coordinate,
    *,
    shape: Sphere,
    heading: AxisHeading | None = None,
) -> bool:
    """Check an axis-aligned segment against occupied cubes and solid world bounds.

    A sphere is invariant to heading. Other shapes are intentionally unsupported.
    The grid must be complete occupancy truth, not a visibility or clearance mask.
    """

    if not isinstance(shape, Sphere):
        raise TypeError("only Sphere collision shapes are supported")
    if heading is not None and not isinstance(heading, AxisHeading):
        raise ValueError("heading must be one of the six AxisHeading values")
    if truth.ndim != 3 or truth.dtype.kind not in "biu":
        raise ValueError("expected a binary three-dimensional voxel grid")
    _check_coordinate(start)
    _check_coordinate(endpoint)
    if sum(a != b for a, b in zip(start, endpoint)) != 1:
        raise ValueError("candidate path must be one axis-aligned segment")
    extent = truth.shape
    if any(not 0 <= start[axis] < extent[axis] for axis in range(3)):
        raise ValueError("start outside world")
    if any(not 0 <= endpoint[axis] < extent[axis] for axis in range(3)):
        return False

    radius = shape.radius_voxels
    if radius > 0 and any(
        min(start[axis], endpoint[axis]) - radius <= -0.5
        or max(start[axis], endpoint[axis]) + radius >= extent[axis] - 0.5
        for axis in range(3)
    ):
        return False

    travel_axis = next(axis for axis in range(3) if start[axis] != endpoint[axis])
    margin = math.ceil(radius + 0.5)
    low = [max(0, min(start[axis], endpoint[axis]) - margin) for axis in range(3)]
    high = [min(extent[axis], max(start[axis], endpoint[axis]) + margin + 1) for axis in range(3)]
    region = truth[tuple(slice(a, b) for a, b in zip(low, high))]
    if np.any(region > 1):
        raise ValueError("expected a binary three-dimensional voxel grid")
    occupied = np.argwhere(region != 0)
    if len(occupied) == 0:
        return True
    occupied += np.asarray(low)
    distances = np.zeros((len(occupied), 3), dtype=np.float64)
    for axis in range(3):
        if axis == travel_axis:
            segment_low, segment_high = sorted((start[axis], endpoint[axis]))
            distances[:, axis] = np.maximum.reduce((
                occupied[:, axis] - 0.5 - segment_high,
                segment_low - occupied[:, axis] - 0.5,
                np.zeros(len(occupied)),
            ))
        else:
            distances[:, axis] = np.maximum(np.abs(occupied[:, axis] - start[axis]) - 0.5, 0)
    return bool(np.all(np.sum(distances**2, axis=1) > radius**2))
