"""Finite-world extent compatibility helpers.

Task coordinates remain one-based. Storage coordinates in compiled packs remain
zero-based; conversion belongs at the world boundary rather than in callers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, TypeAlias

WorldExtent: TypeAlias = tuple[int, int, int]


def resolve_extent(config: Mapping[str, Any], default: int = 32) -> WorldExtent:
    """Resolve an explicit three-axis extent or the legacy cubic shorthand."""

    raw = config.get("extent")
    if raw is None:
        size = int(config.get("grid_size") or default)
        raw = (size, size, size)
    extent = tuple(int(axis) for axis in raw)
    if len(extent) != 3 or any(axis < 1 for axis in extent):
        raise ValueError("extent must contain three positive axes")
    scalar = config.get("grid_size")
    if scalar is not None and extent != (int(scalar),) * 3:
        raise ValueError("grid_size and extent describe different world bounds")
    return extent  # type: ignore[return-value]


def contains_task_coordinate(extent: WorldExtent, coordinate: Sequence[int]) -> bool:
    """Return whether a one-based task coordinate lies in ``extent``."""

    return len(coordinate) == 3 and all(
        1 <= int(value) <= extent[index] for index, value in enumerate(coordinate)
    )


def task_center(extent: WorldExtent) -> tuple[int, int, int]:
    """Return the legacy-compatible lower central voxel on each axis."""

    return tuple((axis + 1) // 2 for axis in extent)  # type: ignore[return-value]


def maximum_manhattan(extent: WorldExtent) -> int:
    return sum(axis - 1 for axis in extent)


def maximum_euclidean(extent: WorldExtent) -> float:
    return math.sqrt(sum((axis - 1) ** 2 for axis in extent))


def normalization_extent(config: Mapping[str, Any]) -> WorldExtent:
    """Resolve bounds used by observations without changing observation shapes."""

    return resolve_extent(config)


def resolve_task_extent(config: Mapping[str, Any]) -> WorldExtent:
    """Resolve bounds supported by the current native task-coordinate ABI."""

    extent = resolve_extent(config)
    if any(axis > 2**16 - 1 for axis in extent):
        raise ValueError(
            "live environment task coordinates currently support extent axes up to 65535"
        )
    return extent
