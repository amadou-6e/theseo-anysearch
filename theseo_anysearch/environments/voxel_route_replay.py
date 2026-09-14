"""Continuous axis-segment checks against the complete occupied voxel cubes."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

Coordinate = tuple[int, int, int]


def axis_segment_clear(
    truth: np.ndarray,
    start: Coordinate,
    endpoint: Coordinate,
    *,
    body_radius_voxels: float,
) -> bool:
    """Check a swept sphere against occupied AABBs and solid world bounds."""

    if body_radius_voxels < 0 or not np.isfinite(body_radius_voxels):
        raise ValueError("body radius must be finite and nonnegative")
    if truth.ndim != 3 or truth.dtype.kind not in "biu":
        raise ValueError("expected a binary three-dimensional voxel grid")
    if sum(a != b for a, b in zip(start, endpoint)) != 1:
        raise ValueError("candidate path must be one axis-aligned segment")
    shape = truth.shape
    if any(not 0 <= start[axis] < shape[axis] for axis in range(3)):
        raise ValueError("start outside world")
    if any(not 0 <= point[axis] < shape[axis] for point in (start, endpoint) for axis in range(3)):
        return False
    if body_radius_voxels > 0 and any(
        min(start[index], endpoint[index]) - body_radius_voxels <= -0.5
        or max(start[index], endpoint[index]) + body_radius_voxels >= shape[index] - 0.5
        for index in range(3)
    ):
        return False
    axis = next(index for index in range(3) if start[index] != endpoint[index])
    margin = int(np.ceil(body_radius_voxels + 0.5))
    lows = [max(0, min(start[i], endpoint[i]) - margin) for i in range(3)]
    highs = [min(shape[i], max(start[i], endpoint[i]) + margin + 1) for i in range(3)]
    region = truth[tuple(slice(a, b) for a, b in zip(lows, highs))]
    if np.any(region > 1):
        raise ValueError("expected a binary three-dimensional voxel grid")
    occupied = np.argwhere(region != 0)
    if len(occupied) == 0:
        return True
    occupied += np.asarray(lows)
    distances = np.zeros((len(occupied), 3), dtype=np.float64)
    for index in range(3):
        if index == axis:
            low, high = sorted((start[index], endpoint[index]))
            distances[:, index] = np.maximum.reduce(
                (
                    occupied[:, index] - 0.5 - high,
                    low - occupied[:, index] - 0.5,
                    np.zeros(len(occupied)),
                )
            )
        else:
            distances[:, index] = np.maximum(
                np.abs(occupied[:, index] - start[index]) - 0.5, 0
            )
    return bool(np.all(np.sum(distances**2, axis=1) > body_radius_voxels**2))


def shortest_six_axis_route(
    passable: np.ndarray, start: Coordinate, goal: Coordinate
) -> tuple[Coordinate, ...] | None:
    """Deterministic BFS on a conservative body-center mask."""

    if passable.ndim != 3 or passable.dtype != np.bool_:
        raise ValueError("expected a Boolean three-dimensional passability mask")
    if any(not 0 <= point[axis] < passable.shape[axis] for point in (start, goal) for axis in range(3)):
        raise ValueError("route endpoint outside world")
    if not passable[start] or not passable[goal]:
        return None
    queue = deque([start])
    parents: dict[Coordinate, Coordinate | None] = {start: None}
    while queue:
        cell = queue.popleft()
        if cell == goal:
            route = []
            while cell is not None:
                route.append(cell)
                cell = parents[cell]
            return tuple(reversed(route))
        for axis in range(3):
            for delta in (-1, 1):
                neighbor = list(cell)
                neighbor[axis] += delta
                candidate = tuple(neighbor)
                if (
                    0 <= neighbor[axis] < passable.shape[axis]
                    and candidate not in parents
                    and passable[candidate]
                ):
                    parents[candidate] = cell
                    queue.append(candidate)
    return None


def replay_six_axis_route(
    truth: np.ndarray,
    route: tuple[Coordinate, ...],
    *,
    body_radius_m: float,
    meters_per_voxel: float,
) -> None:
    """Reject a route unless every segment is clear of all source voxel cubes."""

    if not math.isfinite(meters_per_voxel) or meters_per_voxel <= 0:
        raise ValueError("meters per voxel must be positive and finite")
    if truth.ndim != 3 or truth.dtype.kind not in "biu" or np.any(truth > 1):
        raise ValueError("expected a binary three-dimensional voxel grid")
    radius = body_radius_m / meters_per_voxel
    if not route:
        raise ValueError("empty route")
    if len(route) == 1:
        raise ValueError("route needs at least one segment")
    for start, end in zip(route, route[1:]):
        if sum(abs(a - b) for a, b in zip(start, end)) != 1:
            raise ValueError("route contains a non-six-axis step")
        if not axis_segment_clear(truth, start, end, body_radius_voxels=radius):
            raise ValueError("route collides with occupied voxel cubes or world bounds")
