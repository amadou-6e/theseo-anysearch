"""Continuous axis-segment checks against the complete occupied voxel cubes."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from theseo_anysearch.environments.shape_collision import Coordinate, Sphere, swept_voxel_segment_clear


def axis_segment_clear(
    truth: np.ndarray,
    start: Coordinate,
    endpoint: Coordinate,
    *,
    body_radius_voxels: float,
) -> bool:
    """Compatibility wrapper for the sphere-only shared voxel checker."""

    return swept_voxel_segment_clear(
        truth, start, endpoint, shape=Sphere(body_radius_voxels)
    )


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
