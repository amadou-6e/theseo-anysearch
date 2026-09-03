"""Exact empty-grid route generation for waypoint curricula."""

from __future__ import annotations

import math
from typing import TypeAlias

import numpy as np

from theseo_anysearch.environments.action_spaces import action_step_distance
from theseo_anysearch.worlds.extent import WorldExtent, contains_task_coordinate
from pydantic import BaseModel, ConfigDict

Waypoint: TypeAlias = tuple[int, int, int]


class WaypointRoute(BaseModel):
    """One curriculum stage containing an ordered route."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    start: Waypoint
    waypoints: tuple[Waypoint, ...]

    @property
    def goal(self) -> Waypoint:
        return self.waypoints[-1]


def segment_lengths(total_distance: int, segment_distance: int) -> tuple[int, ...]:
    """Split an exact route length into equal segments and one residual."""
    if total_distance < 1 or segment_distance < 1:
        raise ValueError("route and segment distances must be positive")
    full_segments, residual = divmod(total_distance, segment_distance)
    lengths = [segment_distance] * full_segments
    if residual:
        lengths.append(residual)
    return tuple(lengths)


def sample_route(
    *,
    start: Waypoint,
    total_distance: int,
    segment_distance: int,
    action_mode: str,
    seed: int,
    grid_size: int | None = None,
    extent: WorldExtent | None = None,
) -> WaypointRoute:
    """Sample spherical directions while enforcing exact graph distances."""
    if extent is None:
        size = 32 if grid_size is None else grid_size
        extent = (size, size, size)
    if not contains_task_coordinate(extent, start):
        raise ValueError("route start must be inside the grid")
    rng = np.random.default_rng(seed)
    current = start
    waypoints: list[Waypoint] = []
    visited = {start}
    for distance in segment_lengths(total_distance, segment_distance):
        ranges = tuple(
            range(max(1, coordinate - distance), min(axis, coordinate + distance) + 1)
            for coordinate, axis in zip(current, extent)
        )
        candidates = [
            (x, y, z)
            for x in ranges[0]
            for y in ranges[1]
            for z in ranges[2]
            if (x, y, z) not in visited
            and action_step_distance(current, (x, y, z), action_mode) == distance
        ]
        if not candidates:
            raise ValueError(
                f"segment distance {distance} cannot fit from {current} in this grid"
            )
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)

        def alignment(candidate: Waypoint) -> float:
            delta = np.asarray(candidate, dtype=float) - np.asarray(current, dtype=float)
            return float(np.dot(delta / np.linalg.norm(delta), direction))

        best_alignment = max(alignment(candidate) for candidate in candidates)
        best = [
            candidate
            for candidate in candidates
            if math.isclose(alignment(candidate), best_alignment)
        ]
        current = best[int(rng.integers(len(best)))]
        waypoints.append(current)
        visited.add(current)
    return WaypointRoute(start=start, waypoints=tuple(waypoints))


def route_distance(route: WaypointRoute, action_mode: str) -> int:
    """Return cumulative configured-action distance across the route."""
    points = (route.start, *route.waypoints)
    return sum(
        action_step_distance(points[index], points[index + 1], action_mode)
        for index in range(len(points) - 1)
    )
