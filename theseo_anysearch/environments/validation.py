"""Reusable structural geometry and task-feasibility validation contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from theseo_anysearch.environments.action_spaces import action_step_distance
from theseo_anysearch.heuristic.search import PlannerBudgetExceeded, astar_path
from theseo_anysearch.settings.environment.geometry import RoutingDifficultyBand
from theseo_anysearch.worlds.extent import WorldExtent, contains_task_coordinate

Coordinate = tuple[int, int, int]
GeometryRejection = Literal["out_of_bounds", "duplicate_coordinate"]
TaskRejection = Literal[
    "missing_goal",
    "invalid_endpoint",
    "occupied_start",
    "occupied_goal",
    "no_path",
    "planner_budget_exhausted",
    "episode_budget_exceeded",
    "difficulty_band_rejected",
]


class OccupancyRead(Protocol):
    """Minimal bounded occupancy interface needed by task validation."""

    def occupied(self, coordinate: Coordinate) -> bool: ...


class InMemoryOccupancy:
    """Point-query adapter over an explicitly materialized coordinate set."""

    def __init__(self, coordinates: Iterable[Coordinate]) -> None:
        self._coordinates = frozenset(tuple(item) for item in coordinates)

    def occupied(self, coordinate: Coordinate) -> bool:
        return coordinate in self._coordinates


class BoundedWorldRead:
    """Point-query adapter that never requests or enumerates a whole world."""

    def __init__(self, occupied: Callable[[Coordinate], bool]) -> None:
        self._occupied = occupied

    def occupied(self, coordinate: Coordinate) -> bool:
        return bool(self._occupied(coordinate))


class GeometryValidationResult(BaseModel):
    """Stable result of structural validation for in-memory geometry."""

    model_config = ConfigDict(frozen=True)
    valid: bool
    coordinate_count: int
    rejection_reason: GeometryRejection | None = None
    rejected_coordinate: Coordinate | None = None


class TaskFeasibilityResult(BaseModel):
    """Stable planner-backed result for one selected navigation task."""

    model_config = ConfigDict(frozen=True)
    feasible: bool
    planner: Literal["astar"] = "astar"
    path: tuple[Coordinate, ...] = ()
    path_length: int | None = None
    graph_nodes: int = 0
    graph_edges: int = 0
    rejection_reason: TaskRejection | None = None
    difficulty: "RoutingDifficultyDescriptors | None" = None
    difficulty_band: str | None = None


class RoutingDifficultyDescriptors(BaseModel):
    """Deterministic, non-scalar description derived from one planned path."""

    model_config = ConfigDict(frozen=True)
    direct_distance: int
    shortest_path_length: int
    detour_ratio: float
    direction_changes: int
    vertical_displacement: int
    expansion_count: int
    minimum_clearance: int | None = None
    mean_clearance: float | None = None


def validate_geometry(
    coordinates: Iterable[Sequence[int]], extent: WorldExtent
) -> GeometryValidationResult:
    """Validate bounds and uniqueness without imposing task semantics."""

    seen: set[Coordinate] = set()
    for raw in coordinates:
        coordinate = tuple(int(value) for value in raw)
        if len(coordinate) != 3 or not contains_task_coordinate(extent, coordinate):
            return GeometryValidationResult(
                valid=False,
                coordinate_count=len(seen),
                rejection_reason="out_of_bounds",
                rejected_coordinate=coordinate if len(coordinate) == 3 else None,
            )
        if coordinate in seen:
            return GeometryValidationResult(
                valid=False,
                coordinate_count=len(seen),
                rejection_reason="duplicate_coordinate",
                rejected_coordinate=coordinate,
            )
        seen.add(coordinate)
    return GeometryValidationResult(valid=True, coordinate_count=len(seen))


def validate_task_feasibility(
    world: OccupancyRead,
    *,
    start: Coordinate | None,
    goal: Coordinate | None,
    extent: WorldExtent,
    directions: Sequence[Coordinate],
    action_mode: str,
    maximum_search_nodes: int,
    maximum_steps: int,
    recovery_margin_steps: int = 0,
    clearance_radius: int | None = None,
    difficulty_bands: Sequence[RoutingDifficultyBand] = (),
    accepted_difficulty_bands: Sequence[str] = (),
) -> TaskFeasibilityResult:
    """Validate endpoints and run the shared deterministic A* implementation."""

    if start is None or goal is None:
        return TaskFeasibilityResult(feasible=False, rejection_reason="missing_goal")
    if not contains_task_coordinate(extent, start) or not contains_task_coordinate(extent, goal):
        return TaskFeasibilityResult(feasible=False, rejection_reason="invalid_endpoint")
    if world.occupied(start):
        return TaskFeasibilityResult(feasible=False, rejection_reason="occupied_start")
    if world.occupied(goal):
        return TaskFeasibilityResult(feasible=False, rejection_reason="occupied_goal")
    try:
        search = astar_path(
            start,
            goal,
            extent=extent,
            directions=directions,
            occupied=world.occupied,
            maximum_search_nodes=maximum_search_nodes,
        )
    except PlannerBudgetExceeded as exc:
        return TaskFeasibilityResult(
            feasible=False,
            graph_nodes=exc.graph_nodes,
            graph_edges=exc.graph_edges,
            rejection_reason="planner_budget_exhausted",
        )
    if search is None:
        return TaskFeasibilityResult(feasible=False, rejection_reason="no_path")
    path_length = len(search.path) - 1
    difficulty = routing_difficulty(
        search.path,
        start=start,
        goal=goal,
        action_mode=action_mode,
        expansion_count=search.graph_nodes,
        world=world,
        extent=extent,
        clearance_radius=clearance_radius,
    )
    band = next(
        (candidate.name for candidate in difficulty_bands if _band_matches(candidate, difficulty)),
        None,
    )
    if accepted_difficulty_bands and band not in accepted_difficulty_bands:
        return TaskFeasibilityResult(
            feasible=False,
            path=search.path,
            path_length=path_length,
            graph_nodes=search.graph_nodes,
            graph_edges=search.graph_edges,
            rejection_reason="difficulty_band_rejected",
            difficulty=difficulty,
            difficulty_band=band,
        )
    if path_length + recovery_margin_steps > maximum_steps:
        return TaskFeasibilityResult(
            feasible=False,
            path=search.path,
            path_length=path_length,
            graph_nodes=search.graph_nodes,
            graph_edges=search.graph_edges,
            rejection_reason="episode_budget_exceeded",
            difficulty=difficulty,
            difficulty_band=band,
        )
    return TaskFeasibilityResult(
        feasible=True,
        path=search.path,
        path_length=path_length,
        graph_nodes=search.graph_nodes,
        graph_edges=search.graph_edges,
        difficulty=difficulty,
        difficulty_band=band,
    )


def routing_difficulty(
    path: Sequence[Coordinate],
    *,
    start: Coordinate,
    goal: Coordinate,
    action_mode: str,
    expansion_count: int,
    world: OccupancyRead,
    extent: WorldExtent,
    clearance_radius: int | None = None,
) -> RoutingDifficultyDescriptors:
    """Derive descriptors from an existing path without running another search."""

    movements = [
        tuple(after[index] - before[index] for index in range(3))
        for before, after in zip(path, path[1:])
    ]
    turns = sum(left != right for left, right in zip(movements, movements[1:]))
    direct = action_step_distance(start, goal, action_mode)
    length = max(len(path) - 1, 0)
    clearances: list[int] = []
    if clearance_radius is not None:
        for coordinate in path:
            clearances.append(
                _bounded_clearance(world, coordinate, extent, clearance_radius)
            )
    return RoutingDifficultyDescriptors(
        direct_distance=direct,
        shortest_path_length=length,
        detour_ratio=float(length / direct) if direct else 1.0,
        direction_changes=turns,
        vertical_displacement=abs(goal[2] - start[2]),
        expansion_count=expansion_count,
        minimum_clearance=min(clearances) if clearances else None,
        mean_clearance=(sum(clearances) / len(clearances)) if clearances else None,
    )


def _bounded_clearance(
    world: OccupancyRead,
    coordinate: Coordinate,
    extent: WorldExtent,
    maximum_radius: int,
) -> int:
    for radius in range(1, maximum_radius + 1):
        for axis in range(3):
            for sign in (-1, 1):
                candidate = list(coordinate)
                candidate[axis] += sign * radius
                point = tuple(candidate)
                if not contains_task_coordinate(extent, point) or world.occupied(point):
                    return radius - 1
    return maximum_radius


def _band_matches(
    band: RoutingDifficultyBand, descriptors: RoutingDifficultyDescriptors
) -> bool:
    mapping = {
        "path_length": descriptors.shortest_path_length,
        "detour_ratio": descriptors.detour_ratio,
        "direction_changes": descriptors.direction_changes,
        "vertical_displacement": descriptors.vertical_displacement,
        "expansion_count": descriptors.expansion_count,
    }
    for field, value in mapping.items():
        interval = getattr(band, field)
        if interval is None:
            continue
        if interval.minimum is not None and value < interval.minimum:
            return False
        if interval.maximum is not None and value > interval.maximum:
            return False
    return True


__all__ = [
    "BoundedWorldRead",
    "GeometryValidationResult",
    "InMemoryOccupancy",
    "OccupancyRead",
    "TaskFeasibilityResult",
    "RoutingDifficultyDescriptors",
    "routing_difficulty",
    "validate_geometry",
    "validate_task_feasibility",
]
