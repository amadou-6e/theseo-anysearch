"""Reusable structural geometry and task-feasibility validation contracts."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from theseo_anysearch.heuristic.search import PlannerBudgetExceeded, astar_path
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
    maximum_search_nodes: int,
    maximum_steps: int,
    recovery_margin_steps: int = 0,
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
    if path_length + recovery_margin_steps > maximum_steps:
        return TaskFeasibilityResult(
            feasible=False,
            path=search.path,
            path_length=path_length,
            graph_nodes=search.graph_nodes,
            graph_edges=search.graph_edges,
            rejection_reason="episode_budget_exceeded",
        )
    return TaskFeasibilityResult(
        feasible=True,
        path=search.path,
        path_length=path_length,
        graph_nodes=search.graph_nodes,
        graph_edges=search.graph_edges,
    )


__all__ = [
    "BoundedWorldRead",
    "GeometryValidationResult",
    "InMemoryOccupancy",
    "OccupancyRead",
    "TaskFeasibilityResult",
    "validate_geometry",
    "validate_task_feasibility",
]
