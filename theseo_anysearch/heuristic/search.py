"""Backend-neutral deterministic voxel A* search."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import heapq
import itertools
import math

from theseo_anysearch.worlds.extent import WorldExtent, contains_task_coordinate

Position = tuple[int, int, int]


class PlannerBudgetExceeded(RuntimeError):
    """Raised with partial counters when a search exceeds its node budget."""

    def __init__(self, maximum_nodes: int, graph_nodes: int, graph_edges: int) -> None:
        super().__init__(f"A* exceeded its {maximum_nodes}-node search budget")
        self.graph_nodes = graph_nodes
        self.graph_edges = graph_edges


@dataclass(frozen=True)
class SearchResult:
    path: tuple[Position, ...]
    graph_nodes: int
    graph_edges: int


def astar_path(
    start: Position,
    goal: Position,
    *,
    extent: WorldExtent,
    directions: Sequence[Position],
    occupied: Callable[[Position], bool],
    maximum_search_nodes: int | None = None,
    heuristic_weight: float = 1.0,
) -> SearchResult | None:
    """Search via deterministic direction and heap insertion ordering."""

    frontier: list[tuple[float, int, Position]] = [(0.0, 0, start)]
    serial = itertools.count(1)
    cost = {start: 0.0}
    parent: dict[Position, Position] = {}
    closed: set[Position] = set()
    edges = 0
    while frontier:
        _, _, current = heapq.heappop(frontier)
        if current in closed:
            continue
        closed.add(current)
        if maximum_search_nodes is not None and len(closed) > maximum_search_nodes:
            raise PlannerBudgetExceeded(maximum_search_nodes, len(closed), edges)
        if current == goal:
            break
        for dx, dy, dz in directions:
            neighbor = (current[0] + dx, current[1] + dy, current[2] + dz)
            if not contains_task_coordinate(extent, neighbor):
                continue
            edges += 1
            if neighbor != goal and occupied(neighbor):
                continue
            next_cost = cost[current] + 1.0
            if next_cost >= cost.get(neighbor, math.inf):
                continue
            cost[neighbor] = next_cost
            parent[neighbor] = current
            heuristic = max(abs(a - b) for a, b in zip(neighbor, goal))
            priority = next_cost + heuristic_weight * float(heuristic)
            heapq.heappush(frontier, (priority, next(serial), neighbor))
    if goal not in closed:
        return None
    path = [goal]
    while path[-1] != start:
        path.append(parent[path[-1]])
    path.reverse()
    return SearchResult(tuple(path), len(closed), edges)


__all__ = ["PlannerBudgetExceeded", "SearchResult", "astar_path"]
