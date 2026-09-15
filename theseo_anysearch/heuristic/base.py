"""Shared graph planning and replay mechanics for voxel heuristics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import networkx as nx

from theseo_anysearch.environments.action_spaces import ACTION_OFFSETS_26, offsets_for_mode
from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.heuristic.models import (
    VoxelOraclePlan,
    VoxelOracleReplay,
    VoxelPosition,
)
from theseo_anysearch.heuristic.search import PlannerBudgetExceeded, astar_path
from theseo_anysearch.worlds.extent import contains_task_coordinate, resolve_extent


class PlannerBudgetExceeded(RuntimeError):
    """Raised when a bounded heuristic search exceeds its node budget."""


class BaseVoxelHeuristic(ABC):
    """Provide environment translation, graph construction, and plan replay."""

    def __init__(self, env: VoxelEnv, *, maximum_search_nodes: int | None = None) -> None:
        self.env = env
        self.extent = resolve_extent(env._config)
        self.grid_size = max(self.extent)
        self.action_mode = str(env._config.get("action_mode", "discrete_26"))
        self.directions = (
            ACTION_OFFSETS_26
            if self.action_mode == "vector_3"
            else offsets_for_mode(self.action_mode)
        )
        self.direction_to_index = {
            direction: (
                tuple(value + 1 for value in direction)
                if self.action_mode == "vector_3"
                else index
            )
            for index, direction in enumerate(self.directions)
        }
        self._last_search_nodes = 0
        self._last_search_edges = 0
        self.maximum_search_nodes = maximum_search_nodes

    def plan(self) -> VoxelOraclePlan:
        """Build a plan from the environment's current state."""
        rust_env = self.env._rust_env
        start = self._position(rust_env.cursor_pos())
        raw_goal = rust_env.goal_pos()
        if raw_goal is None:
            raise ValueError("Voxel environment has no goal after reset")
        goal = self._position(raw_goal)

        if not contains_task_coordinate(self.extent, start) or not contains_task_coordinate(
            self.extent, goal
        ):
            raise nx.NodeNotFound(
                f"Start {start} or goal {goal} is outside the free voxel graph"
            )
        positions = tuple(self._find_path(start, goal))
        actions = tuple(
            self.direction_to_index[self._subtract(after, before)]
            for before, after in zip(positions, positions[1:])
        )
        return VoxelOraclePlan(
            positions=positions,
            action_indices=actions,
            graph_nodes=self._last_search_nodes,
            graph_edges=self._last_search_edges,
        )

    def replay(self, plan: VoxelOraclePlan) -> VoxelOracleReplay:
        """Execute a plan and report planner/environment disagreement."""
        actual_positions = [self._position(self.env._rust_env.cursor_pos())]
        if not plan.positions or actual_positions[0] != plan.positions[0]:
            return VoxelOracleReplay(
                goal_reached=False,
                terminated=False,
                truncated=False,
                steps_executed=0,
                positions=tuple(actual_positions),
                rewards=(),
                mismatch="Environment is not at the plan's start position",
            )

        rewards: list[float] = []
        terminated = False
        truncated = False
        goal_reached = False
        for step_index, action_index in enumerate(plan.action_indices, start=1):
            _, reward, terminated, truncated, info = self.env.step(action_index)
            rewards.append(float(reward))
            actual = self._position(self.env._rust_env.cursor_pos())
            actual_positions.append(actual)
            expected = plan.positions[step_index]
            if actual != expected:
                return VoxelOracleReplay(
                    goal_reached=False,
                    terminated=terminated,
                    truncated=truncated,
                    steps_executed=step_index,
                    positions=tuple(actual_positions),
                    rewards=tuple(rewards),
                    action_indices=plan.action_indices[:step_index],
                    mismatch=f"Step {step_index}: expected {expected}, got {actual}",
                )
            goal_reached = bool(info.get("goal_reached", actual == plan.positions[-1]))
            if terminated or truncated:
                break

        mismatch = None
        if not goal_reached:
            mismatch = "Plan ended without the environment reporting goal_reached"
        return VoxelOracleReplay(
            goal_reached=goal_reached,
            terminated=terminated,
            truncated=truncated,
            steps_executed=len(rewards),
            positions=tuple(actual_positions),
            rewards=tuple(rewards),
            action_indices=plan.action_indices[:len(rewards)],
            mismatch=mismatch,
        )

    @abstractmethod
    def _find_path(
        self,
        start: VoxelPosition,
        goal: VoxelPosition,
    ) -> list[VoxelPosition]:
        """Return a path through ``graph`` from ``start`` to ``goal``."""

    def _search_path(
        self,
        start: VoxelPosition,
        goal: VoxelPosition,
        *,
        heuristic_weight: float,
    ) -> list[VoxelPosition]:
        """Search the regional world lazily without materializing its volume."""

        try:
            result = astar_path(
                start,
                goal,
                extent=self.extent,
                directions=self.directions,
                occupied=self.env._rust_env.world_occupied,
                maximum_search_nodes=self.maximum_search_nodes,
                heuristic_weight=heuristic_weight,
            )
        except PlannerBudgetExceeded as exc:
            self._last_search_nodes = exc.graph_nodes
            self._last_search_edges = exc.graph_edges
            raise
        if result is None:
            raise nx.NetworkXNoPath(f"No path between {start} and {goal}")
        self._last_search_nodes = result.graph_nodes
        self._last_search_edges = result.graph_edges
        return list(result.path)

    @staticmethod
    def _chebyshev(left: VoxelPosition, right: VoxelPosition) -> float:
        """Return Chebyshev distance for 26-neighbor movement."""
        return float(max(abs(a - b) for a, b in zip(left, right)))

    @staticmethod
    def _subtract(left: VoxelPosition, right: VoxelPosition) -> VoxelPosition:
        """Return the component-wise displacement from ``right`` to ``left``."""
        return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]

    @staticmethod
    def _position(values: Iterable[Any]) -> VoxelPosition:
        """Normalize an iterable position to integer voxel coordinates."""
        x, y, z = values
        return int(x), int(y), int(z)


__all__ = ["BaseVoxelHeuristic", "PlannerBudgetExceeded"]
