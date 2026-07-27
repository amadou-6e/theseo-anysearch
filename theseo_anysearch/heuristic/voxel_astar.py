"""NetworkX search heuristics for the single-agent voxel environment."""

from __future__ import annotations

from typing import Any, Iterable

import networkx as nx
from pydantic import BaseModel, ConfigDict

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv

VoxelPosition = tuple[int, int, int]


class VoxelOraclePlan(BaseModel):
    """A shortest collision-free path predicted from the reset state."""

    model_config = ConfigDict(frozen=True)

    positions: tuple[VoxelPosition, ...]
    action_indices: tuple[int, ...]
    graph_nodes: int
    graph_edges: int

    @property
    def steps(self) -> int:
        return len(self.action_indices)


class VoxelOracleReplay(BaseModel):
    """Result of executing an oracle plan in the real environment."""

    model_config = ConfigDict(frozen=True)

    goal_reached: bool
    terminated: bool
    truncated: bool
    steps_executed: int
    positions: tuple[VoxelPosition, ...]
    rewards: tuple[float, ...]
    action_indices: tuple[int, ...] = ()
    mismatch: str | None = None


class VoxelAStarOracle:
    """Build and execute a 26-neighbor A* plan for a reset ``VoxelEnv``.

    NetworkX owns the explicit static voxel graph and A* implementation. The
    Rust environment remains authoritative for the reset geometry and for plan
    replay. A shortest path is simple, so it never revisits a voxel and remains
    valid when trail mode turns visited cells into obstacles.
    """

    def __init__(self, env: VoxelEnv) -> None:
        self.env = env
        self.grid_size = int(env._config.get("grid_size", 32))
        self.directions = tuple(
            (dx, dy, dz)
            for dx in range(-1, 2)
            for dy in range(-1, 2)
            for dz in range(-1, 2)
            if (dx, dy, dz) != (0, 0, 0)
        )
        self.direction_to_index = {
            direction: index for index, direction in enumerate(self.directions)
        }

    def plan(self) -> VoxelOraclePlan:
        """Return an A* plan from the environment's current reset state."""

        rust_env = self.env._rust_env
        start = self._position(rust_env.cursor_pos())
        raw_goal = rust_env.goal_pos()
        if raw_goal is None:
            raise ValueError("Voxel environment has no goal after reset")
        goal = self._position(raw_goal)

        blocked = {self._position(cell) for cell in rust_env.filled_voxels()}
        blocked.discard(start)
        blocked.discard(goal)
        graph = self._build_graph(blocked)
        if start not in graph or goal not in graph:
            raise nx.NodeNotFound(f"Start {start} or goal {goal} is outside the free voxel graph")

        positions = tuple(self._find_path(graph, start, goal))
        actions = tuple(
            self.direction_to_index[self._subtract(after, before)]
            for before, after in zip(positions, positions[1:])
        )
        return VoxelOraclePlan(
            positions=positions,
            action_indices=actions,
            graph_nodes=graph.number_of_nodes(),
            graph_edges=graph.number_of_edges(),
        )

    def replay(self, plan: VoxelOraclePlan) -> VoxelOracleReplay:
        """Execute ``plan`` and detect any planner/environment disagreement."""

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
        action_indices: list[int] = []
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
            goal_reached = bool(
                info.get("goal_reached", actual == plan.positions[-1])
            )
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

    def _find_path(
        self,
        graph: nx.Graph,
        start: VoxelPosition,
        goal: VoxelPosition,
    ) -> list[VoxelPosition]:
        return nx.astar_path(
            graph,
            start,
            goal,
            heuristic=self._chebyshev,
            weight="weight",
        )

    def _build_graph(self, blocked: set[VoxelPosition]) -> nx.Graph:
        graph = nx.Graph()
        size = self.grid_size
        free = {
            (x, y, z)
            for x in range(1, size + 1)
            for y in range(1, size + 1)
            for z in range(1, size + 1)
            if (x, y, z) not in blocked
        }
        graph.add_nodes_from(free)
        forward_directions = tuple(direction for direction in self.directions if direction > (0, 0, 0))
        for position in free:
            x, y, z = position
            for dx, dy, dz in forward_directions:
                neighbor = (x + dx, y + dy, z + dz)
                if neighbor in free:
                    graph.add_edge(position, neighbor, weight=1.0)
        return graph

    @staticmethod
    def _chebyshev(left: VoxelPosition, right: VoxelPosition) -> float:
        return float(max(abs(a - b) for a, b in zip(left, right)))

    @staticmethod
    def _subtract(left: VoxelPosition, right: VoxelPosition) -> VoxelPosition:
        return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]

    @staticmethod
    def _position(values: Iterable[Any]) -> VoxelPosition:
        x, y, z = values
        return int(x), int(y), int(z)

class VoxelDijkstraHeuristic(VoxelAStarOracle):
    """Use Dijkstra's zero-heuristic search as an optimality baseline."""

    def _find_path(
        self,
        graph: nx.Graph,
        start: VoxelPosition,
        goal: VoxelPosition,
    ) -> list[VoxelPosition]:
        return nx.dijkstra_path(graph, start, goal, weight="weight")


class VoxelWeightedAStarHeuristic(VoxelAStarOracle):
    """Trade shortest-path guarantees for greedier search with ``weight > 1``."""

    def __init__(self, env: VoxelEnv, heuristic_weight: float = 1.5) -> None:
        super().__init__(env)
        if heuristic_weight <= 0.0:
            raise ValueError("heuristic_weight must be greater than zero")
        self.heuristic_weight = float(heuristic_weight)

    def _find_path(
        self,
        graph: nx.Graph,
        start: VoxelPosition,
        goal: VoxelPosition,
    ) -> list[VoxelPosition]:
        def weighted_heuristic(left: VoxelPosition, right: VoxelPosition) -> float:
            return self.heuristic_weight * self._chebyshev(left, right)

        return nx.astar_path(
            graph,
            start,
            goal,
            heuristic=weighted_heuristic,
            weight="weight",
        )


class VoxelReplanningAStarHeuristic(VoxelAStarOracle):
    """Recompute A* from the current environment state before every action."""

    def replay(self, plan: VoxelOraclePlan | None = None) -> VoxelOracleReplay:
        """Replan and execute one action at a time until the episode ends."""

        del plan
        rust_env = self.env._rust_env
        raw_goal = rust_env.goal_pos()
        if raw_goal is None:
            raise ValueError("Voxel environment has no goal after reset")
        goal = self._position(raw_goal)
        actual_positions = [self._position(rust_env.cursor_pos())]
        rewards: list[float] = []
        action_indices: list[int] = []
        terminated = False
        truncated = False
        goal_reached = actual_positions[0] == goal

        while not goal_reached and not terminated and not truncated:
            current_plan = self.plan()
            if not current_plan.action_indices:
                break
            action_index = current_plan.action_indices[0]
            action_indices.append(action_index)
            _, reward, terminated, truncated, info = self.env.step(action_index)
            rewards.append(float(reward))
            actual = self._position(rust_env.cursor_pos())
            actual_positions.append(actual)
            expected = current_plan.positions[1]
            if actual != expected:
                return VoxelOracleReplay(
                    goal_reached=False,
                    terminated=terminated,
                    truncated=truncated,
                    steps_executed=len(rewards),
                    positions=tuple(actual_positions),
                    rewards=tuple(rewards),
                    action_indices=tuple(action_indices),
                    mismatch=(
                        f"Step {len(rewards)}: expected {expected}, got {actual}"
                    ),
                )
            goal_reached = bool(info.get("goal_reached", actual == goal))

        mismatch = None
        if not goal_reached:
            mismatch = "Replanning ended without reaching the goal"
        return VoxelOracleReplay(
            goal_reached=goal_reached,
            terminated=terminated,
            truncated=truncated,
            steps_executed=len(rewards),
            positions=tuple(actual_positions),
            rewards=tuple(rewards),
            action_indices=tuple(action_indices),
            mismatch=mismatch,
        )

def build_voxel_heuristic(
    env: VoxelEnv,
    heuristic_type: str,
    *,
    weight: float | None = None,
) -> VoxelAStarOracle:
    """Build a configured voxel heuristic for reference evaluation."""

    if heuristic_type == "astar":
        return VoxelAStarOracle(env)
    if heuristic_type == "dijkstra":
        return VoxelDijkstraHeuristic(env)
    if heuristic_type == "weighted_astar":
        return VoxelWeightedAStarHeuristic(
            env,
            heuristic_weight=1.5 if weight is None else weight,
        )
    if heuristic_type == "replanning_astar":
        return VoxelReplanningAStarHeuristic(env)
    raise ValueError(f"Unknown voxel heuristic type: {heuristic_type!r}")
