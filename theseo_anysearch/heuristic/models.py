"""Validated result models shared by heuristic planners."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

VoxelPosition = tuple[int, int, int]
VoxelAction = int | tuple[int, int, int]


class VoxelOraclePlan(BaseModel):
    """A collision-free path predicted from the current environment state."""

    model_config = ConfigDict(frozen=True)

    positions: tuple[VoxelPosition, ...]
    action_indices: tuple[VoxelAction, ...]
    graph_nodes: int
    graph_edges: int

    @property
    def steps(self) -> int:
        """Return the number of actions in the plan."""
        return len(self.action_indices)


class VoxelOracleReplay(BaseModel):
    """Result of executing a heuristic plan in the real environment."""

    model_config = ConfigDict(frozen=True)

    goal_reached: bool
    terminated: bool
    truncated: bool
    steps_executed: int
    positions: tuple[VoxelPosition, ...]
    rewards: tuple[float, ...]
    action_indices: tuple[VoxelAction, ...] = ()
    mismatch: str | None = None


__all__ = ["VoxelAction", "VoxelOraclePlan", "VoxelOracleReplay", "VoxelPosition"]
