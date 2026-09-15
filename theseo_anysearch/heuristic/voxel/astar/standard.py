"""Standard A* search for voxel environments, including compiled worlds."""

from theseo_anysearch.environments.action_spaces import (
    offsets_for_mode,
    shortest_actions,
)
from theseo_anysearch.heuristic.base import BaseVoxelHeuristic
from theseo_anysearch.heuristic.models import VoxelPosition
from theseo_anysearch.worlds.extent import contains_task_coordinate


class VoxelAStarOracle(BaseVoxelHeuristic):
    """Plan an optimal static path using the Chebyshev heuristic."""

    def _find_path(
        self,
        start: VoxelPosition,
        goal: VoxelPosition,
    ) -> list[VoxelPosition]:
        if self.env._config.get("compiled_world_path"):
            direct = self._clear_shortest_path(start, goal)
            if direct is not None:
                self._last_search_nodes = len(direct)
                self._last_search_edges = max(len(direct) - 1, 0)
                return direct
        return self._search_path(start, goal, heuristic_weight=1.0)

    def _clear_shortest_path(
        self,
        start: VoxelPosition,
        goal: VoxelPosition,
    ) -> list[VoxelPosition] | None:
        """Use an optimal direct route only after checking every occupied voxel."""
        mode = self.action_mode
        planning_mode = "discrete_26" if mode == "vector_3" else mode
        offsets = offsets_for_mode(planning_mode)
        path = [start]
        for action in shortest_actions(start, goal, mode):
            if mode == "vector_3":
                offset = tuple(value - 1 for value in action)
            else:
                offset = offsets[action]
            current = path[-1]
            next_position = tuple(current[axis] + offset[axis] for axis in range(3))
            if not contains_task_coordinate(self.extent, next_position):
                return None
            # The active goal is an occupied task overlay, not necessarily a
            # geometry cell; the environment validates it during execution.
            if next_position != goal and self.env._rust_env.world_occupied(next_position):
                return None
            path.append(next_position)
        return path


__all__ = ["VoxelAStarOracle"]
