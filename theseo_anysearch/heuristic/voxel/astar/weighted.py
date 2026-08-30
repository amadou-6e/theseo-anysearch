"""Weighted A* search for faster, potentially suboptimal voxel planning."""

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.heuristic.base import BaseVoxelHeuristic
from theseo_anysearch.heuristic.models import VoxelPosition


class VoxelWeightedAStarHeuristic(BaseVoxelHeuristic):
    """Trade shortest-path guarantees for greedier search with weight > 1."""

    def __init__(self, env: VoxelEnv, heuristic_weight: float = 1.5) -> None:
        super().__init__(env)
        if heuristic_weight <= 0.0:
            raise ValueError("heuristic_weight must be greater than zero")
        self.heuristic_weight = float(heuristic_weight)

    def _find_path(
        self,
        start: VoxelPosition,
        goal: VoxelPosition,
    ) -> list[VoxelPosition]:
        return self._search_path(
            start,
            goal,
            heuristic_weight=self.heuristic_weight,
        )


__all__ = ["VoxelWeightedAStarHeuristic"]
