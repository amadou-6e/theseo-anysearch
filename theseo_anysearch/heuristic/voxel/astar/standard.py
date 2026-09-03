"""Standard A* search for the 26-neighbor voxel environment."""

from theseo_anysearch.heuristic.base import BaseVoxelHeuristic
from theseo_anysearch.heuristic.models import VoxelPosition


class VoxelAStarOracle(BaseVoxelHeuristic):
    """Plan an optimal static path using the Chebyshev heuristic."""

    def _find_path(
        self,
        start: VoxelPosition,
        goal: VoxelPosition,
    ) -> list[VoxelPosition]:
        return self._search_path(start, goal, heuristic_weight=1.0)


__all__ = ["VoxelAStarOracle"]
