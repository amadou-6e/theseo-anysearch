"""Dijkstra search baseline for the voxel environment."""

from theseo_anysearch.heuristic.base import BaseVoxelHeuristic
from theseo_anysearch.heuristic.models import VoxelPosition


class VoxelDijkstraHeuristic(BaseVoxelHeuristic):
    """Use Dijkstra search as an optimality baseline."""

    def _find_path(
        self,
        start: VoxelPosition,
        goal: VoxelPosition,
    ) -> list[VoxelPosition]:
        return self._search_path(start, goal, heuristic_weight=0.0)


__all__ = ["VoxelDijkstraHeuristic"]
