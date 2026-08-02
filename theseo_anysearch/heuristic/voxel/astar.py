"""A* search for the 26-neighbor voxel environment."""

import networkx as nx

from theseo_anysearch.heuristic.base import BaseVoxelHeuristic
from theseo_anysearch.heuristic.models import VoxelPosition


class VoxelAStarOracle(BaseVoxelHeuristic):
    """Plan an optimal static path using the Chebyshev heuristic."""

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


__all__ = ["VoxelAStarOracle"]
