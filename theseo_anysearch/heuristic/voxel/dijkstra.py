"""Dijkstra search baseline for the voxel environment."""

import networkx as nx

from theseo_anysearch.heuristic.base import BaseVoxelHeuristic
from theseo_anysearch.heuristic.models import VoxelPosition


class VoxelDijkstraHeuristic(BaseVoxelHeuristic):
    """Use Dijkstra search as an optimality baseline."""

    def _find_path(
        self,
        graph: nx.Graph,
        start: VoxelPosition,
        goal: VoxelPosition,
    ) -> list[VoxelPosition]:
        return nx.dijkstra_path(graph, start, goal, weight="weight")


__all__ = ["VoxelDijkstraHeuristic"]
