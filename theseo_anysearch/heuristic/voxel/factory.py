"""Configuration-name dispatch for voxel heuristic strategies."""

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.heuristic.base import BaseVoxelHeuristic
from theseo_anysearch.heuristic.voxel.astar import VoxelAStarOracle
from theseo_anysearch.heuristic.voxel.dijkstra import VoxelDijkstraHeuristic
from theseo_anysearch.heuristic.voxel.replanning_astar import VoxelReplanningAStarHeuristic
from theseo_anysearch.heuristic.voxel.weighted_astar import VoxelWeightedAStarHeuristic


def build_voxel_heuristic(
    env: VoxelEnv,
    heuristic_type: str,
    *,
    weight: float | None = None,
) -> BaseVoxelHeuristic:
    """Build the voxel heuristic selected by its stable YAML name."""
    factories = {
        "astar": lambda: VoxelAStarOracle(env),
        "dijkstra": lambda: VoxelDijkstraHeuristic(env),
        "weighted_astar": lambda: VoxelWeightedAStarHeuristic(
            env,
            heuristic_weight=1.5 if weight is None else weight,
        ),
        "replanning_astar": lambda: VoxelReplanningAStarHeuristic(env),
    }
    try:
        return factories[heuristic_type]()
    except KeyError as exc:
        raise ValueError(f"Unknown voxel heuristic type: {heuristic_type!r}") from exc


__all__ = ["build_voxel_heuristic"]
