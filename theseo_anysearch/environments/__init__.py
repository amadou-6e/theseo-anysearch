"""Environment package for voxel and surface navigation tasks."""

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.environments.pettingzoo.surface_env import SurfaceEnv
from theseo_anysearch.environments.pettingzoo.multi_voxel_env import MultiVoxelEnv

try:
    from ray.tune.registry import register_env

    register_env("voxel_env", lambda cfg: VoxelEnv(cfg))
    register_env("surface_env", lambda cfg: SurfaceEnv(cfg))
    register_env("multi_voxel_env", lambda cfg: MultiVoxelEnv(cfg))
except ImportError:
    pass

__all__ = ["VoxelEnv", "SurfaceEnv", "MultiVoxelEnv"]
