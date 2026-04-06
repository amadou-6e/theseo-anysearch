from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stl_path: Path | None = None
    scale: float = 1.0
    agent_count: int = 4
    max_steps: int = 200
    seed: int = 42
    obs_mode: Literal["scalar", "box", "radial", "hierarchical_box"] = "scalar"
    box_radius: int = 2
    box_radii: list[int] | None = None   # hierarchical_box mode: list of radii to concatenate
    ray_max_len: int = 16
    grid_size: int = 32                  # side length of the cubic grid (coords in [1, grid_size]³)
    trail_mode: bool = True              # movement auto-fills visited cells (one-way)
    geometry_boxes: list[list[int]] | None = None  # [[xmin,ymin,zmin,xmax,ymax,zmax], ...]
    # Navigation / reward (modifiable from Python, computed in Rust)
    waypoints_file: str | None = None   # path to JSON {"start":[x,y,z],"goal":[x,y,z]}
    step_cost: float = -0.01            # per-step reward penalty
    collision_cost: float = 0.0         # extra penalty subtracted on blocked moves
    goal_reward: float = 1.0            # bonus when cursor reaches goal position
    distance_shaping: float = 0.0       # potential-based shaping coefficient toward goal
    distance_metric: Literal["euclidean", "manhattan"] = "euclidean"

    # --- Training diversity ---
    # Geometry pool: pre-load N geometries at init; at each reset pick one randomly.
    # stl_paths + scale_range: voxelise each STL at M random scales → large pool.
    # geometry_pool_size alone: procedural random-box geometries.
    stl_paths: list[Path] | None = None           # multiple STL maps to cycle through
    scale_range: list[float] | None = None        # [min, max] voxelisation scale for stl_paths
    geometry_pool_size: int = 0                   # >0: use this many random-box geometries
    scale_variants_per_map: int = 4               # STL re-voxelisations per scale sweep
    geometry_padding: int = 2                     # free voxels on each side of the geometry (circumnavigation margin)

    # --- Geometry pool (pre-computed .npy files built by `anysearch extract`) ---
    # When set, each episode loads a random .npy file from pool_dir instead of
    # re-voxelizing at runtime. stl_path / scale_range still work independently.
    geometry_pool: dict | None = None             # {pool_dir, augmentation: {paste_boxes: {...}}}



class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: str
    model: str = "voxel_encoder"
    runner: Literal["local", "anyscale"] = "local"
    iterations: int = 100
    checkpoint_interval: int = 10
    require_gpu: bool = False
    num_gpus: float | None = None  # override _detect_num_gpus (e.g. 0.5 for two concurrent Tune trials)
    num_env_runners: int = 0       # CPU rollout workers (0 = inline; >0 = parallel actors)
    trajectory_every: int = 10
    best_trajectory: bool = True
    output_dir: Path = Path("runtime/")
    video_every: int = 10


class AnyscaleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cluster_env: str
    compute_config: str
    project: str


class AlgorithmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lr: float = 3e-4
    gamma: float = 0.99
    train_batch_size: int = 4096


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hidden_sizes: list[int] = [256, 256]
    activation: Literal["relu", "tanh", "elu"] = "relu"
    custom_model: str | None = None
    custom_model_config: dict[str, Any] | None = None


class AlgorithmEnvCompatibilityMixin:
    @model_validator(mode="after")
    def _validate_algorithm_env_compatibility(self):
        single_agent_algorithms = {"ppo", "dqn", "sac", "rainbow"}
        algorithm = self.training.algorithm.lower()
        if algorithm in single_agent_algorithms and self.env.agent_count != 1:
            raise ValueError(
                f"training.algorithm='{self.training.algorithm}' only supports single-agent "
                f"VoxelEnv, but env.agent_count={self.env.agent_count}. "
                "Use env.agent_count=1 or switch to training.algorithm='multi_agent_voxel_ppo'."
            )
        return self


class Settings(AlgorithmEnvCompatibilityMixin, BaseModel):
    # Pydantic v2 reserves 'model_config' as a class variable.
    # The YAML key is 'model_config'; stored as 'model_cfg' with an alias.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    env: EnvConfig
    training: TrainingConfig
    anyscale: AnyscaleConfig
    algorithm_config: AlgorithmConfig
    model_cfg: ModelConfig = Field(alias="model_config")
