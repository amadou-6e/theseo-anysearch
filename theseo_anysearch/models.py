from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    algorithm: str
    model: str = "voxel_encoder"
    runner: Literal["local", "anyscale"] = "local"
    iterations: int = 100
    checkpoint_interval: int = 10
    require_gpu: bool = False
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


class Settings(BaseModel):
    # Pydantic v2 reserves 'model_config' as a class variable.
    # The YAML key is 'model_config'; stored as 'model_cfg' with an alias.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    env: EnvConfig
    training: TrainingConfig
    anyscale: AnyscaleConfig
    algorithm_config: AlgorithmConfig
    model_cfg: ModelConfig = Field(alias="model_config")
