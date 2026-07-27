"""Core Pydantic models shared across training, environments, and config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EnvConfig(BaseModel):
    """Environment configuration shared by training and experiment loading.

    Parameters
    ----------
    stl_path : Path | None
        Optional STL file to voxelize into the environment geometry.
    scale : float
        Fixed STL voxelization scale when ``scale_range`` is not used.
    agent_count : int
        Number of agents requested by the environment configuration.
    max_steps : int
        Maximum number of environment steps per episode.
    seed : int
        Base random seed for resets and procedural choices.
    obs_mode : {"scalar", "box", "radial", "hierarchical_box"}
        Observation encoding exposed to the policy.
    box_radius : int
        Radius used for single-scale local voxel box observations.
    box_radii : list[int] | None
        Radii used for hierarchical box observations.
    ray_max_len : int
        Maximum ray length for radial observations.
    grid_size : int
        Side length of the cubic voxel grid.
    trail_mode : bool
        Whether movement automatically fills visited cells.
    geometry_boxes : list[list[int]] | None
        Procedural box geometry definitions.
    waypoints_file : str | None
        Optional JSON file with fixed start and goal waypoints.
    step_cost : float
        Per-step reward penalty.
    collision_cost : float
        Additional reward penalty on blocked moves.
    goal_reward : float
        Terminal reward awarded when the goal is reached.
    distance_shaping : float
        Potential-based shaping coefficient toward the goal.
    distance_reward_mode : {"progress", "zone"}
        Strategy used for distance-based per-step rewards.
    zone_reward_min : float
        Most negative per-step zone reward when far from the goal.
    zone_reward_max : float
        Least negative per-step zone reward when near the goal.
    zone_reward_curve : {"linear", "exponential"}
        Curve used to interpolate between zone reward values.
    distance_metric : {"euclidean", "manhattan"}
        Distance metric used for shaping.
    stl_paths : list[Path] | None
        Optional set of STL files used for map diversity.
    scale_range : list[float] | None
        Minimum and maximum voxelization scale for STL diversity.
    geometry_pool_size : int
        Number of procedural geometries to pre-generate.
    scale_variants_per_map : int
        Number of STL re-voxelizations generated per map.
    geometry_padding : int
        Free-space padding around imported geometry.
    geometry_pool : dict | None
        Precomputed geometry pool configuration produced by extraction tools.
    """
    include_voxel_count: bool = True
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
    distance_reward_mode: Literal["progress", "zone"] = "progress"
    zone_reward_min: float = -1.0       # farthest-from-goal reward in zone mode
    zone_reward_max: float = -0.01      # nearest-to-goal reward in zone mode
    zone_reward_curve: Literal["linear", "exponential"] = "linear"
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

    @model_validator(mode="after")
    def validate_zone_rewards(self) -> "EnvConfig":
        """Ensure zone reward configuration remains negative and ordered."""

        if self.zone_reward_min > self.zone_reward_max:
            raise ValueError("zone_reward_min must be less than or equal to zone_reward_max")
        if self.zone_reward_max >= 0.0:
            raise ValueError("zone_reward_max must stay negative")
        if self.zone_reward_min >= 0.0:
            raise ValueError("zone_reward_min must stay negative")
        return self



class TrainingConfig(BaseModel):
    """Training configuration for RLlib runs.

    Parameters
    ----------
    algorithm : str
        Registered trainer algorithm name.
    model : str
        Registered model configuration family name.
    runner : {"local", "anyscale"}
        Execution backend used for training.
    iterations : int
        Number of training iterations to run.
    checkpoint_interval : int
        Iteration interval for checkpoint creation.
    require_gpu : bool
        Whether training must fail if no GPU is available.
    num_gpus : float | None
        Explicit RLlib GPU allocation override.
    num_env_runners : int
        Number of rollout workers or env runners.
    trajectory_every : int
        Iteration interval for periodic trajectory snapshots.
    best_trajectory : bool
        Whether to keep a best-so-far evaluation trajectory.
    output_dir : Path
        Base output directory for runtime artifacts.
    video_every : int
        Iteration interval for rendered videos.
    """
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
    evaluation_episodes: int = Field(default=1, ge=1)
    evaluation_seed: int = 42
    evaluation_min_success_rate: float = Field(default=0.5, ge=0.0, le=1.0)


class AnyscaleConfig(BaseModel):
    """Anyscale execution configuration.

    Parameters
    ----------
    cluster_env : str
        Cluster environment name.
    compute_config : str
        Compute configuration name.
    project : str
        Anyscale project identifier.
    """
    model_config = ConfigDict(extra="forbid")

    cluster_env: str
    compute_config: str
    project: str


class AlgorithmConfig(BaseModel):
    """Base algorithm hyperparameters shared by multiple trainers.

    Parameters
    ----------
    lr : float
        Learning rate.
    gamma : float
        Discount factor.
    train_batch_size : int
        Training batch size passed into RLlib.
    """
    model_config = ConfigDict(extra="forbid")

    lr: float = 3e-4
    gamma: float = 0.99
    train_batch_size: int = 4096


class ModelConfig(BaseModel):
    """Base model configuration shared by trainer integrations.

    Parameters
    ----------
    hidden_sizes : list[int]
        Hidden layer sizes for the default fully connected policy.
    activation : {"relu", "tanh", "elu"}
        Hidden activation used by the default policy.
    custom_model : str | None
        Registered RLlib custom model name.
    custom_model_config : dict[str, Any] | None
        Extra configuration passed to the custom model.
    """
    model_config = ConfigDict(extra="forbid")

    hidden_sizes: list[int] = [256, 256]
    activation: Literal["relu", "tanh", "elu"] = "relu"
    custom_model: str | None = None
    custom_model_config: dict[str, Any] | None = None


class AlgorithmEnvCompatibilityMixin:
    """Validate algorithm and environment combinations after model creation."""

    @model_validator(mode="after")
    def _validate_algorithm_env_compatibility(self):
        """Reject unsupported algorithm and agent-count combinations."""
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
    """Validated runtime settings consumed by trainers and runners.

    Parameters
    ----------
    env : EnvConfig
        Environment configuration.
    training : TrainingConfig
        Training execution configuration.
    anyscale : AnyscaleConfig
        Anyscale execution configuration.
    algorithm_config : AlgorithmConfig
        Algorithm-specific hyperparameter block.
    model_cfg : ModelConfig
        Model configuration block loaded from ``model_config`` in YAML.
    """
    # Pydantic v2 reserves 'model_config' as a class variable.
    # The YAML key is 'model_config'; stored as 'model_cfg' with an alias.
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    env: EnvConfig
    training: TrainingConfig
    anyscale: AnyscaleConfig
    algorithm_config: AlgorithmConfig
    model_cfg: ModelConfig = Field(alias="model_config")
