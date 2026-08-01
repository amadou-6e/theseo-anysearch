"""Core Pydantic models shared across training, environments, and config loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from theseo_anysearch.environments.task import TaskConfig


class GeometryConfig(BaseModel):
    """Geometry source and voxelization settings."""

    model_config = ConfigDict(extra="forbid")
    stl_path: Path | None = None
    stl_paths: list[Path] | None = None
    scale: float = 1.0
    scale_range: list[float] | None = None
    grid_size: int = Field(default=32, ge=1)
    boxes: list[list[int]] | None = None
    pool_size: int = Field(default=0, ge=0)
    scale_variants_per_map: int = Field(default=4, ge=1)
    padding: int = Field(default=2, ge=0)
    pool: dict[str, Any] | None = None


class ObservationConfig(BaseModel):
    """Policy observation representation."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["scalar", "box", "radial", "hierarchical_box"] = "scalar"
    box_radius: int = Field(default=2, ge=0)
    box_radii: list[int] | None = None
    ray_max_len: int = Field(default=16, ge=1)
    include_voxel_count: bool = True


class ActionExtensionSelector(BaseModel):
    """Named built-in or compiled Rust action extension."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class ActionConfig(BaseModel):
    """Policy action-space and Rust behavior pipeline."""

    model_config = ConfigDict(extra="forbid")
    mode: Literal["discrete_6", "discrete_18", "discrete_26", "vector_3"] = "discrete_26"
    behavior: Literal["legacy", "cursor_navigation", "trail_navigation"] = "legacy"
    predicates: tuple[ActionExtensionSelector, ...] | None = None
    outcomes: tuple[ActionExtensionSelector, ...] | None = None
    history_length: int = Field(default=16, ge=0, le=4096)

    @field_validator("predicates", "outcomes", mode="before")
    @classmethod
    def expand_selector_shorthand(cls, value: Any) -> Any:
        if value is None:
            return None
        return [({"name": item} if isinstance(item, str) else item) for item in value]

    @model_validator(mode="after")
    def validate_unique_pipeline_names(self) -> "ActionConfig":
        for kind, selectors in (
            ("predicate", self.predicates),
            ("outcome", self.outcomes),
        ):
            if selectors is None:
                continue
            names = [selector.name for selector in selectors]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(f"duplicate action {kind} names: {duplicates}")
        return self

    def resolved_pipeline(
        self, *, trail_mode: bool
    ) -> tuple[tuple[ActionExtensionSelector, ...], tuple[ActionExtensionSelector, ...]]:
        behavior = self.behavior
        if behavior == "legacy":
            behavior = "trail_navigation" if trail_mode else "cursor_navigation"
        predicates = self.predicates if self.predicates is not None else tuple(
            ActionExtensionSelector(name=name)
            for name in ("valid_action", "bounds", "unoccupied")
        )
        default_outcomes = (
            ("cursor_movement", "trail_placement")
            if behavior == "trail_navigation"
            else ("cursor_movement",)
        )
        outcomes = self.outcomes if self.outcomes is not None else tuple(
            ActionExtensionSelector(name=name) for name in default_outcomes
        )
        return predicates, outcomes


class CustomRewardConfig(BaseModel):
    """Named custom reward and its user-defined JSON parameters."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class RewardConfig(BaseModel):
    """Reward terms computed by the voxel environment."""

    model_config = ConfigDict(extra="forbid")
    step_cost: float = -0.01
    collision_cost: float = 0.0
    goal_reward: float = 1.0
    distance_shaping: float = 0.0
    distance_reward_mode: Literal["progress", "zone"] = "progress"
    zone_reward_min: float = -1.0
    zone_reward_max: float = -0.01
    zone_reward_curve: Literal["linear", "exponential"] = "linear"
    distance_metric: Literal["euclidean", "manhattan"] = "euclidean"
    invalid_action_cost: float = 0.0
    construction_residual_weight: float = Field(default=0.0, ge=0.0)
    construction_overshoot_weight: float = Field(default=0.0, ge=0.0)
    custom: CustomRewardConfig | None = None

    @field_validator("custom", mode="before")
    @classmethod
    def expand_custom_reward_shorthand(cls, value: Any) -> Any:
        if isinstance(value, str):
            return {"name": value}
        return value

    @model_validator(mode="after")
    def validate_zone_rewards(self) -> "RewardConfig":
        if self.zone_reward_min > self.zone_reward_max:
            raise ValueError("zone_reward_min must be less than or equal to zone_reward_max")
        if self.zone_reward_max >= 0.0:
            raise ValueError("zone_reward_max must stay negative")
        if self.zone_reward_min >= 0.0:
            raise ValueError("zone_reward_min must stay negative")
        return self


_LEGACY_ENV_FIELDS: dict[str, tuple[str, str]] = {
    "stl_path": ("geometry", "stl_path"),
    "stl_paths": ("geometry", "stl_paths"),
    "scale": ("geometry", "scale"),
    "scale_range": ("geometry", "scale_range"),
    "grid_size": ("geometry", "grid_size"),
    "geometry_boxes": ("geometry", "boxes"),
    "geometry_pool_size": ("geometry", "pool_size"),
    "scale_variants_per_map": ("geometry", "scale_variants_per_map"),
    "geometry_padding": ("geometry", "padding"),
    "geometry_pool": ("geometry", "pool"),
    "obs_mode": ("observation", "mode"),
    "box_radius": ("observation", "box_radius"),
    "box_radii": ("observation", "box_radii"),
    "ray_max_len": ("observation", "ray_max_len"),
    "include_voxel_count": ("observation", "include_voxel_count"),
    "action_mode": ("action", "mode"),
    "step_cost": ("rewards", "step_cost"),
    "collision_cost": ("rewards", "collision_cost"),
    "goal_reward": ("rewards", "goal_reward"),
    "distance_shaping": ("rewards", "distance_shaping"),
    "distance_reward_mode": ("rewards", "distance_reward_mode"),
    "zone_reward_min": ("rewards", "zone_reward_min"),
    "zone_reward_max": ("rewards", "zone_reward_max"),
    "zone_reward_curve": ("rewards", "zone_reward_curve"),
    "distance_metric": ("rewards", "distance_metric"),
    "invalid_action_cost": ("rewards", "invalid_action_cost"),
    "construction_residual_weight": ("rewards", "construction_residual_weight"),
    "construction_overshoot_weight": ("rewards", "construction_overshoot_weight"),
}


class NestedFieldAccessMixin:
    """Expose Pydantic fields as ``<container>__<field>`` attributes."""

    exposed_nested_fields: ClassVar[tuple[str, ...]] = ()

    def __getattr__(self, name: str) -> Any:
        container_name, separator, nested_name = name.partition("__")
        if separator and container_name in self.exposed_nested_fields:
            container = getattr(self, container_name)
            if nested_name in type(container).model_fields:
                return getattr(container, nested_name)
        return super().__getattr__(name)


class EnvConfig(NestedFieldAccessMixin, BaseModel):
    """Environment settings grouped by geometry, observation, action, and rewards."""

    model_config = ConfigDict(extra="forbid")
    exposed_nested_fields: ClassVar[tuple[str, ...]] = (
        "geometry",
        "observation",
        "action",
        "rewards",
    )
    agent_count: int = Field(default=4, ge=1)
    max_steps: int = Field(default=200, ge=1)
    seed: int = 42
    trail_mode: bool = True
    target_fill: int | None = Field(default=None, ge=0)
    waypoints_file: str | None = None
    task: TaskConfig = Field(default_factory=TaskConfig)
    geometry: GeometryConfig = Field(default_factory=GeometryConfig)
    observation: ObservationConfig = Field(default_factory=ObservationConfig)
    action: ActionConfig = Field(default_factory=ActionConfig)
    rewards: RewardConfig = Field(default_factory=RewardConfig)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_blocks(cls, value: Any) -> Any:
        """Accept legacy-only input during migration, but reject mixed blocks."""
        if not isinstance(value, dict):
            return value
        data = dict(value)
        legacy_by_block: dict[str, list[str]] = {}
        for legacy, (block, _) in _LEGACY_ENV_FIELDS.items():
            if legacy in data:
                legacy_by_block.setdefault(block, []).append(legacy)
        conflicts = sorted(block for block in legacy_by_block if block in data)
        if conflicts:
            joined = ", ".join(conflicts)
            raise ValueError(
                f"legacy flattened environment fields cannot be mixed with nested blocks: {joined}"
            )
        for block, legacy_fields in legacy_by_block.items():
            nested: dict[str, Any] = {}
            for legacy in legacy_fields:
                _, nested_name = _LEGACY_ENV_FIELDS[legacy]
                nested[nested_name] = data.pop(legacy)
            data[block] = nested
        return data

    def to_runtime_dict(self) -> dict[str, Any]:
        """Return the flat dictionary consumed by the existing environments."""
        action_predicates, action_outcomes = self.action.resolved_pipeline(
            trail_mode=self.trail_mode
        )
        return {
            "stl_path": str(self.geometry__stl_path) if self.geometry__stl_path else None,
            "stl_paths": (
                [str(path) for path in self.geometry__stl_paths] if self.geometry__stl_paths else None
            ),
            "scale": self.geometry__scale,
            "scale_range": self.geometry__scale_range,
            "grid_size": self.geometry__grid_size,
            "geometry_boxes": self.geometry__boxes,
            "geometry_pool_size": self.geometry__pool_size,
            "scale_variants_per_map": self.geometry__scale_variants_per_map,
            "geometry_padding": self.geometry__padding,
            "geometry_pool": self.geometry__pool,
            "obs_mode": self.observation__mode,
            "box_radius": self.observation__box_radius,
            "box_radii": self.observation__box_radii,
            "ray_max_len": self.observation__ray_max_len,
            "include_voxel_count": self.observation__include_voxel_count,
            "action_mode": self.action__mode,
            "action_behavior": self.action.behavior,
            "action_predicates": [
                item.model_dump(mode="json") for item in action_predicates
            ],
            "action_outcomes": [
                item.model_dump(mode="json") for item in action_outcomes
            ],
            "action_history_length": self.action.history_length,
            "agent_count": self.agent_count,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "trail_mode": self.trail_mode,
            "target_fill": self.target_fill,
            "waypoints_file": self.waypoints_file,
            "step_cost": self.rewards__step_cost,
            "collision_cost": self.rewards__collision_cost,
            "goal_reward": self.rewards__goal_reward,
            "distance_shaping": self.rewards__distance_shaping,
            "distance_reward_mode": self.rewards__distance_reward_mode,
            "zone_reward_min": self.rewards__zone_reward_min,
            "zone_reward_max": self.rewards__zone_reward_max,
            "zone_reward_curve": self.rewards__zone_reward_curve,
            "distance_metric": self.rewards__distance_metric,
            "invalid_action_cost": self.rewards__invalid_action_cost,
            "construction_residual_weight": self.rewards__construction_residual_weight,
            "construction_overshoot_weight": self.rewards__construction_overshoot_weight,
            "custom_reward": self.rewards__custom.name if self.rewards__custom else None,
            "custom_reward_parameters": (
                self.rewards__custom.parameters if self.rewards__custom else {}
            ),
            "task": self.task.model_dump(mode="json"),
        }


class TrainingEarlyStopConfig(BaseModel):
    """Evaluation condition that can finish a standard training run early."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    mode: Literal["reward", "heuristic_accuracy", "heuristic_distance", "goal_finishes"] | None = None
    min_iterations: int = Field(default=1, ge=1)
    min_consecutive_evaluation: int = Field(default=1, ge=1)
    min_reward: float | None = None
    min_heuristic_accuracy: float | None = Field(default=None, ge=0.0, le=1.0)
    max_heuristic_distance: float | None = Field(default=None, ge=0.0)
    min_goal_finishes: int | None = Field(default=None, ge=1)
    heuristic_distance_metric: Literal["l1", "l2"] = "l1"
    heuristic_type: Literal[
        "astar", "dijkstra", "weighted_astar", "replanning_astar"
    ] = "astar"
    heuristic_weight: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_selected_threshold(self) -> "TrainingEarlyStopConfig":
        thresholds = {
            "reward": self.min_reward,
            "heuristic_accuracy": self.min_heuristic_accuracy,
            "heuristic_distance": self.max_heuristic_distance,
            "goal_finishes": self.min_goal_finishes,
        }
        configured = [name for name, value in thresholds.items() if value is not None]
        if not self.enabled:
            if self.mode is not None or configured:
                raise ValueError("disabled training.early_stop cannot configure a mode or threshold")
            return self
        if self.mode is None:
            raise ValueError("enabled training.early_stop requires mode")
        if configured != [self.mode]:
            raise ValueError(
                f"training.early_stop mode '{self.mode}' requires exactly its matching threshold"
            )
        if self.heuristic_weight is not None and self.heuristic_type != "weighted_astar":
            raise ValueError("heuristic_weight is only valid for weighted_astar")
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
    num_envs_per_env_runner : int
        Number of vectorized environments hosted by each rollout worker.
    num_gpus_per_env_runner : float
        GPU allocation for each rollout worker. Zero keeps rollout inference on CPU.
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
    num_envs_per_env_runner: int = Field(default=1, ge=1)
    num_gpus_per_env_runner: float = Field(default=0.0, ge=0.0)
    trajectory_every: int = 10
    best_trajectory: bool = True
    output_dir: Path = Path("runtime/")
    video_every: int = 10
    early_stop: TrainingEarlyStopConfig = Field(default_factory=TrainingEarlyStopConfig)


class EvaluationConfig(BaseModel):
    """Deterministic policy evaluation and RLlib evaluation-worker settings."""

    model_config = ConfigDict(extra="forbid")

    episodes: int = Field(default=1, ge=1)
    seed: int = 42
    min_success_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    num_env_runners: int = Field(default=0, ge=0)
    num_envs_per_env_runner: int = Field(default=1, ge=1)


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
        threshold = self.training.early_stop.min_goal_finishes
        if threshold is not None and threshold > self.evaluation.episodes:
            raise ValueError(
                "training.early_stop.min_goal_finishes cannot exceed evaluation.episodes"
            )
        single_agent_algorithms = {"ppo", "dqn", "sac", "rainbow"}
        algorithm = self.training.algorithm.lower()
        if (
            self.training.early_stop.enabled
            and self.training.early_stop.mode in {"heuristic_accuracy", "heuristic_distance"}
            and self.env.agent_count != 1
        ):
            raise ValueError(
                "heuristic comparison early stopping requires env.agent_count: 1"
            )
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
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    anyscale: AnyscaleConfig
    algorithm_config: AlgorithmConfig
    model_cfg: ModelConfig = Field(alias="model_config")
