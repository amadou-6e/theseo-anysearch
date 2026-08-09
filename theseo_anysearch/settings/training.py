"""Training configuration models."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

class TrainingEarlyStopConfig(BaseModel):
    """Evaluation condition that can finish a standard training run early."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(False, description="Enable evaluation-based early termination of training.")
    mode: Literal["reward", "heuristic_accuracy", "heuristic_distance", "goal_finishes"] | None = Field(None, description="Metric family used by the stopping condition.")
    min_iterations: int = Field(default=1, ge=1, description="Training iterations required before stopping is allowed.")
    min_consecutive_evaluation: int = Field(default=1, ge=1, description="Consecutive qualifying evaluations required to stop.")
    min_reward: float | None = Field(None, description="Minimum evaluation reward for reward-mode stopping.")
    min_heuristic_accuracy: float | None = Field(default=None, ge=0.0, le=1.0, description="Minimum next-action agreement with the selected heuristic.")
    max_heuristic_distance: float | None = Field(default=None, ge=0.0, description="Maximum L1 or L2 action distance from the heuristic.")
    min_goal_finishes: int | None = Field(default=None, ge=1, description="Minimum successful evaluation episodes.")
    heuristic_distance_metric: Literal["l1", "l2"] = Field("l1", description="Unit-block distance used for heuristic comparison.")
    heuristic_type: Literal[
        "astar", "dijkstra", "weighted_astar", "replanning_astar"
    ] = Field("astar", description="Heuristic used by comparison-based stopping.")
    heuristic_weight: float | None = Field(default=None, gt=0.0, description="Weighted-A-star heuristic multiplier.")

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
    num_learners : int
        Number of remote Learner actors; zero keeps learning on the driver.
    num_cpus_per_learner : int
        CPU allocation for each remote Learner actor.
    num_gpus_per_learner : float | None
        GPU allocation per Learner; defaults to training.num_gpus.
    num_env_runners : int
        Number of rollout workers or env runners.
    num_envs_per_env_runner : int
        Number of vectorized environments hosted by each rollout worker.
    num_gpus_per_env_runner : float
        GPU allocation for each rollout worker. Zero keeps rollout inference on CPU.
    weight_sync_interval : int
        DQN training steps between EnvRunner weight broadcasts.
    max_requests_in_flight_per_env_runner : int
        Maximum concurrent sample requests queued for each remote EnvRunner.
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

    algorithm: str = Field(description="Registered trainer algorithm name.")
    model: str = Field("voxel_encoder", description="Registered model configuration family.")
    runner: Literal["local", "anyscale"] = Field("local", description="Execution backend used for training.")
    iterations: int = Field(100, ge=1, description="Maximum number of training iterations.")
    checkpoint_interval: int = Field(10, ge=1, description="Training iterations between checkpoints.")
    require_gpu: bool = Field(False, description="Fail startup when no compatible GPU is available.")
    num_gpus: float | None = Field(None, ge=0.0, description="Explicit RLlib GPU allocation override.")
    num_learners: int = Field(default=0, ge=0, description="Remote Learner actors; zero learns in the driver process.")
    num_cpus_per_learner: int = Field(default=1, ge=0, description="CPU allocation assigned to each remote Learner.")
    num_gpus_per_learner: float | None = Field(default=None, ge=0.0, description="GPU allocation per Learner; defaults to training.num_gpus.")
    num_env_runners: int = Field(0, ge=0, description="Rollout workers; zero samples inline.")
    num_envs_per_env_runner: int = Field(default=1, ge=1, description="Vectorized environments hosted by each rollout worker.")
    num_gpus_per_env_runner: float = Field(default=0.0, ge=0.0, description="GPU allocation assigned to each rollout worker.")
    weight_sync_interval: int = Field(default=1, ge=1, description="DQN training steps between EnvRunner weight broadcasts.")
    max_requests_in_flight_per_env_runner: int = Field(default=2, ge=1, description="Maximum concurrent sample requests per remote EnvRunner.")
    trajectory_every: int = Field(10, ge=1, description="Iterations between trajectory snapshots.")
    best_trajectory: bool = Field(True, description="Retain the best evaluation trajectory.")
    output_dir: Path = Field(default=Path("runtime/"), description="Base directory for training artifacts.")
    video_every: int = Field(10, ge=1, description="Iterations between rendered videos.")
    early_stop: TrainingEarlyStopConfig = Field(default_factory=TrainingEarlyStopConfig, description="Evaluation-based training stop condition.")
