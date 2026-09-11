"""Experiment-level configuration models for runs, sweeps, rendering, and tracking."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from theseo_anysearch.imitation.models import ImitationConfig
from theseo_anysearch.settings import (
    AlgorithmEnvCompatibilityMixin,
    AlgorithmConfig,
    AnyscaleConfig,
    EnvConfig,
    EvaluationConfig,
    ModelConfig,
    Settings,
    TrainingConfig,
)


def _default_anyscale_config() -> AnyscaleConfig:
    return AnyscaleConfig(cluster_env="", compute_config="", project="")


# ---------------------------------------------------------------------------
# Experiment meta
# ---------------------------------------------------------------------------

class ExperimentMeta(BaseModel):
    """Experiment identity and output metadata.

    Parameters
    ----------
    name : str
        Experiment name used to derive output paths and registry references.
    output_dir : Path
        Base directory containing experiment outputs.
    seed : int
        Experiment-level seed used for reproducibility.
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    output_dir: Path = Path("runtime/experiments")
    seed: int = 42


# ---------------------------------------------------------------------------
# Render config
# ---------------------------------------------------------------------------

class CameraPosition(BaseModel):
    """Named camera preset used by rendering configuration.

    Parameters
    ----------
    name : str
        Camera preset name.
    yaw : float
        Yaw angle in degrees.
    pitch : float
        Pitch angle in degrees.
    """
    model_config = ConfigDict(extra="forbid")

    name: str
    yaw: float = 0.0
    pitch: float = 90.0


class RendersConfig(BaseModel):
    """Rendering configuration for experiment visualization outputs.

    Parameters
    ----------
    camera_positions : list[CameraPosition]
        Named camera presets to use for rendered outputs.
    """
    model_config = ConfigDict(extra="forbid")

    camera_positions: list[CameraPosition] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# MLflow config
# ---------------------------------------------------------------------------

class MLflowConfig(BaseModel):
    """MLflow tracking configuration.

    Parameters
    ----------
    tracking_uri : str | None
        MLflow backend URI. When omitted, MLflow uses its default behavior.
    experiment_name : str | None
        MLflow experiment name override.
    artifact_store : str | None
        Artifact root override for MLflow-managed files.
    """
    model_config = ConfigDict(extra="forbid")

    tracking_uri: str | None = None       # None → MLflow default (./mlruns)
    experiment_name: str | None = None    # None → use experiment.name from YAML
    artifact_store: str | None = None     # None → MLflow default artifact root


# ---------------------------------------------------------------------------
# Tune config — search space + scheduler settings
# ---------------------------------------------------------------------------

class PBTConfig(BaseModel):
    """Population Based Training scheduler configuration.

    Parameters
    ----------
    perturbation_interval : int
        Iteration interval between PBT mutations.
    resample_probability : float
        Probability of resampling a mutated hyperparameter.
    hyperparam_mutations : dict[str, Any]
        Mutation rules applied by the scheduler.
    """
    model_config = ConfigDict(extra="forbid")

    perturbation_interval: int = 20
    resample_probability: float = 0.25
    hyperparam_mutations: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Per-scheduler config blocks (Phase 1 extra params + Phase 2 + Phase 3)
# ---------------------------------------------------------------------------

class ASHASchedulerConfig(BaseModel):
    """ASHA-specific overrides; all fields optional (defaults applied in CLI)."""
    model_config = ConfigDict(extra="forbid")

    max_t: int = 100
    grace_period: int = 10
    reduction_factor: int = 3
    brackets: int = 1


class HyperbandSchedulerConfig(BaseModel):
    """Synchronous Hyperband overrides."""
    model_config = ConfigDict(extra="forbid")

    max_t: int = 100
    reduction_factor: int = 3


class BOHBSchedulerConfig(BaseModel):
    """BOHB-specific overrides."""
    model_config = ConfigDict(extra="forbid")

    max_t: int = 100
    reduction_factor: int = 3


class OptunaSchedulerConfig(BaseModel):
    """Optuna TPE overrides."""
    model_config = ConfigDict(extra="forbid")

    n_startup_trials: int = 10
    max_t: int = 100
    grace_period: int = 10


class CMAESSchedulerConfig(BaseModel):
    """CMA-ES overrides."""
    model_config = ConfigDict(extra="forbid")

    sigma0: float = 0.5


class FLAMLSchedulerConfig(BaseModel):
    """FLAML CFO overrides."""
    model_config = ConfigDict(extra="forbid")

    time_budget_s: int = 600
    max_iter: int | None = None


class TuneConfig(BaseModel):
    """Hyperparameter sweep configuration for experiment-level tuning.

    Parameters
    ----------
    scheduler : str
        Tune scheduler name.
    num_samples : int
        Number of sampled trials to run.
    metric : str
        Optimization metric name.
    mode : {"max", "min"}
        Optimization direction for ``metric``.
    max_concurrent : int
        Maximum number of concurrent trials.
    search_space : dict[str, Any]
        Search space specification expressed in YAML-friendly form.
    asha_config : ASHASchedulerConfig | None
        Optional ASHA scheduler overrides.
    pbt_config : PBTConfig | None
        Optional PBT scheduler overrides.
    hyperband_config : HyperbandSchedulerConfig | None
        Optional Hyperband scheduler overrides.
    bohb_config : BOHBSchedulerConfig | None
        Optional BOHB scheduler overrides.
    optuna_config : OptunaSchedulerConfig | None
        Optional Optuna scheduler overrides.
    cmaes_config : CMAESSchedulerConfig | None
        Optional CMA-ES scheduler overrides.
    flaml_config : FLAMLSchedulerConfig | None
        Optional FLAML scheduler overrides.
    ray_storage_dir : Path | None
        Override for Ray Tune storage root.
    ray_temp_dir : Path | None
        Override for Ray runtime temp directory.
    """
    model_config = ConfigDict(extra="forbid")

    scheduler: Literal[
        "asha", "pbt",
        "random", "hyperband", "bohb",
        "optuna", "cmaes", "flaml",
    ] = "asha"
    num_samples: int = 10
    metric: str = "episode_reward_mean"
    mode: Literal["max", "min"] = "max"
    max_concurrent: int = 4
    search_space: dict[str, Any] = Field(default_factory=dict)
    checkpoint_frequency: int = Field(default=1, ge=1)
    max_environment_steps: int | None = Field(default=None, ge=1)
    max_wall_time_s: float | None = Field(default=None, gt=0.0)
    target_success_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    preserve_trial_artifacts: bool = True
    # Per-scheduler config blocks (all optional; None → use defaults)
    asha_config: ASHASchedulerConfig | None = None
    pbt_config: PBTConfig | None = None
    hyperband_config: HyperbandSchedulerConfig | None = None
    bohb_config: BOHBSchedulerConfig | None = None
    optuna_config: OptunaSchedulerConfig | None = None
    cmaes_config: CMAESSchedulerConfig | None = None
    flaml_config: FLAMLSchedulerConfig | None = None
    # Windows MAX_PATH mitigation — see spec/bugs.md
    # Ray writes deep nested paths under both of these roots.  On Windows the
    # combined path can exceed 260 characters when rooted in a deep workspace.
    # Set either field to a short absolute path (e.g. C:/ray_tmp) to avoid the
    # FileNotFoundError that Ray raises when it cannot create those files.
    # Both default to the system temp dir; override only if you see that error.
    ray_storage_dir: Path | None = None  # where Ray Tune writes driver_artifacts & search state
    ray_temp_dir: Path | None = None
    ray_address: str | None = None
    shutdown_ray_on_complete: bool | None = None     # where Ray writes session temp files (actor logs, etc.)


# ---------------------------------------------------------------------------
# Full experiment config
# ---------------------------------------------------------------------------

class HeuristicConfig(BaseModel):
    """Configuration for an independent heuristic reference trajectory."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    type: Literal[
        "astar",
        "dijkstra",
        "weighted_astar",
        "replanning_astar",
    ] = "astar"
    weight: float | None = None

    @model_validator(mode="after")
    def validate_type_options(self) -> "HeuristicConfig":
        if self.type == "weighted_astar":
            if self.weight is None:
                self.weight = 1.5
            elif self.weight <= 0.0:
                raise ValueError("heuristic.weight must be greater than zero")
        elif self.weight is not None:
            raise ValueError(
                "heuristic.weight is only valid when type is 'weighted_astar'"
            )
        return self


class StageCompletionConfig(BaseModel):
    """Composable condition controlling when a training stage advances."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["iterations", "performance", "all", "any", "not", "python"]
    iterations: int | None = Field(default=None, ge=1)
    metric: str | None = None
    threshold: float | None = None
    comparison: Literal["gte", "gt", "lte", "lt", "eq"] = "gte"
    consecutive_iterations: int = Field(default=1, ge=1)
    conditions: list["StageCompletionConfig"] = Field(default_factory=list)
    condition: "StageCompletionConfig | None" = None
    callable: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    max_iterations: int | None = Field(default=None, ge=1)
    on_max_iterations: Literal["advance", "stop", "error"] | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> "StageCompletionConfig":
        if self.type == "iterations" and self.iterations is None:
            raise ValueError("iterations completion requires iterations")
        if self.type == "performance" and (
            self.metric is None or self.threshold is None
        ):
            raise ValueError("performance completion requires metric and threshold")
        if self.type in {"all", "any"} and not self.conditions:
            raise ValueError(f"{self.type} completion requires conditions")
        if self.type == "not" and self.condition is None:
            raise ValueError("not completion requires condition")
        if self.type == "python" and not self.callable:
            raise ValueError("python completion requires callable")
        if self.max_iterations is not None and self.on_max_iterations is None:
            raise ValueError(
                "completion.max_iterations requires an explicit "
                "on_max_iterations: advance, stop, or error"
            )
        if self.max_iterations is None and self.on_max_iterations is not None:
            raise ValueError(
                "completion.on_max_iterations requires max_iterations"
            )
        return self

    def iteration_limit(self) -> int | None:
        """Return a finite upper bound implied by this condition tree."""
        if self.max_iterations is not None:
            return self.max_iterations
        if self.type == "iterations":
            return self.iterations
        child_limits = [child.iteration_limit() for child in self.conditions]
        if self.type == "any" and any(limit is not None for limit in child_limits):
            return min(limit for limit in child_limits if limit is not None)
        if self.type == "all" and child_limits and all(
            limit is not None for limit in child_limits
        ):
            return max(limit for limit in child_limits if limit is not None)
        return None


class TrainingStageConfig(BaseModel):
    """One ordered training task applied over the shared experiment config."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    completion: StageCompletionConfig
    env: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    algorithm_config: dict[str, Any] = Field(default_factory=dict)
    replay_transition: Literal["clear", "preserve"] | None = None

    @model_validator(mode="after")
    def validate_bounded_completion(self) -> "TrainingStageConfig":
        if self.completion.iteration_limit() is None:
            raise ValueError(
                "stage completion must be bounded by max_iterations or an "
                "iteration condition"
            )
        return self


class StagingConfig(BaseModel):
    """Ordered task stages for progressive policy training."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    resume: bool = True
    replay_transition: Literal["clear", "preserve"] = "clear"
    stages: list[TrainingStageConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_names(self) -> "StagingConfig":
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("staging stage names must be unique")
        return self


def _merge_stage_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> None:
    """Recursively merge one stage override mapping into a copied base mapping."""
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_stage_overrides(base[key], value)
        else:
            base[key] = value

class ExperimentConfig(AlgorithmEnvCompatibilityMixin, BaseModel):
    """
    A single fully-specified experiment.  Extends the base Settings fields
    with experiment meta, renders, MLflow, and an optional tune_config.

    YAML shape (single experiment):
        experiment:
          name: my-run
          output_dir: ./runtime/experiments
          seed: 42
        env: ...
        training: ...
        algorithm_config: ...
        model_config: ...
        renders: ...
        mlflow: ...
        tune_config: ...   # omit for plain training run
    """
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    description: str = ""
    experiment: ExperimentMeta
    env: EnvConfig
    training: TrainingConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    anyscale: AnyscaleConfig = Field(default_factory=_default_anyscale_config)
    algorithm_config: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    model_cfg: ModelConfig = Field(alias="model_config", default_factory=ModelConfig)
    renders: RendersConfig = Field(default_factory=RendersConfig)
    heuristic: HeuristicConfig = Field(default_factory=HeuristicConfig)
    imitation: ImitationConfig = Field(default_factory=ImitationConfig)
    mlflow: MLflowConfig | None = None    # None → tracking disabled
    tune_config: TuneConfig | None = None
    staging: StagingConfig | None = None

    @model_validator(mode="after")
    def validate_heuristic_compatibility(self) -> "ExperimentConfig":
        standalone = self.training.algorithm.lower() == "heuristic"
        if self.heuristic.enabled and self.env.agent_count != 1:
            raise ValueError("heuristic evaluation requires env.agent_count: 1")
        if standalone and not self.heuristic.enabled:
            raise ValueError(
                "training.algorithm='heuristic' requires heuristic.enabled: true"
            )
        if standalone and self.training.runner != "local":
            raise ValueError("standalone heuristic execution requires runner: local")
        if standalone and self.tune_config is not None:
            raise ValueError("standalone heuristic execution does not support tune_config")
        if self.staging is not None and self.staging.enabled:
            if standalone:
                raise ValueError("staged training does not support the heuristic algorithm")
            if self.tune_config is not None:
                raise ValueError("staged training does not support tune_config")
            immutable_env_paths = {
                "agent_count",
                "observation",
                "action",
            }
            replay_algorithms = {"dqn", "rainbow", "sac", "td3", "ddpg"}
            for stage in self.staging.stages:
                changed = immutable_env_paths.intersection(stage.env)
                geometry = stage.env.get("geometry")
                if isinstance(geometry, dict) and "grid_size" in geometry:
                    changed.add("geometry.grid_size")
                if changed:
                    fields = ", ".join(sorted(changed))
                    raise ValueError(
                        f"stage '{stage.name}' changes policy-contract fields: {fields}"
                    )
                changed_training = {"algorithm", "model"}.intersection(stage.training)
                if changed_training:
                    fields = ", ".join(sorted(changed_training))
                    raise ValueError(
                        f"stage '{stage.name}' changes policy-contract training "
                        f"fields: {fields}"
                    )
                transition = stage.replay_transition or self.staging.replay_transition
                if (
                    transition == "preserve"
                    and self.training.algorithm.lower() in replay_algorithms
                ):
                    raise ValueError(
                        f"stage '{stage.name}' requests replay_transition='preserve', "
                        f"but {self.training.algorithm} checkpoints do not guarantee "
                        "replay-buffer preservation; use 'clear'"
                    )
                self._resolved_stage_payload(stage, completed_iterations=0)
        if self.imitation.enabled:
            if standalone:
                raise ValueError("imitation requires a trainable RL algorithm")
            if self.training.algorithm.lower() != "ppo":
                raise ValueError("imitation currently supports PPO only")
            if self.env.agent_count != 1:
                raise ValueError("imitation currently requires env.agent_count: 1")
        return self

    def _resolved_stage_payload(
        self,
        stage: TrainingStageConfig,
        *,
        completed_iterations: int,
    ) -> dict[str, Any]:
        """Build and eagerly validate one base-relative stage configuration."""
        raw = self.model_dump(by_alias=True, mode="python")
        raw.pop("staging", None)
        _merge_stage_overrides(raw["env"], stage.env)
        _merge_stage_overrides(raw["training"], stage.training)
        _merge_stage_overrides(raw["evaluation"], stage.evaluation)
        _merge_stage_overrides(raw["algorithm_config"], stage.algorithm_config)
        limit = stage.completion.iteration_limit()
        if limit is None:
            raise RuntimeError("stage completion has no finite iteration limit")
        raw["training"]["iterations"] = completed_iterations + limit

        from theseo_anysearch.experiments.loader import _resolve_typed_configs

        resolved = _resolve_typed_configs(raw)
        ExperimentConfig(**resolved)
        return resolved

    def stage_experiment(
        self,
        stage_index: int,
        *,
        completed_iterations: int,
    ) -> "ExperimentConfig":
        """Resolve one stage into a complete experiment configuration."""
        if self.staging is None or not self.staging.enabled:
            return self
        stage = self.staging.stages[stage_index]
        return ExperimentConfig(**self._resolved_stage_payload(
            stage,
            completed_iterations=completed_iterations,
        ))

    @property
    def run_output_dir(self) -> Path:
        """Base directory under which run_id subdirectories are created."""
        return self.experiment.output_dir / self.experiment.name

    def to_settings(self) -> Settings:
        """Return a plain Settings object for use with Trainer."""
        return Settings(
            env=self.env,
            training=self.training,
            evaluation=self.evaluation,
            anyscale=self.anyscale,
            algorithm_config=self.algorithm_config,
            model_config=self.model_cfg,
            imitation=self.imitation,
        )


# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------

class SweepConfig(BaseModel):
    """
    A sweep YAML has a top-level ``sweep:`` key with ``base`` + ``experiments``.
    Each entry deep-merges over ``base`` and produces an independent ExperimentConfig.
    """
    model_config = ConfigDict(extra="forbid")

    description: str = ""
    base: dict[str, Any]
    experiments: list[dict[str, Any]]
