from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.models import (
    AlgorithmConfig,
    AnyscaleConfig,
    EnvConfig,
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
    model_config = ConfigDict(extra="forbid")

    name: str
    output_dir: Path = Path("runtime/experiments")
    seed: int = 42


# ---------------------------------------------------------------------------
# Render config
# ---------------------------------------------------------------------------

class CameraPosition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    yaw: float = 0.0
    pitch: float = 90.0


class RendersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_positions: list[CameraPosition] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# MLflow config
# ---------------------------------------------------------------------------

class MLflowConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tracking_uri: str | None = None       # None → MLflow default (./mlruns)
    experiment_name: str | None = None    # None → use experiment.name from YAML
    artifact_store: str | None = None     # None → MLflow default artifact root


# ---------------------------------------------------------------------------
# Tune config — search space + scheduler settings
# ---------------------------------------------------------------------------

class PBTConfig(BaseModel):
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
    # Per-scheduler config blocks (all optional; None → use defaults)
    asha_config: ASHASchedulerConfig | None = None
    pbt_config: PBTConfig | None = None
    hyperband_config: HyperbandSchedulerConfig | None = None
    bohb_config: BOHBSchedulerConfig | None = None
    optuna_config: OptunaSchedulerConfig | None = None
    cmaes_config: CMAESSchedulerConfig | None = None
    flaml_config: FLAMLSchedulerConfig | None = None


# ---------------------------------------------------------------------------
# Full experiment config
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
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

    experiment: ExperimentMeta
    env: EnvConfig
    training: TrainingConfig
    anyscale: AnyscaleConfig = Field(default_factory=_default_anyscale_config)
    algorithm_config: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    model_cfg: ModelConfig = Field(alias="model_config", default_factory=ModelConfig)
    renders: RendersConfig = Field(default_factory=RendersConfig)
    mlflow: MLflowConfig | None = None    # None → tracking disabled
    tune_config: TuneConfig | None = None

    @property
    def run_output_dir(self) -> Path:
        """Base directory under which run_id subdirectories are created."""
        return self.experiment.output_dir / self.experiment.name

    def to_settings(self) -> Settings:
        """Return a plain Settings object for use with Trainer."""
        return Settings(
            env=self.env,
            training=self.training,
            anyscale=self.anyscale,
            algorithm_config=self.algorithm_config,
            model_config=self.model_cfg,
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

    base: dict[str, Any]
    experiments: list[dict[str, Any]]
