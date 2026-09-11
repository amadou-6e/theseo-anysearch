"""Configuration models for RLlib hyperparameter tuning."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TuneConfig(BaseModel):
    """Base config for a Ray Tune experiment."""
    model_config = ConfigDict(extra="forbid")

    metric: str = "episode_reward_mean"
    mode: Literal["min", "max"] = "max"
    num_samples: int = 20


class ASHAConfig(TuneConfig):
    """Config for the ASHA (Asynchronous Successive Halving Algorithm) scheduler."""
    model_config = ConfigDict(extra="forbid")

    max_t: int = 100
    grace_period: int = 10
    reduction_factor: int = 3
    brackets: int = 1


class PBTConfig(TuneConfig):
    """Config for Population Based Training scheduler."""
    model_config = ConfigDict(extra="forbid")

    perturbation_interval: int = 10
    resample_probability: float = 0.25
    hyperparam_mutations: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------

class RandomConfig(TuneConfig):
    """Config for Random Search (no early stopping; baseline comparator)."""
    model_config = ConfigDict(extra="forbid")


class HyperbandConfig(TuneConfig):
    """Config for the synchronous Hyperband scheduler."""
    model_config = ConfigDict(extra="forbid")

    max_t: int = 100
    reduction_factor: int = 3


class BOHBConfig(TuneConfig):
    """Config for BOHB (Hyperband + Bayesian Optimisation via TuneBOHB)."""
    model_config = ConfigDict(extra="forbid")

    max_t: int = 100
    reduction_factor: int = 3


# ---------------------------------------------------------------------------
# Phase 3
# ---------------------------------------------------------------------------

class OptunaConfig(TuneConfig):
    """Config for Optuna TPE search + ASHA early stopping."""
    model_config = ConfigDict(extra="forbid")

    n_startup_trials: int = 10
    # ASHA-compatible early-stopping for Optuna trials
    max_t: int = 100
    grace_period: int = 10


class CMAESConfig(TuneConfig):
    """Config for CMA-ES evolutionary strategy via nevergrad."""
    model_config = ConfigDict(extra="forbid")

    sigma0: float = 0.5


class FLAMLConfig(TuneConfig):
    """Config for FLAML cost-aware search (CFO)."""
    model_config = ConfigDict(extra="forbid")

    time_budget_s: int = 600
    max_iter: int | None = None
