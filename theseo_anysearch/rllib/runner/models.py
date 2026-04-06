from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class RunnerConfig(BaseModel):
    """Base config for all runner backends."""
    model_config = ConfigDict(extra="forbid")


class LocalRunnerConfig(RunnerConfig):
    """Config for the local single-process runner."""
    model_config = ConfigDict(extra="forbid")


class AnyscaleRunnerConfig(RunnerConfig):
    """Config for the Anyscale distributed runner."""
    model_config = ConfigDict(extra="forbid")

    cluster_env: str
    compute_config: str
    project: str
