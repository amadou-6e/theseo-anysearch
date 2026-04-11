"""Configuration models for PettingZoo environment wrappers."""

from __future__ import annotations

from pydantic import ConfigDict

from theseo_anysearch.models import EnvConfig


class ParallelEnvConfig(EnvConfig):
    """Config for multi-agent PettingZoo wrappers. Extends EnvConfig with multi-agent fields."""
    model_config = ConfigDict(extra="forbid")
