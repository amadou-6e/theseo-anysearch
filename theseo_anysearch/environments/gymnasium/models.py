"""Configuration models for Gymnasium environment wrappers."""

from __future__ import annotations

from pydantic import ConfigDict

from theseo_anysearch.models import EnvConfig


class GymnasiumEnvConfig(EnvConfig):
    """Config for single-agent Gymnasium wrappers. Extends EnvConfig with gym-specific fields."""
    model_config = ConfigDict(extra="forbid")
