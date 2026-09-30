"""Trainer-specific helper models used by RLlib training code."""

from __future__ import annotations

from pydantic import ConfigDict

from theseo_anysearch.settings import TrainingConfig


class TrainerConfig(TrainingConfig):
    """Config for the Trainer. Extends TrainingConfig with trainer-specific fields."""
    model_config = ConfigDict(extra="forbid")
