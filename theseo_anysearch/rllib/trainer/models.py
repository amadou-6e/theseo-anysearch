from __future__ import annotations

from pydantic import ConfigDict

from theseo_anysearch.models import TrainingConfig


class TrainerConfig(TrainingConfig):
    """Config for the Trainer. Extends TrainingConfig with trainer-specific fields."""
    model_config = ConfigDict(extra="forbid")
