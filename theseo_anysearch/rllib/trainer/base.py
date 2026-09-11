"""Abstract interfaces for RLlib training orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theseo_anysearch.settings import Settings
    from theseo_anysearch.rllib.trainer.results import TrainResult


class BaseTrainer(ABC):
    """Define the public lifecycle implemented by project trainers."""

    @classmethod
    @abstractmethod
    def from_settings(cls, config: "Settings") -> "BaseTrainer":
        """Construct a trainer from validated project settings."""

    @abstractmethod
    def train(self) -> list["TrainResult"]:
        """Execute the configured training lifecycle."""

    @abstractmethod
    def checkpoint(self) -> Path:
        """Persist trainer state and return its checkpoint path."""

    @abstractmethod
    def restore(self, checkpoint_dir: Path) -> None:
        """Restore trainer state from a checkpoint."""

    @abstractmethod
    def resume(self) -> bool:
        """Restore the latest checkpoint when one exists."""


__all__ = ["BaseTrainer"]
