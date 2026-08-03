"""Abstract interfaces for RLlib training orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from theseo_anysearch.settings import Settings
    from theseo_anysearch.rllib.trainer.results import TrainResult


class BaseTrainer(ABC):
    """Define the public lifecycle implemented by project trainers.

    Notes
    -----
    This interface contains no RLlib algorithm construction or training-loop
    implementation. Those responsibilities belong to algorithm adapters and
    the concrete trainer module.
    """

    @classmethod
    @abstractmethod
    def from_settings(cls, config: "Settings") -> "BaseTrainer":
        """Construct a trainer from validated project settings.

        Parameters
        ----------
        config : Settings
            Validated experiment settings.

        Returns
        -------
        BaseTrainer
            Configured trainer instance.
        """

    @abstractmethod
    def train(self) -> list["TrainResult"]:
        """Execute the configured training lifecycle.

        Returns
        -------
        list[TrainResult]
            Results produced for completed training iterations.
        """

    @abstractmethod
    def checkpoint(self) -> Path:
        """Persist trainer state and return its checkpoint path.

        Returns
        -------
        pathlib.Path
            Created checkpoint directory.
        """

    @abstractmethod
    def restore(self, checkpoint_dir: Path) -> None:
        """Restore trainer state from a checkpoint.

        Parameters
        ----------
        checkpoint_dir : pathlib.Path
            Checkpoint directory to restore.
        """

    @abstractmethod
    def resume(self) -> bool:
        """Restore the latest checkpoint when one exists.

        Returns
        -------
        bool
            True when state was restored, otherwise False.
        """


__all__ = ["BaseTrainer"]
