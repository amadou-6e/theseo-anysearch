"""Abstract interfaces for RLlib training orchestration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from theseo_anysearch.models import Settings
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


def __getattr__(name: str) -> Any:
    """Resolve legacy imports from trainer.base lazily.

    Parameters
    ----------
    name : str
        Requested compatibility export.

    Returns
    -------
    Any
        Object provided by its new owning module.

    Raises
    ------
    AttributeError
        If name is not a supported compatibility export.
    """
    if name in {"Trainer", "_detect_num_gpus", "_resolve_pool_dir"}:
        from theseo_anysearch.rllib.trainer import trainer
        from theseo_anysearch.rllib.trainer import runtime

        return getattr(trainer if name == "Trainer" else runtime, name)
    if name in {"RllibTrainResult", "TrainResult"}:
        from theseo_anysearch.rllib.trainer import results

        return getattr(results, name)
    raise AttributeError(name)


__all__ = ["BaseTrainer"]
