"""Abstract ownership points for RLlib runner implementations."""

from __future__ import annotations

from abc import ABC, abstractmethod

from theseo_anysearch.rllib.trainer.trainer import Trainer


class Runner(ABC):
    """Abstract base for training execution backends."""

    @abstractmethod
    def submit(self, trainer: Trainer) -> dict:
        """Submit a training job and return the result."""

    @abstractmethod
    def status(self) -> str:
        """Return the current job status."""

    @abstractmethod
    def cancel(self) -> None:
        """Cancel the running job."""
