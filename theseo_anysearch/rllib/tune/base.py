from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal


class BaseTuneConfig(ABC):
    """Abstract base for Ray Tune experiment configurations."""

    @abstractmethod
    def search_space(self) -> dict:
        """Return a ray.tune param_space dict."""

    @abstractmethod
    def scheduler(self) -> object:
        """Return a Ray Tune TrialScheduler instance."""

    @abstractmethod
    def metric(self) -> str:
        """Return the RLlib result key to optimise."""

    @abstractmethod
    def mode(self) -> Literal["min", "max"]:
        """Return the optimisation direction."""

    def search_alg(self) -> object | None:
        """Return a Ray Tune SearchAlgorithm, or None to use random search."""
        return None
