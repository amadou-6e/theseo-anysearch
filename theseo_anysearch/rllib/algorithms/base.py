from __future__ import annotations

from abc import ABC, abstractmethod


class BaseAlgorithmConfig(ABC):
    """Abstract base for project algorithm configs. Wraps RLlib AlgorithmConfig construction."""

    @abstractmethod
    def build(self) -> object:
        """Return a fully configured ray.rllib.algorithms.AlgorithmConfig."""

    @abstractmethod
    def env_config(self) -> dict:
        """Return env kwargs passed through to the registered environment."""
