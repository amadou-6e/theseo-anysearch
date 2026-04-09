"""Base ownership points for reward configuration objects."""

from __future__ import annotations

from abc import ABC, abstractmethod


class RewardShaper(ABC):
    """
    Abstract base for composable reward shaping strategies.
    Applied in a pipeline before RLlib processes step rewards.
    """

    @abstractmethod
    def shape(self, agent_id: str, reward: float, obs: dict, done: bool) -> float:
        """Return a shaped reward for the given agent step."""
