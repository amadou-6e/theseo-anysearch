from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRLModule(ABC):
    """Abstract base for custom RLlib neural network modules."""

    @abstractmethod
    def encode_observation(self, obs: dict) -> object:
        """Map raw observation dict to a latent embedding tensor."""
