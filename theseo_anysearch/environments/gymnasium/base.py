from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import gymnasium


class RustGymnasiumEnv(gymnasium.Env, ABC):
    """
    Abstract base for single-agent Gymnasium wrappers over PyO3-bound Rust environments.
    Subclasses implement _build_rust_env, _observation_space, and _action_space.
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self._config = config
        self._reset_count = 0
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()
        self._rust_env: Any = self._build_rust_env(config)

    @abstractmethod
    def _build_rust_env(self, config: dict) -> Any:
        """Instantiate the Rust environment via PyO3."""

    @abstractmethod
    def _observation_space(self) -> gymnasium.Space:
        """Return the observation space for this environment."""

    @abstractmethod
    def _action_space(self) -> gymnasium.Space:
        """Return the action space for this environment."""

    @abstractmethod
    def _obs_to_numpy(self, rust_obs: Any) -> dict:
        """Convert a Rust observation struct to a dict of numpy arrays."""

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self._reset_count += 1
        seed_val = seed if seed is not None else self._config.get("seed", 42) + self._reset_count
        if self._rust_env is not None:
            rust_obs = self._rust_env.reset(seed_val)
            obs = self._obs_to_numpy(rust_obs)
        else:
            obs = {k: np.zeros(v.shape, dtype=v.dtype) for k, v in self.observation_space.items()}
        return obs, {}

    def step(self, action):
        if self._rust_env is None:
            raise NotImplementedError("Rust env not initialised")
        result = self._rust_env.step(self._encode_action(action))
        obs = self._obs_to_numpy(result.observation)
        return obs, result.reward, result.done, False, {}

    def _encode_action(self, action: Any) -> Any:
        """Encode a numpy/int action into a Rust action type. Override in subclasses."""
        return action
