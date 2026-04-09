"""Base models and helpers for PettingZoo-backed environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import gymnasium
from pettingzoo import ParallelEnv


class RustParallelEnv(ParallelEnv, ABC):
    """
    Abstract base for multi-agent PettingZoo wrappers over PyO3-bound Rust environments.
    All agents act and observe simultaneously (parallel API).
    """

    def __init__(self, config: dict) -> None:
        super().__init__()
        self._config = config
        self._reset_count = 0
        self._possible_agents: list[str] = self._init_possible_agents(config)
        self.agents: list[str] = list(self._possible_agents)
        self._rust_env: Any = self._build_rust_env(config)

    @abstractmethod
    def _build_rust_env(self, config: dict) -> Any:
        """Instantiate the Rust environment via PyO3."""

    @abstractmethod
    def _observation_space(self, agent: str) -> gymnasium.Space:
        """Return the observation space for a given agent."""

    @abstractmethod
    def _action_space(self, agent: str) -> gymnasium.Space:
        """Return the action space for a given agent."""

    @abstractmethod
    def _init_possible_agents(self, config: dict) -> list[str]:
        """Return the list of all possible agent IDs for this config."""

    def observation_space(self, agent: str) -> gymnasium.Space:
        return self._observation_space(agent)

    def action_space(self, agent: str) -> gymnasium.Space:
        return self._action_space(agent)

    @property
    def possible_agents(self) -> list[str]:
        return list(self._possible_agents)

    def reset(self, seed: int | None = None, options: dict | None = None):
        self.agents = list(self._possible_agents)
        self._reset_count += 1
        seed_val = seed if seed is not None else self._config.get("seed", 42) + self._reset_count
        if self._rust_env is not None:
            rust_obs = self._rust_env.reset(seed_val)
            obs = self._fanout_obs(rust_obs)
        else:
            obs = {agent: self._zero_obs(agent) for agent in self.agents}
        return obs, {agent: {} for agent in self.agents}

    def step(self, actions: dict):
        if self._rust_env is None:
            raise NotImplementedError("Rust env not initialised")
        result = self._rust_env.step(self._encode_actions(actions))
        obs = self._fanout_obs(result.observation)
        rewards = self._fanout_rewards(result)
        terminations = {agent: result.done for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        if result.done:
            self.agents = []
        return obs, rewards, terminations, truncations, {agent: {} for agent in self.agents}

    def _zero_obs(self, agent: str) -> dict:
        space = self._observation_space(agent)
        return {k: np.zeros(v.shape, dtype=v.dtype) for k, v in space.items()}

    def _fanout_obs(self, rust_obs: Any) -> dict:
        """Convert Rust observation to per-agent dict. Override in subclasses."""
        return {agent: self._zero_obs(agent) for agent in self.agents}

    def _fanout_rewards(self, result: Any) -> dict:
        """Extract per-agent rewards from step result. Override in subclasses."""
        return {agent: result.reward for agent in self.agents}

    def _encode_actions(self, actions: dict) -> Any:
        """Encode per-agent action dict into a Rust action. Override in subclasses."""
        return actions
