"""Base models and helpers for Gymnasium-backed environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
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
        self._step_log_count = 0
        self._obs_log_count = 0
        self._debug_log_path = config.get("debug_log_path")
        self.observation_space = self._observation_space()
        self.action_space = self._action_space()
        self._rust_env: Any = self._build_rust_env(config)

    def _log_env_stage(self, message: str) -> None:
        """Append a timestamped environment stage marker when debug logging is enabled."""
        if not getattr(self, "_debug_log_path", None):
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path = Path(str(self._debug_log_path))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(f"[env] {ts} {message}\n")

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
        if self._reset_count <= 3:
            self._log_env_stage(f"reset start count={self._reset_count} seed={seed_val}")
        if self._rust_env is not None:
            rust_obs = self._rust_env.reset(seed_val)
            if self._reset_count <= 3:
                self._log_env_stage(f"reset rust_env returned count={self._reset_count}")
            obs = self._obs_to_numpy(rust_obs)
        else:
            obs = {k: np.zeros(v.shape, dtype=v.dtype) for k, v in self.observation_space.items()}
        if self._reset_count <= 3:
            self._log_env_stage(f"reset done count={self._reset_count}")
        return obs, {}

    def step(self, action):
        if self._rust_env is None:
            raise NotImplementedError("Rust env not initialised")
        self._step_log_count += 1
        if self._step_log_count <= 5:
            self._log_env_stage(f"step start index={self._step_log_count} action={action}")
        result = self._rust_env.step(self._encode_action(action))
        if self._step_log_count <= 5:
            self._log_env_stage(
                f"step rust_env returned index={self._step_log_count} reward={result.reward} done={result.done}"
            )
        obs = self._obs_to_numpy(result.observation)
        if self._step_log_count <= 5:
            self._log_env_stage(f"step done index={self._step_log_count}")
        return obs, result.reward, result.done, False, {}

    def _encode_action(self, action: Any) -> Any:
        """Encode a numpy/int action into a Rust action type. Override in subclasses."""
        return action
