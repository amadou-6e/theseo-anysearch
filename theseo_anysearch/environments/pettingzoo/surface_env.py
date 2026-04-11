"""PettingZoo surface navigation environment adapter."""

from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium
from gymnasium import spaces

from theseo_anysearch.environments.pettingzoo.base import RustParallelEnv


class SurfaceEnv(RustParallelEnv):
    """
    Multi-agent PettingZoo wrapper for the Rust SurfaceEnv.

    All agents share a single joint action (StepAll) — they advance simultaneously.

    Observation space per agent:
        steps_remaining: Box(1,)   normalised steps remaining
        position:        Box(3,)   current voxel coordinate
        target:          Box(3,)   target voxel coordinate
        reached:         Discrete(2)

    Action space per agent: Discrete(1) — only StepAll
    """

    def _init_possible_agents(self, config: dict) -> list[str]:
        n = config.get("agent_count", 4)
        return [f"agent_{i}" for i in range(n)]

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        max_steps = max(config.get("max_steps", 200), 1)
        self._inv_max_steps = 1.0 / max_steps
        # Pre-allocate one set of buffers per agent
        self._buf_steps_remaining = {a: np.zeros(1, dtype=np.float32) for a in self._possible_agents}
        self._buf_position        = {a: np.zeros(3, dtype=np.float32) for a in self._possible_agents}
        self._buf_target          = {a: np.zeros(3, dtype=np.float32) for a in self._possible_agents}

    def _build_rust_env(self, config: dict) -> Any:
        import theseo_core

        stl_path = config.get("stl_path")
        agent_count = config.get("agent_count", 4)
        max_steps = config.get("max_steps", 200)
        if stl_path:
            stl_ascii = open(stl_path, encoding="utf-8").read()
            return theseo_core.PySurfaceEnv.from_stl(
                stl_ascii,
                origin_x=0,
                origin_y=0,
                origin_z=0,
                scale=float(config.get("scale", 1.0)),
                agent_count=agent_count,
                max_steps=max_steps,
            )
        # Fallback: synthetic line surface with enough coords for all agents.
        surface = [(i, 0, 0) for i in range(max(agent_count * 4, 8))]
        return theseo_core.PySurfaceEnv.from_surface_coords(
            surface,
            agent_count=agent_count,
            max_steps=max_steps,
        )

    def _fanout_obs(self, rust_obs: Any) -> dict:
        inv = self._inv_max_steps
        steps_val = rust_obs.steps_remaining * inv
        result = {}
        for i, agent_id in enumerate(self.agents):
            if i < len(rust_obs.agents):
                a = rust_obs.agents[i]
                sr = self._buf_steps_remaining[agent_id]
                pos = self._buf_position[agent_id]
                tgt = self._buf_target[agent_id]
                sr[0]  = steps_val
                pos[0] = a.current[0]; pos[1] = a.current[1]; pos[2] = a.current[2]
                tgt[0] = a.target[0];  tgt[1] = a.target[1];  tgt[2] = a.target[2]
                result[agent_id] = {
                    "steps_remaining": sr.copy(),
                    "position":        pos.copy(),
                    "target":          tgt.copy(),
                    "reached":         int(a.reached),
                }
            else:
                result[agent_id] = self._zero_obs(agent_id)
        return result

    def _encode_actions(self, actions: dict) -> Any:
        return 0  # StepAll; the action argument is ignored by PySurfaceEnv.step()

    def _observation_space(self, agent: str) -> gymnasium.Space:
        return spaces.Dict({
            "steps_remaining": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "position": spaces.Box(low=0.0, high=1000.0, shape=(3,), dtype=np.float32),
            "target": spaces.Box(low=0.0, high=1000.0, shape=(3,), dtype=np.float32),
            "reached": spaces.Discrete(2),
        })

    def _action_space(self, agent: str) -> gymnasium.Space:
        return spaces.Discrete(1)  # StepAll only
