from __future__ import annotations

from typing import Any

import numpy as np
import gymnasium
from gymnasium import spaces

from theseo_anysearch.environments.pettingzoo.base import RustParallelEnv

def _max_manhattan(grid_size: int) -> float:
    return 3.0 * (grid_size - 1)


class MultiVoxelEnv(RustParallelEnv):
    """
    Multi-agent PettingZoo wrapper for the Rust PyMultiVoxelEnv.

    N agents share a single configurable voxel grid. Each agent has its own
    cursor and (when geometry is provided) its own start/goal pair. Agents
    cannot move into geometry or filled cells but may co-exist on the same cell.
    Trail mode auto-fills each agent's destination on successful moves.

    Observation space per agent:
        steps_remaining: Box(1,)   normalised steps remaining [0, 1]
        voxel_count:     Box(1,)   total agent-filled cells
        cursor_pos:      Box(3,)   normalised cursor position [0, 1]³
        goal_distance:   Box(1,)   normalised Manhattan distance (when geometry set)

    Action space per agent: Discrete(26) — all 26 face/edge/corner neighbors
    """

    ray_env_id = "MultiVoxelEnv-v0"

    @classmethod
    def register_with_ray(cls, env_config: dict) -> str:
        from ray.tune.registry import register_env
        register_env(cls.ray_env_id, lambda cfg: cls(cfg or env_config))
        return cls.ray_env_id

    def _init_possible_agents(self, config: dict) -> list[str]:
        n = config.get("agent_count", 2)
        return [f"agent_{i}" for i in range(n)]

    def _build_rust_env(self, config: dict) -> Any:
        import theseo_core

        geometry: list[tuple[int, int, int]] = []
        for box in config.get("geometry_boxes") or []:
            xmin, ymin, zmin, xmax, ymax, zmax = box
            for bx in range(xmin, xmax + 1):
                for by in range(ymin, ymax + 1):
                    for bz in range(zmin, zmax + 1):
                        geometry.append((bx, by, bz))

        return theseo_core.PyMultiVoxelEnv(
            agent_count=config.get("agent_count", 2),
            max_steps=config.get("max_steps", 200),
            trail_mode=config.get("trail_mode", False),
            geometry=geometry or None,
            grid_size=config.get("grid_size", 32),
            step_cost=config.get("step_cost", -0.01),
            goal_reward=config.get("goal_reward", 1.0),
            distance_shaping=config.get("distance_shaping", 0.1),
            collision_cost=config.get("collision_cost", 0.0),
        )

    def _has_goal(self) -> bool:
        return bool(self._config.get("geometry_boxes"))

    def _observation_space(self, agent: str) -> gymnasium.Space:
        base = {
            "steps_remaining": spaces.Box(0.0, 1.0, (1,), np.float32),
            "voxel_count":     spaces.Box(0.0, np.inf, (1,), np.float32),
            "cursor_pos":      spaces.Box(0.0, 1.0, (3,), np.float32),
        }
        if self._has_goal():
            base["goal_distance"] = spaces.Box(0.0, 1.0, (1,), np.float32)
        return spaces.Dict(base)

    def _action_space(self, agent: str) -> gymnasium.Space:
        return spaces.Discrete(26)

    def _fanout_obs(self, rust_obs: Any) -> dict:
        max_steps = self._config.get("max_steps", 200)
        grid_size = self._config.get("grid_size", 32)
        norm = float(max(grid_size - 1, 1))
        steps_norm = np.array([rust_obs.steps_remaining / max(max_steps, 1)], dtype=np.float32)
        voxel_count = np.array([rust_obs.voxel_count], dtype=np.float32)

        result = {}
        for i, agent_id in enumerate(self.agents):
            if i < len(rust_obs.cursors):
                cx, cy, cz = rust_obs.cursors[i]
                cursor_pos = np.array(
                    [(cx - 1) / norm, (cy - 1) / norm, (cz - 1) / norm], dtype=np.float32
                )
                obs = {
                    "steps_remaining": steps_norm,
                    "voxel_count": voxel_count,
                    "cursor_pos": cursor_pos,
                }
                if self._has_goal() and i < len(rust_obs.goal_distances):
                    gd = rust_obs.goal_distances[i]
                    dist = (gd / _max_manhattan(grid_size)) if gd is not None else 0.0
                    obs["goal_distance"] = np.array([dist], dtype=np.float32)
            else:
                obs = self._zero_obs(agent_id)
            result[agent_id] = obs
        return result

    def _encode_actions(self, actions: dict) -> list[int]:
        return [int(actions.get(agent, 0)) for agent in self.possible_agents]

    def _fanout_rewards(self, result: Any) -> dict:
        return {
            agent: float(result.rewards[i]) if i < len(result.rewards) else 0.0
            for i, agent in enumerate(self.agents)
        }

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
