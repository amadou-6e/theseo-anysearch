from __future__ import annotations

import math
from typing import Any

import numpy as np
import gymnasium
from gymnasium import spaces

from theseo_anysearch.environments.pettingzoo.base import RustParallelEnv


def _max_manhattan(grid_size: int) -> float:
    return 3.0 * (grid_size - 1)


def _max_euclidean(grid_size: int) -> float:
    return math.sqrt(3.0) * (grid_size - 1)


def _stl_max_extent(path: str) -> float:
    """Return the maximum bounding-box extent of an ASCII STL's vertices."""
    verts: list[list[float]] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("vertex"):
                parts = stripped.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        return 1.0
    arr = np.array(verts, dtype=np.float64)
    extent = float((arr.max(axis=0) - arr.min(axis=0)).max())
    return extent if extent > 0.0 else 1.0


def _load_stl_geometry(path: str, scale: float, grid_size: int) -> list[tuple[int, int, int]]:
    """Voxelize an STL file and return the filled (solid) voxel coordinates.

    ``scale`` is the *target resolution* — the number of voxels the STL's
    longest axis should span.  It is converted to voxels-per-unit before
    being passed to the Rust voxelizer, which prevents the sampler from
    iterating over millions of sub-voxel steps on large STL meshes.
    """
    import theseo_core
    max_extent = _stl_max_extent(path)
    voxels_per_unit = scale / max_extent
    sampler = theseo_core.PyVoxelSampler(grid_size=grid_size)
    sampler.load_stl(path, voxels_per_unit)
    free = set(sampler.free_cells())
    return [
        (x, y, z)
        for x in range(1, grid_size + 1)
        for y in range(1, grid_size + 1)
        for z in range(1, grid_size + 1)
        if (x, y, z) not in free
    ]


class MultiVoxelEnv(RustParallelEnv):
    """
    Multi-agent PettingZoo wrapper for the Rust PyMultiVoxelEnv.

    N agents share a single configurable voxel grid. Each agent has its own
    cursor and (when geometry is provided) its own start/goal pair. Agents
    cannot move into geometry or filled cells but may co-exist on the same cell.
    Trail mode auto-fills each agent's destination on successful moves.

    Observation space per agent:
        cursor_pos:      Box(3,)   normalised cursor position [0, 1]³
        face_neighbors:  Box(6,)   binary fill state of 6 cardinal neighbors (+x-x+y-y+z-z)
        local_grid:      Box(N³,)  binary fill state of (2*box_radius+1)³ box around cursor
        ray_cast:        Box(27,)  distance to nearest filled cell in 27 directions (0=adjacent, 1=none)
        goal_distance:   Box(1,)   normalised distance to goal (when geometry set)
        goal_direction:  Box(3,)   unit vector toward goal, zeros when at goal (when geometry set)

    Config keys:
        box_radius:       int  (default 2)         — local_grid cube side = 2*box_radius+1
        ray_max_len:      int  (default 16)        — maximum ray length for ray_cast
        distance_metric:  str  (default "euclidean") — "euclidean" or "manhattan"

    Action space per agent: Discrete(26) — all 26 face/edge/corner neighbors
    """

    ray_env_id = "MultiVoxelEnv-v0"

    def __init__(self, config: dict) -> None:
        self._obs_rng = np.random.default_rng(config.get("seed", 42))
        super().__init__(config)

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

        grid_size = config.get("grid_size", 32)
        geometry: list[tuple[int, int, int]] = []

        if config.get("stl_path"):
            scale = float(config.get("scale", 1.0))
            geometry = _load_stl_geometry(str(config["stl_path"]), scale, grid_size)
        else:
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
            grid_size=grid_size,
            step_cost=config.get("step_cost", -0.01),
            goal_reward=config.get("goal_reward", 1.0),
            distance_shaping=config.get("distance_shaping", 0.1),
            collision_cost=config.get("collision_cost", 0.0),
        )

    def _has_goal(self) -> bool:
        return bool(self._config.get("geometry_boxes") or self._config.get("stl_path"))

    def reset(self, seed: int | None = None, options: dict | None = None):
        scale_range = self._config.get("scale_range")
        stl_path = self._config.get("stl_path")
        if scale_range and stl_path:
            lo, hi = float(scale_range[0]), float(scale_range[1])
            new_scale = float(self._obs_rng.uniform(lo, hi))
            cfg = dict(self._config)
            cfg["scale"] = new_scale
            self._rust_env = self._build_rust_env(cfg)
        return super().reset(seed, options)

    def _observation_space(self, agent: str) -> gymnasium.Space:
        radius = self._config.get("box_radius", 2)
        n = 2 * radius + 1
        base = {
            "cursor_pos":     spaces.Box(0.0, 1.0, (3,),    np.float32),
            "face_neighbors": spaces.Box(0.0, 1.0, (6,),    np.float32),
            "local_grid":     spaces.Box(0.0, 1.0, (n**3,), np.float32),
            "ray_cast":       spaces.Box(0.0, 1.0, (27,),   np.float32),
        }
        if self._has_goal():
            base["goal_distance"]  = spaces.Box(0.0, 1.0,  (1,), np.float32)
            base["goal_direction"] = spaces.Box(-1.0, 1.0, (3,), np.float32)
        return spaces.Dict(base)

    def _action_space(self, agent: str) -> gymnasium.Space:
        return spaces.Discrete(26)

    def _fanout_obs(self, rust_obs: Any) -> dict:
        grid_size = self._config.get("grid_size", 32)
        norm = float(max(grid_size - 1, 1))
        radius = self._config.get("box_radius", 2)
        ray_max_len = self._config.get("ray_max_len", 16)
        use_euclidean = self._config.get("distance_metric", "euclidean") == "euclidean"

        goal_positions = self._rust_env.goal_positions() if self._has_goal() else []

        noise_prob   = self._config.get("obs_noise_prob", 0.0)
        cutout_count = self._config.get("obs_cutout_count", 0)
        cutout_size  = self._config.get("obs_cutout_size", 1)
        radius       = self._config.get("box_radius", 2)
        n            = 2 * radius + 1

        result = {}
        for i, agent_id in enumerate(self.agents):
            if i < len(rust_obs.cursors):
                cx, cy, cz = rust_obs.cursors[i]
                obs = {
                    "cursor_pos":     np.array(
                        [(cx - 1) / norm, (cy - 1) / norm, (cz - 1) / norm], dtype=np.float32
                    ),
                    "face_neighbors": np.array(
                        self._rust_env.face_neighbors(i), dtype=np.float32
                    ),
                    "local_grid":     np.array(
                        self._rust_env.box_obs(i, radius), dtype=np.float32
                    ),
                    "ray_cast":       np.array(
                        self._rust_env.ray_cast(i, ray_max_len), dtype=np.float32
                    ),
                }
                if self._has_goal() and i < len(goal_positions) and goal_positions[i] is not None:
                    gx, gy, gz = goal_positions[i]
                    dx, dy, dz = gx - cx, gy - cy, gz - cz
                    eucl = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if use_euclidean:
                        dist = eucl / _max_euclidean(grid_size)
                    else:
                        dist = (abs(dx) + abs(dy) + abs(dz)) / _max_manhattan(grid_size)
                    obs["goal_distance"] = np.array([dist], dtype=np.float32)
                    if eucl > 0.0:
                        obs["goal_direction"] = np.array(
                            [dx / eucl, dy / eucl, dz / eucl], dtype=np.float32
                        )
                    else:
                        obs["goal_direction"] = np.zeros(3, dtype=np.float32)
            else:
                obs = self._zero_obs(agent_id)

            if (noise_prob > 0.0 or cutout_count > 0) and "local_grid" in obs:
                obs["local_grid"] = self._augment_grid(obs["local_grid"], n, noise_prob, cutout_count, cutout_size)

            result[agent_id] = obs
        return result

    def _augment_grid(
        self,
        flat: np.ndarray,
        n: int,
        noise_prob: float,
        cutout_count: int,
        cutout_size: int,
    ) -> np.ndarray:
        grid = flat.reshape(n, n, n).copy()
        if noise_prob > 0.0:
            mask = self._obs_rng.random(grid.shape) < noise_prob
            grid[mask] = 1.0 - grid[mask]
        for _ in range(cutout_count):
            s = int(self._obs_rng.integers(1, max(2, cutout_size + 1)))
            x = int(self._obs_rng.integers(0, max(1, n - s + 1)))
            y = int(self._obs_rng.integers(0, max(1, n - s + 1)))
            z = int(self._obs_rng.integers(0, max(1, n - s + 1)))
            grid[x:x + s, y:y + s, z:z + s] = 0.0
        return grid.flatten()

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
