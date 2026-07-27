"""Single-agent Gymnasium voxel environment backed by the Rust core."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium
from gymnasium import spaces

from theseo_anysearch.environments.gymnasium.base import RustGymnasiumEnv

log = logging.getLogger(__name__)

MAX_RAY_HIT_TYPE = 5.0


def _max_manhattan(grid_size: int) -> float:
    """Maximum Manhattan distance in a cubic grid of given side length."""
    return 3.0 * (grid_size - 1)


class VoxelEnv(RustGymnasiumEnv):
    """
    Single-agent Gymnasium wrapper for the Rust VoxelEnv.

    Observation space: Dict(steps_remaining: Box(1,), voxel_count: Box(1,),
                            [goal_distance: Box(1,), goal_direction: Box(3,)]
                            when geometry is present)
    Action space: Discrete(26) — all 26 face/edge/corner neighbors in {-1,0,1}³

    Movement collision (boundary hit or occupied cell) cancels the move —
    cursor stays, step is consumed with step_cost only.
    """

    ray_env_id = "VoxelEnv-v0"

    def __init__(self, config: dict) -> None:
        self._obs_rng = np.random.default_rng(config.get("seed", 42))
        pool_config = (config.get("geometry_pool") or {})
        if pool_config.get("pool_dir"):
            from theseo_anysearch.environments.geometry_pool import GeometryPool
            self._geo_pool: "GeometryPool | None" = GeometryPool(
                pool_config["pool_dir"], seed=config.get("seed", 42)
            )
            self._augmentation_config: dict = pool_config.get("augmentation") or {}
        else:
            self._geo_pool = None
            self._augmentation_config = {}
        super().__init__(config)
        self._init_obs_cache(config)

    def _init_obs_cache(self, config: dict) -> None:
        """Cache config values and pre-allocate obs buffers. Called from __init__
        and can be called manually in tests that use VoxelEnv.__new__."""
        max_steps = max(config.get("max_steps", 200), 1)
        grid_size  = config.get("grid_size", 32)
        self._inv_max_steps     = 1.0 / max_steps
        self._inv_max_manhattan = 1.0 / _max_manhattan(grid_size)
        self._inv_norm          = 1.0 / float(max(grid_size - 1, 1))
        self._obs_mode          = config.get("obs_mode", "scalar")
        self._box_radius        = config.get("box_radius", 2)
        self._ray_max_len       = config.get("ray_max_len", 16)
        self._box_radii         = config.get("box_radii") or [1, 4]
        self._has_goal_flag     = self._has_goal()

        # Pre-allocate observation buffers — avoids per-step np.array([x]) allocation
        self._buf_steps = np.zeros(1, dtype=np.float32)
        self._buf_voxel = np.zeros(1, dtype=np.float32)
        self._buf_goal  = np.zeros(1, dtype=np.float32)
        self._buf_goal_direction = np.zeros(3, dtype=np.float32)
        if self._obs_mode != "scalar":
            self._buf_cursor = np.zeros(3, dtype=np.float32)
        if self._obs_mode == "box":
            n = 2 * self._box_radius + 1
            self._buf_grid = np.empty(n ** 3, dtype=np.float32)
        elif self._obs_mode == "radial":
            self._buf_rays = np.empty(26, dtype=np.float32)
            self._buf_ray_types = np.empty(26, dtype=np.float32)
        elif self._obs_mode == "hierarchical_box":
            flat_size = sum((2 * r + 1) ** 3 for r in self._box_radii)
            self._buf_grid = np.empty(flat_size, dtype=np.float32)

    @classmethod
    def register_with_ray(cls, env_config: dict) -> str:
        from ray.tune.registry import register_env

        register_env(cls.ray_env_id, lambda cfg: cls(cfg or env_config))
        return cls.ray_env_id

    def _build_rust_env(self, config: dict) -> Any:
        from theseo_anysearch.environments.pettingzoo.multi_voxel_env import _load_stl_geometry
        import theseo_core

        grid_size = config.get("grid_size", 32)
        geometry: list[tuple[int, int, int]] = []

        if config.get("stl_path"):
            scale = float(config.get("scale", 1.0))
            padding = int(config.get("geometry_padding", 2))
            geometry = _load_stl_geometry(str(config["stl_path"]), scale, grid_size, padding=padding)
        else:
            for box in config.get("geometry_boxes") or []:
                xmin, ymin, zmin, xmax, ymax, zmax = box
                for bx in range(xmin, xmax + 1):
                    for by in range(ymin, ymax + 1):
                        for bz in range(zmin, zmax + 1):
                            geometry.append((bx, by, bz))

        env = theseo_core.PyVoxelEnv(
            max_steps=config.get("max_steps", 200),
            trail_mode=config.get("trail_mode", True),
            geometry=geometry or None,
            grid_size=grid_size,
            step_cost=config.get("step_cost", -0.01),
            goal_reward=config.get("goal_reward", 1.0),
            distance_shaping=config.get("distance_shaping", 0.0),
            collision_cost=config.get("collision_cost", 0.0),
            distance_reward_mode=config.get("distance_reward_mode", "progress"),
            zone_reward_min=config.get("zone_reward_min", -1.0),
            zone_reward_max=config.get("zone_reward_max", -0.01),
            zone_reward_curve=config.get("zone_reward_curve", "linear"),
        )

        # Load fixed waypoints from file if specified.
        waypoints_file = config.get("waypoints_file")
        if waypoints_file:
            wp = self._load_waypoints(waypoints_file)
            if wp:
                env.set_waypoints(tuple(wp["start"]), tuple(wp["goal"]))

        return env

    @staticmethod
    def _load_waypoints(path: str) -> dict | None:
        """Load waypoints JSON: {"start": [x, y, z], "goal": [x, y, z]}."""
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path(os.getcwd()) / resolved
        try:
            data = json.loads(resolved.read_text())
            s, g = data["start"], data["goal"]
            return {"start": s, "goal": g}
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Could not load waypoints from %s: %s", path, exc
            )
            return None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self._geo_pool is not None:
            from theseo_anysearch.environments.geometry_pool import GeometryPool, paste_boxes
            grid = self._geo_pool.sample()
            paste_cfg = self._augmentation_config.get("paste_boxes")
            if paste_cfg:
                grid = paste_boxes(grid, paste_cfg, self._obs_rng)
            cells = GeometryPool.grid_to_cells(grid)
            self._rust_env.set_geometry(cells)
            log.debug("VoxelEnv reset: pool sample -> %d filled cells", len(cells))
            return super().reset(seed=seed, options=options)

        scale_range = self._config.get("scale_range")
        stl_path = self._config.get("stl_path")
        if scale_range and stl_path:
            lo, hi = float(scale_range[0]), float(scale_range[1])
            new_scale = float(self._obs_rng.uniform(lo, hi))
            log.debug("VoxelEnv reset: scale=%.1f (range [%.1f, %.1f])", new_scale, lo, hi)
            cfg = dict(self._config)
            cfg["scale"] = new_scale
            self._rust_env = self._build_rust_env(cfg)
        return super().reset(seed=seed, options=options)

    def _encode_action(self, action: Any) -> Any:
        return int(action)

    def _has_goal(self) -> bool:
        """True when geometry is configured so a goal can be selected."""
        return bool(
            self._config.get("geometry_boxes")
            or self._config.get("waypoints_file")
            or self._config.get("stl_path")
            or self._config.get("geometry_pool")
        )

    def _observation_space(self) -> gymnasium.Space:
        mode = self._config.get("obs_mode", "scalar")
        goal_space = (
            {
                "goal_distance": spaces.Box(0.0, 1.0, (1,), np.float32),
                "goal_direction": spaces.Box(-1.0, 1.0, (3,), np.float32),
            }
            if self._has_goal()
            else {}
        )

        voxel_count_space = (
            {"voxel_count": spaces.Box(0.0, np.inf, (1,), np.float32)}
            if self._config.get("include_voxel_count", True)
            else {}
        )

        if mode == "scalar":
            return spaces.Dict({
                "steps_remaining": spaces.Box(0.0, 1.0, (1,), np.float32),
                **voxel_count_space,
                **goal_space,
            })
        if mode == "box":
            n = 2 * self._config.get("box_radius", 2) + 1
            return spaces.Dict({
                "steps_remaining": spaces.Box(0.0, 1.0,   (1,),    np.float32),
                **voxel_count_space,
                "cursor_pos":      spaces.Box(0.0, 1.0,   (3,),    np.float32),
                "local_grid":      spaces.Box(0.0, 1.0,   (n**3,), np.float32),
                **goal_space,
            })
        if mode == "radial":
            return spaces.Dict({
                "steps_remaining": spaces.Box(0.0, 1.0,   (1,),  np.float32),
                **voxel_count_space,
                "cursor_pos":      spaces.Box(0.0, 1.0,   (3,),  np.float32),
                "ray_hits":        spaces.Box(0.0, 1.0,   (26,), np.float32),
                "ray_hit_types":   spaces.Box(0.0, 1.0,   (26,), np.float32),
                **goal_space,
            })
        if mode == "hierarchical_box":
            radii = self._config.get("box_radii") or [1, 4]
            flat_size = sum((2 * r + 1) ** 3 for r in radii)
            return spaces.Dict({
                "steps_remaining": spaces.Box(0.0, 1.0,    (1,),         np.float32),
                **voxel_count_space,
                "cursor_pos":      spaces.Box(0.0, 1.0,    (3,),         np.float32),
                "local_grid":      spaces.Box(0.0, 1.0,    (flat_size,), np.float32),
                **goal_space,
            })
        raise ValueError(
            f"Unknown obs_mode: {mode!r}. "
            "Expected 'scalar', 'box', 'radial', or 'hierarchical_box'."
        )

    def _action_space(self) -> gymnasium.Space:
        return spaces.Discrete(26)  # all 26 neighbors in {-1,0,1}³ \ {origin}

    def _obs_to_numpy(self, rust_obs: Any) -> dict:
        self._obs_log_count += 1
        if self._obs_log_count <= 5:
            self._log_env_stage(
                f"obs_to_numpy start index={self._obs_log_count} mode={self._obs_mode}"
            )
        # Write into pre-allocated buffers; copy before returning so RLlib's
        # sample collector (which holds per-step references) sees stable data.
        self._buf_steps[0] = rust_obs.steps_remaining * self._inv_max_steps
        self._buf_voxel[0] = rust_obs.filled
        base = {"steps_remaining": self._buf_steps.copy()}
        if self._config.get("include_voxel_count", True):
            base["voxel_count"] = self._buf_voxel.copy()
        if self._has_goal_flag and rust_obs.goal_distance is not None:
            self._buf_goal[0] = rust_obs.goal_distance * self._inv_max_manhattan
            base["goal_distance"] = self._buf_goal.copy()
            if rust_obs.goal_direction is not None:
                self._buf_goal_direction[:] = rust_obs.goal_direction
                base["goal_direction"] = self._buf_goal_direction.copy()

        if self._obs_mode == "scalar":
            return base

        cx, cy, cz = self._rust_env.cursor_pos()
        inv = self._inv_norm
        self._buf_cursor[0] = (cx - 1) * inv
        self._buf_cursor[1] = (cy - 1) * inv
        self._buf_cursor[2] = (cz - 1) * inv
        base["cursor_pos"] = self._buf_cursor.copy()

        if self._obs_mode == "box":
            self._buf_grid[:] = self._rust_env.box_obs(self._box_radius)
            base["local_grid"] = self._buf_grid.copy()
        elif self._obs_mode == "radial":
            self._buf_rays[:] = self._rust_env.radial_obs(self._ray_max_len)
            self._buf_ray_types[:] = self._rust_env.radial_obs_types(self._ray_max_len)
            self._buf_ray_types *= 1.0 / MAX_RAY_HIT_TYPE
            base["ray_hits"] = self._buf_rays.copy()
            base["ray_hit_types"] = self._buf_ray_types.copy()
        elif self._obs_mode == "hierarchical_box":
            offset = 0
            for r in self._box_radii:
                seg = self._rust_env.box_obs(r)
                n = (2 * r + 1) ** 3
                self._buf_grid[offset:offset + n] = seg
                offset += n
            base["local_grid"] = self._buf_grid.copy()
        else:
            raise ValueError(
                f"Unknown obs_mode: {self._obs_mode!r}. "
                "Expected 'scalar', 'box', 'radial', or 'hierarchical_box'."
            )
        if self._obs_log_count <= 5:
            self._log_env_stage(f"obs_to_numpy done index={self._obs_log_count}")
        return base
