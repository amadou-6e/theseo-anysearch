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

from theseo_anysearch.environments.action_spaces import (
    NOOP_ACTION_INDEX,
    build_action_space,
    encode_action,
)

from theseo_anysearch.environments.gymnasium.base import RustGymnasiumEnv
from theseo_anysearch.environments.task import (
    TaskConfig,
    goal_distance,
    goal_voxels,
    is_success,
)

log = logging.getLogger(__name__)

MAX_RAY_HIT_TYPE = 5.0


def _max_manhattan(grid_size: int) -> float:
    """Maximum Manhattan distance in a cubic grid of given side length."""
    return 3.0 * (grid_size - 1)


class VoxelEnv(RustGymnasiumEnv):
    """
    Single-agent Gymnasium wrapper for the Rust VoxelEnv.

    Observation space: Dict(steps_remaining: Box(1,),
                            [goal_distance: Box(1,), goal_direction: Box(3,)]
                            when geometry is present)
    Action space: Discrete(26) — all 26 face/edge/corner neighbors in {-1,0,1}³

    Movement collision (boundary hit or occupied cell) cancels the move —
    cursor stays, step is consumed with step_cost only.
    """

    ray_env_id = "VoxelEnv-v0"

    def __init__(self, config: dict) -> None:
        self._task = TaskConfig.model_validate(config.get("task") or {})
        from theseo_anysearch.experiments.custom_rewards import load_reward_provider

        reward_module_path = config.get("reward_module_path")
        self._reward_provider = load_reward_provider(
            Path(reward_module_path) if reward_module_path else None,
            config.get("custom_reward"),
        )
        self._reward_parameters = dict(config.get("custom_reward_parameters") or {})
        self._episode_steps = 0
        self._episode_reward_breakdown: dict[str, float] = {}
        self._initial_distance = 0.0
        self._minimum_distance = 0.0
        self._previous_task_distance = 0.0
        self._initial_filled: set[tuple[int, int, int]] = set()
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

        native_reward_path = None
        native_action_path = None
        native_manifest_path = config.get("native_extension_manifest")
        if native_manifest_path:
            from theseo_anysearch.experiments.native_extensions import (
                NativeExtensionManifest,
            )

            manifest_path = Path(native_manifest_path)
            manifest = NativeExtensionManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            library_path = str(
                manifest_path.parent.joinpath(manifest.library).resolve()
            )
            if "reward" in manifest.capabilities and config.get("custom_reward"):
                native_reward_path = library_path
            if {"predicate", "outcome"} & set(manifest.capabilities):
                native_action_path = library_path
        action_predicates = config.get("action_predicates")
        if action_predicates is None:
            action_predicates = [
                {"name": "valid_action"},
                {"name": "bounds"},
                {"name": "unoccupied"},
            ]
        action_outcomes = config.get("action_outcomes")
        if action_outcomes is None:
            action_outcomes = [{"name": "cursor_movement"}]
            if config.get("trail_mode", True):
                action_outcomes.append({"name": "trail_placement"})

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
            invalid_action_cost=config.get("invalid_action_cost", 0.0),
            construction_residual_weight=config.get("construction_residual_weight", 0.0),
            construction_overshoot_weight=config.get("construction_overshoot_weight", 0.0),
            construction_target=list(self._task.construction_target_voxels) or None,
            max_consecutive_collisions=self._task.max_consecutive_collisions,
            terminate_on_success=self._task.termination.terminate_on_success,
            success_targets=list(goal_voxels(self._task.goal, None)) or None,
            goal_tolerance=float(getattr(self._task.goal, "tolerance", 0.0)),
            native_reward_path=native_reward_path,
            custom_reward=config.get("custom_reward"),
            custom_reward_parameters_json=json.dumps(
                config.get("custom_reward_parameters") or {},
                separators=(",", ":"),
                sort_keys=True,
            ),
            native_action_path=native_action_path,
            action_predicates_json=json.dumps(
                action_predicates, separators=(",", ":")
            ),
            action_outcomes_json=json.dumps(
                action_outcomes, separators=(",", ":")
            ),
            action_history_length=int(config.get("action_history_length", 16)),
            box_radius=(
                config.get("box_radius", 2)
                if config.get("obs_mode", "scalar") == "box"
                else None
            ),
        )

        # Load fixed waypoints from file if specified.
        wp = None
        waypoints_file = config.get("waypoints_file")
        if waypoints_file:
            wp = self._load_waypoints(waypoints_file)
            if wp:
                env.set_waypoints(tuple(wp["start"]), tuple(wp["goal"]))

        configured_targets = goal_voxels(self._task.goal, None)
        if configured_targets:
            if not wp:
                raise ValueError("A configured task goal requires waypoints_file to provide the episode start")
            env.set_waypoints(tuple(wp["start"]), configured_targets[0])

        return env

    @staticmethod
    def _load_waypoints(path: str) -> dict | None:
        """Load waypoints JSON: {"start": [x, y, z], "goal": [x, y, z]}.

        Only called when ``waypoints_file`` is configured (non-empty), so any
        failure here means the user pointed at a file that is missing,
        unreadable, or malformed — that is a configuration error and must
        raise rather than be silently swallowed as "no waypoints configured".
        """
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = Path(os.getcwd()) / resolved

        if not resolved.exists():
            raise FileNotFoundError(
                f"waypoints_file {path!r} (resolved to {resolved}) does not exist."
            )

        try:
            raw_text = resolved.read_text()
        except OSError as exc:
            raise ValueError(
                f"waypoints_file {path!r} (resolved to {resolved}) could not be read: {exc}"
            ) from exc

        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"waypoints_file {path!r} (resolved to {resolved}) is not valid JSON: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"waypoints_file {path!r} (resolved to {resolved}) must contain a JSON "
                f"object with 'start' and 'goal' keys, got {type(data).__name__}."
            )

        missing = [key for key in ("start", "goal") if key not in data]
        if missing:
            raise ValueError(
                f"waypoints_file {path!r} (resolved to {resolved}) is missing required "
                f"key(s): {', '.join(missing)}. Expected schema: "
                '{"start": [x, y, z], "goal": [x, y, z]}.'
            )

        s, g = data["start"], data["goal"]
        for name, value in (("start", s), ("goal", g)):
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 3
                or not all(isinstance(v, (int, float)) for v in value)
            ):
                raise ValueError(
                    f"waypoints_file {path!r} (resolved to {resolved}) has an invalid "
                    f"'{name}' value: {value!r}. Expected a 3-element [x, y, z] coordinate."
                )

        return {"start": s, "goal": g}

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
            return self._reset_task_state(super().reset(seed=seed, options=options))

        scale_range = self._config.get("scale_range")
        stl_path = self._config.get("stl_path")
        if scale_range and stl_path:
            lo, hi = float(scale_range[0]), float(scale_range[1])
            new_scale = float(self._obs_rng.uniform(lo, hi))
            log.debug("VoxelEnv reset: scale=%.1f (range [%.1f, %.1f])", new_scale, lo, hi)
            cfg = dict(self._config)
            cfg["scale"] = new_scale
            self._rust_env = self._build_rust_env(cfg)
        return self._reset_task_state(super().reset(seed=seed, options=options))

    def _reset_task_state(self, reset_result):
        """Initialize episode-level task metrics after a Rust reset."""

        observation, _ = reset_result
        cursor = tuple(self._rust_env.cursor_pos())
        fallback = self._rust_env.goal_pos()
        distance = goal_distance(self._task.goal, cursor, fallback)
        self._episode_steps = 0
        self._initial_distance = distance
        self._minimum_distance = distance
        self._previous_task_distance = distance
        self._episode_reward_breakdown = {}
        self._initial_filled = {tuple(coord) for coord in self._rust_env.filled_voxels()}
        self._last_observation = observation
        return observation, {
            "task_version": self._task.version,
            "initial_goal_distance": distance,
        }

    def step(self, action):
        """Apply one action and expose task-owned reward and termination data."""

        previous_observation = self._last_observation
        invalid_action = not self.action_space.contains(action)
        action_index = self._encode_action(action)
        previous_cursor = tuple(self._rust_env.cursor_pos())
        result = self._rust_env.step(action_index)
        observation = self._obs_to_numpy(result.observation)
        cursor = tuple(self._rust_env.cursor_pos())
        fallback = self._rust_env.goal_pos()
        current_distance = float(result.goal_distance_l2)
        collision = bool(result.collision)
        success = bool(result.goal_reached)
        breakdown = dict(result.reward_breakdown)
        standard_reward = float(result.reward)
        self._episode_steps += 1
        terminated = bool(result.terminated)
        truncated = bool(result.truncated)

        residual = int(result.construction_residual)
        overshoot = int(result.construction_overshoot)
        reward = standard_reward
        if self._reward_provider is not None:
            from theseo_anysearch.experiments.custom_rewards import (
                RewardContext,
                apply_custom_reward,
            )

            raw_goal = fallback
            goal = tuple(raw_goal) if raw_goal is not None else None
            reward_context = RewardContext(
                step=self._episode_steps,
                action=action,
                action_index=int(action_index),
                previous_observation=previous_observation,
                observation=observation,
                previous_cursor=previous_cursor,
                cursor=cursor,
                goal=goal,
                previous_goal_distance=self._previous_task_distance,
                goal_distance=current_distance,
                invalid_action=invalid_action,
                collision=collision,
                terminated=terminated,
                truncated=truncated,
                standard_reward=standard_reward,
                standard_breakdown=breakdown,
                env_config=dict(self._config),
                parameters=dict(self._reward_parameters),
                info={
                    "success": success,
                    "construction_residual": residual,
                    "construction_overshoot": overshoot,
                    "consecutive_collisions": int(result.consecutive_collisions),
                },
            )
            reward, breakdown = apply_custom_reward(
                self._reward_provider, reward_context
            )
        unshaped_reward = reward - breakdown.get("distance_progress", 0.0)
        for name, value in breakdown.items():
            self._episode_reward_breakdown[name] = (
                self._episode_reward_breakdown.get(name, 0.0) + value
            )

        self._minimum_distance = min(self._minimum_distance, current_distance)
        self._previous_task_distance = current_distance
        self._last_observation = observation
        reason = str(result.termination_reason)
        info = {
            "task_version": self._task.version,
            "goal_reached": success,
            "termination_reason": reason,
            "reward_breakdown": breakdown,
            "episode_reward_breakdown": dict(self._episode_reward_breakdown),
            "unshaped_reward": unshaped_reward,
            "initial_goal_distance": self._initial_distance,
            "final_goal_distance": current_distance,
            "minimum_goal_distance": self._minimum_distance,
            "invalid_action": invalid_action,
            "collision": collision,
            "consecutive_collisions": int(result.consecutive_collisions),
            "construction_residual": residual,
            "construction_overshoot": overshoot,
        }
        return observation, reward, terminated, truncated, info

    def action_mask(self) -> np.ndarray:
        """Return the configured action-space mask from Rust predicates."""
        raw_mask = self._rust_env.action_mask()
        canonical = (
            np.frombuffer(raw_mask, dtype=np.uint8).astype(np.int8)
            if isinstance(raw_mask, (bytes, bytearray))
            else np.asarray(raw_mask, dtype=np.int8)
        )
        mode = self._config.get("action_mode", "discrete_26")
        if mode == "vector_3":
            return canonical
        indices = [encode_action(index, mode) for index in range(self.action_space.n)]
        return canonical[indices]
    def _encode_action(self, action: Any) -> Any:
        return encode_action(action, self._config.get("action_mode", "discrete_26"))

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

        if mode == "scalar":
            return spaces.Dict({
                "steps_remaining": spaces.Box(0.0, 1.0, (1,), np.float32),
                **goal_space,
            })
        if mode == "box":
            n = 2 * self._config.get("box_radius", 2) + 1
            return spaces.Dict({
                "steps_remaining": spaces.Box(0.0, 1.0,   (1,),    np.float32),
                "cursor_pos":      spaces.Box(0.0, 1.0,   (3,),    np.float32),
                "local_grid":      spaces.Box(0.0, 1.0,   (n**3,), np.float32),
                **goal_space,
            })
        if mode == "radial":
            return spaces.Dict({
                "steps_remaining": spaces.Box(0.0, 1.0,   (1,),  np.float32),
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
                "cursor_pos":      spaces.Box(0.0, 1.0,    (3,),         np.float32),
                "local_grid":      spaces.Box(0.0, 1.0,    (flat_size,), np.float32),
                **goal_space,
            })
        raise ValueError(
            f"Unknown obs_mode: {mode!r}. "
            "Expected 'scalar', 'box', 'radial', or 'hierarchical_box'."
        )

    def _action_space(self) -> gymnasium.Space:
        return build_action_space(self._config.get("action_mode", "discrete_26"))

    def _obs_to_numpy(self, rust_obs: Any) -> dict:
        self._obs_log_count = getattr(self, "_obs_log_count", 0) + 1
        if self._obs_log_count <= 5:
            self._log_env_stage(
                f"obs_to_numpy start index={self._obs_log_count} mode={self._obs_mode}"
            )
        # Write into pre-allocated buffers; copy before returning so RLlib's
        # sample collector (which holds per-step references) sees stable data.
        self._buf_steps[0] = rust_obs.steps_remaining * self._inv_max_steps
        base = {"steps_remaining": self._buf_steps.copy()}
        if self._has_goal_flag and rust_obs.goal_distance is not None:
            self._buf_goal[0] = rust_obs.goal_distance * self._inv_max_manhattan
            base["goal_distance"] = self._buf_goal.copy()
            if rust_obs.goal_direction is not None:
                self._buf_goal_direction[:] = rust_obs.goal_direction
                base["goal_direction"] = self._buf_goal_direction.copy()

        if self._obs_mode == "scalar":
            return base

        cx, cy, cz = rust_obs.cursor_pos
        inv = self._inv_norm
        self._buf_cursor[0] = (cx - 1) * inv
        self._buf_cursor[1] = (cy - 1) * inv
        self._buf_cursor[2] = (cz - 1) * inv
        base["cursor_pos"] = self._buf_cursor.copy()

        if self._obs_mode == "box":
            local_grid = rust_obs.local_grid
            if local_grid is None:
                local_grid = self._rust_env.box_obs(self._box_radius)
            self._buf_grid[:] = local_grid
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
