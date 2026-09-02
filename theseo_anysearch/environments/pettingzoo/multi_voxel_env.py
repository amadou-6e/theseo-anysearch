"""Multi-agent voxel environment adapter backed by the Rust core."""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np
import gymnasium
from gymnasium import spaces

from theseo_anysearch.environments.action_spaces import build_action_space, encode_action

from theseo_anysearch.environments.pettingzoo.base import RustParallelEnv
from theseo_anysearch.worlds.extent import (
    maximum_euclidean,
    maximum_manhattan,
    resolve_task_extent,
)

log = logging.getLogger(__name__)
MAX_VOXEL_KIND = 5.0


def _stl_bounding_box(path: str) -> tuple[float, float, float, float]:
    """Return (max_extent, min_x, min_y, min_z) of an ASCII STL's vertices."""
    verts: list[list[float]] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("vertex"):
                parts = stripped.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    if not verts:
        return 1.0, 0.0, 0.0, 0.0
    arr = np.array(verts, dtype=np.float64)
    mins = arr.min(axis=0)
    extent = float((arr.max(axis=0) - mins).max())
    return (extent if extent > 0.0 else 1.0), float(mins[0]), float(mins[1]), float(mins[2])


def _load_stl_geometry(
    path: str, scale: float, grid_size: int, padding: int = 2
) -> list[tuple[int, int, int]]:
    """Voxelize an STL file and return the filled (solid) voxel coordinates.

    ``scale`` is the number of voxels the STL's longest axis should span.
    Values larger than ``grid_size - 2*padding - 1`` are silently clamped so
    the geometry always fits within the padded region.  The geometry is placed
    with ``padding`` free voxels on each side for agent circumnavigation.
    """
    import theseo_core
    max_extent, min_x, min_y, min_z = _stl_bounding_box(path)
    # scale = voxels the STL's longest axis should span in the grid.
    # Clamp so the geometry fits within the padded region.
    max_span = grid_size - 2 * padding - 1
    effective_scale = min(float(scale), float(max_span))
    voxels_per_unit = effective_scale / max_extent
    origin = float(padding + 1)
    sampler = theseo_core.PyVoxelSampler(grid_size=grid_size)
    sampler.load_stl_normalized(path, voxels_per_unit, origin, origin, origin)
    free = set(sampler.free_cells())
    # Clip to padded region — f32 barycentric rounding can create stray surface
    # voxels at padding, which corrupt solid_fill; exclude them explicitly.
    lo, hi = padding + 1, grid_size - padding
    return [
        (x, y, z)
        for x in range(lo, hi + 1)
        for y in range(lo, hi + 1)
        for z in range(lo, hi + 1)
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
        shared_validation = config.get("geometry_validation") or {}
        pool_validation = (
            (((config.get("geometry_pool") or {}).get("augmentation") or {}).get("feasibility"))
            or {}
        )
        if shared_validation.get("enabled", False) or (
            pool_validation and pool_validation.get("enabled", True)
        ):
            raise NotImplementedError(
                "geometry task-feasibility validation currently supports only "
                "single-agent VoxelEnv; joint multi-agent planning is not implemented"
            )
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

    @classmethod
    def register_with_ray(cls, env_config: dict) -> str:
        from ray.tune.registry import register_env
        register_env(cls.ray_env_id, lambda cfg: cls(cfg or env_config))
        return cls.ray_env_id

    def _init_possible_agents(self, config: dict) -> list[str]:
        agents = config.get("agents")
        if agents:
            return [str(agent["id"]) for agent in agents]
        n = config.get("agent_count", 2)
        return [f"agent_{i}" for i in range(n)]

    def _build_rust_env(self, config: dict) -> Any:
        import theseo_core

        extent = resolve_task_extent(config)
        grid_size = max(extent)
        geometry: list[tuple[int, int, int]] = []

        if config.get("compiled_world_path") is not None:
            geometry = []
        elif config.get("stl_path"):
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

        native_action_path = None
        native_manifest_path = config.get("native_extension_manifest")
        if native_manifest_path:
            from pathlib import Path
            from theseo_anysearch.experiments.native_extensions import NativeExtensionManifest

            manifest_path = Path(native_manifest_path)
            manifest = NativeExtensionManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            if {"predicate", "outcome"} & set(manifest.capabilities):
                native_action_path = str(
                    manifest_path.parent.joinpath(manifest.library).resolve()
                )

        env = theseo_core.PyMultiVoxelEnv(
            agent_count=config.get("agent_count", 2),
            max_steps=config.get("max_steps", 200),
            trail_mode=config.get("trail_mode", False),
            geometry=geometry or None,
            grid_size=grid_size,
            extent=extent,
            step_cost=config.get("step_cost", -0.01),
            goal_reward=config.get("goal_reward", 1.0),
            distance_shaping=config.get("distance_shaping", 0.1),
            collision_cost=config.get("collision_cost", 0.0),
            distance_reward_mode=config.get("distance_reward_mode", "progress"),
            zone_reward_min=config.get("zone_reward_min", -1.0),
            zone_reward_max=config.get("zone_reward_max", -0.01),
            zone_reward_curve=config.get("zone_reward_curve", "linear"),
            agents_json=json.dumps(config.get("agents")) if config.get("agents") else None,
            hunter_and_hunted_json=(
                json.dumps(config.get("hunter_and_hunted"))
                if config.get("hunter_and_hunted")
                else None
            ),
            native_action_path=native_action_path,
        )
        compiled_world_path = config.get("compiled_world_path")
        if compiled_world_path is not None:
            from pathlib import Path
            from theseo_anysearch.worlds.compiler import validate_compiled_world

            compiled = validate_compiled_world(Path(compiled_world_path).resolve())
            pack_extent = compiled.manifest.extent.as_tuple()
            if pack_extent != extent:
                raise ValueError(
                    f"configured extent {extent} does not match compiled world {pack_extent}"
                )
            env.set_compiled_world(
                str(compiled.root),
                int(config.get("world_maximum_decoded_bytes", 256 * 1024 * 1024)),
            )
            env.set_world_residency_radius(
                max(
                    int(config.get("box_radius", 2)),
                    int(config.get("ray_max_len", 16)),
                )
                + 2
                + int(config.get("world_prefetch_margin", 2))
            )
        return env

    def _has_goal(self) -> bool:
        return bool(
            self._config.get("geometry_boxes")
            or self._config.get("stl_path")
            or self._config.get("geometry_pool")
            or self._config.get("compiled_world_path")
        )

    def reset(self, seed: int | None = None, options: dict | None = None):
        if self._geo_pool is not None:
            from theseo_anysearch.environments.geometry_pool import GeometryPool, paste_boxes
            grid = self._geo_pool.sample()
            paste_cfg = self._augmentation_config.get("paste_boxes")
            if paste_cfg:
                grid = paste_boxes(grid, paste_cfg, self._obs_rng)
            cells = GeometryPool.grid_to_cells(grid)
            self._rust_env.set_geometry(cells)
            log.debug(
                "MultiVoxelEnv reset: pool sample -> %d filled cells", len(cells)
            )
            return super().reset(seed, options)

        scale_range = self._config.get("scale_range")
        stl_path = self._config.get("stl_path")
        if scale_range and stl_path:
            lo, hi = float(scale_range[0]), float(scale_range[1])
            new_scale = float(self._obs_rng.uniform(lo, hi))
            log.debug("MultiVoxelEnv reset: scale=%.1f (range [%.1f, %.1f])", new_scale, lo, hi)
            cfg = dict(self._config)
            cfg["scale"] = new_scale
            self._rust_env = self._build_rust_env(cfg)
        return super().reset(seed, options)

    def _observation_space(self, agent: str) -> gymnasium.Space:
        radius = self._config.get("box_radius", 2)
        n = 2 * radius + 1
        base = {
            "face_neighbors": spaces.Box(0.0, 1.0, (6,),    np.float32),
            "local_grid":     spaces.Box(0.0, 1.0, (n**3,), np.float32),
            "ray_cast":       spaces.Box(0.0, 1.0, (27,),   np.float32),
            "other_agent_vectors": spaces.Box(
                -1.0, 1.0, (3 * (len(self.possible_agents) - 1),), np.float32
            ),
        }
        if self._has_goal():
            base["goal_distance"]  = spaces.Box(0.0, 1.0,  (1,), np.float32)
            base["goal_direction"] = spaces.Box(-1.0, 1.0, (3,), np.float32)
        return spaces.Dict(base)

    def _action_space(self, agent: str) -> gymnasium.Space:
        agents = self._config.get("agents") or []
        selected = next((item for item in agents if item["id"] == agent), None)
        mode = selected["action_mode"] if selected else self._config.get("action_mode", "discrete_26")
        return build_action_space(mode)

    def _fanout_obs(self, rust_obs: Any) -> dict:
        extent = resolve_task_extent(self._config)
        radius = self._config.get("box_radius", 2)
        ray_max_len = self._config.get("ray_max_len", 16)
        use_euclidean = self._config.get("distance_metric", "euclidean") == "euclidean"
        norm = max(max(extent) - 1, 1)

        goal_positions = self._rust_env.goal_positions() if self._has_goal() else []

        result = {}
        for i, agent_id in enumerate(self.agents):
            if i < len(rust_obs.cursors):
                cx, cy, cz = rust_obs.cursors[i]
                obs = {
                    "face_neighbors": np.array(
                        self._rust_env.face_neighbors(i), dtype=np.float32
                    ),
                    "local_grid":     np.array(
                        self._rust_env.box_obs(i, radius), dtype=np.float32
                    ) / MAX_VOXEL_KIND,
                    "ray_cast":       np.array(
                        self._rust_env.ray_cast(i, ray_max_len), dtype=np.float32
                    ),
                    "other_agent_vectors": np.array(
                        [
                            component / norm
                            for other_index, (ox, oy, oz) in enumerate(rust_obs.cursors)
                            if other_index != i
                            for component in (ox - cx, oy - cy, oz - cz)
                        ],
                        dtype=np.float32,
                    ),
                }
                if self._has_goal() and i < len(goal_positions) and goal_positions[i] is not None:
                    gx, gy, gz = goal_positions[i]
                    dx, dy, dz = gx - cx, gy - cy, gz - cz
                    eucl = math.sqrt(dx * dx + dy * dy + dz * dz)
                    if use_euclidean:
                        dist = eucl / max(maximum_euclidean(extent), 1.0)
                    else:
                        dist = (abs(dx) + abs(dy) + abs(dz)) / max(
                            maximum_manhattan(extent), 1
                        )
                    obs["goal_distance"] = np.array([dist], dtype=np.float32)
                    if eucl > 0.0:
                        obs["goal_direction"] = np.array(
                            [dx / eucl, dy / eucl, dz / eucl], dtype=np.float32
                        )
                    else:
                        obs["goal_direction"] = np.zeros(3, dtype=np.float32)
            else:
                obs = self._zero_obs(agent_id)

            result[agent_id] = obs
        return result

    def _encode_actions(self, actions: dict) -> list[int]:
        configured = {
            item["id"]: item["action_mode"]
            for item in (self._config.get("agents") or [])
        }
        default = self._config.get("action_mode", "discrete_26")
        return [
            encode_action(actions.get(agent, 0), configured.get(agent, default))
            for agent in self.possible_agents
        ]

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
