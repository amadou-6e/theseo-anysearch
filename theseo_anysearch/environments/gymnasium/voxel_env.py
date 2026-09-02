"""Single-agent Gymnasium voxel environment backed by the Rust core."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium
import networkx as nx
from gymnasium import spaces

from theseo_anysearch.environments.action_spaces import (
    ACTION_OFFSETS_26,
    NOOP_ACTION_INDEX,
    action_step_distance,
    build_action_space,
    encode_action,
    maximum_movement_distance,
    offsets_for_mode,
)

from theseo_anysearch.environments.gymnasium.base import RustGymnasiumEnv
from theseo_anysearch.environments.task import (
    TaskConfig,
    goal_distance,
    goal_voxels,
    is_success,
)
from theseo_anysearch.worlds.extent import maximum_manhattan, resolve_task_extent

log = logging.getLogger(__name__)

MAX_RAY_HIT_TYPE = 5.0
MAX_VOXEL_KIND = 5.0


class VoxelEnv(RustGymnasiumEnv):
    """
    Single-agent Gymnasium wrapper for the Rust VoxelEnv.

    Observation space: Dict([goal_distance: Box(1,), goal_direction: Box(3,)]
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
        from theseo_anysearch.environments.lifecycle import build_lifecycle_rules

        self._lifecycle_rules = build_lifecycle_rules(
            config.get("lifecycle_rules") or [{"name": "native"}]
        )
        self._episode_steps = 0
        self._consecutive_collisions = 0
        self._episode_reward_breakdown: dict[str, float] = {}
        self._initial_distance = 0.0
        self._minimum_distance = 0.0
        self._previous_task_distance = 0.0
        self._initial_filled: set[tuple[int, int, int]] = set()
        self._obs_rng = np.random.default_rng(config.get("seed", 42))
        self._pending_waypoints: tuple[
            tuple[int, int, int], tuple[int, int, int]
        ] | None = None
        self._pending_route: dict[str, Any] | None = None
        self._configured_route: dict[str, Any] | None = config.get("waypoint_route")
        self._route_remaining: list[tuple[int, int, int]] = []
        self._route_waypoint_count = 0
        self._route_waypoints_reached = 0
        self._active_start: tuple[int, int, int] | None = None
        self._active_goal: tuple[int, int, int] | None = None
        self._curriculum_stages: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
        self._curriculum_stage_probabilities: list[float] = []
        self._scenario_provider = None
        self._scenario_parameters = dict(config.get("scenario_parameters") or {})
        self._scenario_scope = str(config.get("scenario_scope", "training"))
        self._previous_scenario: dict[str, Any] | None = None
        self._scenario_geometry: tuple[tuple[int, int, int], ...] = ()
        self._last_feasibility_diagnostics: dict[str, Any] | None = None
        self._last_accepted_task_manifest: dict[str, Any] | None = None
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
        shared_validation = config.get("geometry_validation") or {}
        legacy_validation = self._augmentation_config.get("feasibility") or {}
        self._validation_config = (
            shared_validation
            if shared_validation.get("enabled", False)
            else legacy_validation
        )
        super().__init__(config)
        self._scenario_provider = self._load_scenario_provider(config)
        self._init_obs_cache(config)

    def _load_scenario_provider(self, config: dict):
        """Resolve the selected Python or Rust scenario without fallback."""
        name = config.get("scenario_provider")
        if not name:
            return None
        from theseo_anysearch.experiments.custom_scenarios import (
            load_native_scenario_provider,
            load_scenario_provider,
        )

        native_manifest_path = config.get("native_extension_manifest")
        if native_manifest_path:
            from theseo_anysearch.experiments.native_extensions import NativeExtensionManifest

            manifest_path = Path(native_manifest_path)
            manifest = NativeExtensionManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            if name in manifest.scenarios:
                return load_native_scenario_provider(
                    manifest_path.parent.joinpath(manifest.library).resolve(), name
                )
        source = config.get("scenario_module_path")
        provider = load_scenario_provider(Path(source) if source else None, name)
        if provider is None:
            raise ValueError(
                f"scenario provider {name!r} has no compiled Rust export or scenarios.py source"
            )
        return provider

    def _init_obs_cache(self, config: dict) -> None:
        """Cache config values and pre-allocate obs buffers. Called from __init__
        and can be called manually in tests that use VoxelEnv.__new__."""
        extent = resolve_task_extent(config)
        self._extent = extent
        self._inv_max_manhattan = 1.0 / max(maximum_manhattan(extent), 1)
        self._obs_mode          = config.get("obs_mode", "scalar")
        self._box_radius        = config.get("box_radius", 2)
        self._ray_max_len       = config.get("ray_max_len", 16)
        self._box_radii         = config.get("box_radii") or [1, 4]
        self._has_goal_flag     = self._has_goal()

        # Pre-allocate observation buffers — avoids per-step np.array([x]) allocation
        self._buf_goal  = np.zeros(1, dtype=np.float32)
        self._buf_goal_direction = np.zeros(3, dtype=np.float32)
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

        extent = resolve_task_extent(config)
        grid_size = max(extent)
        geometry: list[tuple[int, int, int]] = []

        if config.get("compiled_world_path") is not None:
            # The immutable base is attached from the pack below. Never expand
            # its source boxes/STL back into Python coordinate tuples.
            geometry = []
        elif config.get("geometry_sources"):
            from theseo_anysearch.environments.geometry_sources import (
                resolve_geometry_sources,
            )

            geometry = resolve_geometry_sources(
                config, grid_size=grid_size, load_stl=_load_stl_geometry
            )
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
        self._scenario_geometry = tuple(geometry)

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
            extent=extent,
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
        compiled_world_path = config.get("compiled_world_path")
        if compiled_world_path is not None:
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
            observation_radius = {
                "box": int(config.get("box_radius", 2)),
                "radial": int(config.get("ray_max_len", 16)),
                "hierarchical_box": max(config.get("box_radii") or [1, 4]),
            }.get(config.get("obs_mode", "scalar"), 1)
            movement_radius = int(np.ceil(maximum_movement_distance(config.get("action_mode", "discrete_26"))))
            env.set_world_residency_radius(
                observation_radius
                + movement_radius
                + int(config.get("world_prefetch_margin", 2))
            )

        # Load fixed or curriculum waypoints if specified.
        wp = None
        configured_route = config.get("waypoint_route")
        if configured_route:
            route_waypoints = [tuple(item) for item in configured_route["waypoints"]]
            if not route_waypoints:
                raise ValueError("waypoint_route requires at least one waypoint")
            wp = {
                "start": tuple(configured_route["start"]),
                "goal": route_waypoints[0],
            }
            self._route_remaining = route_waypoints[1:]
            self._route_waypoint_count = len(route_waypoints)
        inline_waypoints = config.get("waypoints")
        curriculum = config.get("waypoint_curriculum") or {}
        if wp is None and not inline_waypoints and curriculum.get("enabled"):
            inline_waypoints = {
                "start": curriculum.get("initial_start"),
                "goal": curriculum.get("initial_goal"),
            }
        if (
            inline_waypoints
            and inline_waypoints.get("start")
            and inline_waypoints.get("goal")
        ):
            wp = inline_waypoints
        waypoints_file = config.get("waypoints_file")
        if wp is None and waypoints_file:
            wp = self._load_waypoints(waypoints_file)
        if wp:
            start = tuple(wp["start"])
            goal = tuple(wp["goal"])
            self._active_start = start
            self._active_goal = goal
            env.set_waypoints(
                start,
                goal,
                action_step_distance(
                    start, goal, config.get("action_mode", "discrete_26")
                ),
            )

        configured_targets = goal_voxels(self._task.goal, None)
        if configured_targets:
            if not wp:
                raise ValueError("A configured task goal requires waypoints_file to provide the episode start")
            start = tuple(wp["start"])
            goal = configured_targets[0]
            self._active_start = start
            self._active_goal = goal
            env.set_waypoints(
                start,
                goal,
                action_step_distance(
                    start, goal, config.get("action_mode", "discrete_26")
                ),
            )

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
            resolved = Path(os.getcwd(), resolved)

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
        self._last_accepted_task_manifest = None
        if self._geo_pool is not None:
            from theseo_anysearch.environments.geometry_pool import GeometryPool, paste_boxes
            configured_feasibility = self._validation_config
            feasibility = (
                configured_feasibility
                if configured_feasibility
                and configured_feasibility.get("enabled", True)
                else None
            )
            maximum_attempts = int((feasibility or {}).get("maximum_attempts", 1))
            if feasibility and maximum_attempts < 1:
                raise ValueError("geometry_pool.augmentation.feasibility.maximum_attempts must be positive")
            if feasibility and int(feasibility.get("maximum_search_nodes", 0)) < 1:
                raise ValueError(
                    "geometry_pool.augmentation.feasibility.maximum_search_nodes must be positive"
                )
            if feasibility and int(feasibility.get("recovery_margin_steps", 0)) < 0:
                raise ValueError(
                    "geometry_pool.augmentation.feasibility.recovery_margin_steps "
                    "must be non-negative"
                )
            rejections: dict[str, int] = {}
            accepted_plan_steps: int | None = None
            reset_seed = (
                int(seed)
                if seed is not None
                else int(self._config.get("seed", 42)) + self._reset_count + 1
            )
            if feasibility:
                self._obs_rng = np.random.default_rng(reset_seed)
            configured_waypoints = self._config.get("waypoints")
            curriculum = self._config.get("waypoint_curriculum") or {}
            if not configured_waypoints and curriculum.get("enabled"):
                configured_waypoints = {
                    "start": curriculum.get("initial_start"),
                    "goal": curriculum.get("initial_goal"),
                }
            if not configured_waypoints and self._config.get("waypoints_file"):
                configured_waypoints = self._load_waypoints(
                    self._config["waypoints_file"]
                )
            configured_start = (
                tuple(configured_waypoints["start"])
                if configured_waypoints and configured_waypoints.get("start")
                else None
            )
            configured_goal = (
                tuple(configured_waypoints["goal"])
                if configured_waypoints and configured_waypoints.get("goal")
                else None
            )
            task_targets = goal_voxels(self._task.goal, None)
            if task_targets:
                configured_goal = tuple(task_targets[0])
            for attempt in range(1, maximum_attempts + 1):
                grid = self._geo_pool.sample(
                    rng=self._obs_rng if feasibility else None
                ).copy()
                paste_cfg = self._augmentation_config.get("paste_boxes")
                if paste_cfg:
                    grid = paste_boxes(grid, paste_cfg, self._obs_rng)
                cells = GeometryPool.grid_to_cells(grid)
                self._rust_env.set_geometry(cells)
                self._scenario_geometry = tuple(cells)
                if feasibility:
                    geometry_result = self._geometry_validation_result()
                    if not geometry_result.valid:
                        reason = str(geometry_result.rejection_reason)
                        rejections[reason] = rejections.get(reason, 0) + 1
                        continue
                log.debug("VoxelEnv reset: pool sample -> %d filled cells", len(cells))
                if self._configured_route:
                    self._activate_route(self._configured_route)
                elif configured_start is not None and configured_goal is not None:
                    self._rust_env.set_waypoints(
                        configured_start,
                        configured_goal,
                        self._segment_length(configured_start, configured_goal),
                    )
                self._apply_pending_waypoints()
                self._apply_pending_route()
                self._sample_curriculum_waypoints()
                self._apply_scenario(seed)
                if feasibility:
                    feasibility_result = self._task_feasibility_result(feasibility)
                    if not feasibility_result.feasible:
                        reason = str(feasibility_result.rejection_reason)
                        rejections[reason] = rejections.get(reason, 0) + 1
                        continue
                    accepted_plan_steps = feasibility_result.path_length
                rust_observation = self._rust_env.reset(reset_seed)
                reset_result = (self._obs_to_numpy(rust_observation), {})
                break
            else:
                raise RuntimeError(
                    "augmented task feasibility exhausted after "
                    f"{maximum_attempts} attempts; rejections={rejections}"
                )
            if feasibility:
                self._last_feasibility_diagnostics = {
                    "enabled": True,
                    "attempts": attempt,
                    "rejections": rejections,
                    "accepted_plan_steps": accepted_plan_steps,
                    "routing_difficulty": (
                        feasibility_result.difficulty.model_dump(mode="json")
                        if feasibility_result.difficulty is not None
                        else None
                    ),
                    "difficulty_band": feasibility_result.difficulty_band,
                }
                self._record_accepted_task_manifest(
                    reset_seed, geometry_result, feasibility_result
                )
            self._reset_count += 1
            return self._reset_task_state(reset_result)

        scale_range = self._config.get("scale_range")
        stl_path = self._config.get("stl_path")
        if scale_range and stl_path:
            lo, hi = float(scale_range[0]), float(scale_range[1])
            new_scale = float(self._obs_rng.uniform(lo, hi))
            log.debug("VoxelEnv reset: scale=%.1f (range [%.1f, %.1f])", new_scale, lo, hi)
            cfg = dict(self._config)
            cfg["scale"] = new_scale
            self._rust_env = self._build_rust_env(cfg)
        if self._configured_route:
            self._activate_route(self._configured_route)
        self._apply_pending_waypoints()
        self._apply_pending_route()
        self._sample_curriculum_waypoints()
        self._apply_scenario(seed)
        if self._validation_config.get("enabled", False):
            geometry_result = self._geometry_validation_result()
            if not geometry_result.valid:
                raise RuntimeError(
                    "geometry validation failed: "
                    f"{geometry_result.rejection_reason} at "
                    f"{geometry_result.rejected_coordinate}"
                )
            feasibility_result = self._task_feasibility_result(
                self._validation_config
            )
            if not feasibility_result.feasible:
                raise RuntimeError(
                    "task feasibility failed: "
                    f"{feasibility_result.rejection_reason}"
                )
            self._last_feasibility_diagnostics = {
                "enabled": True,
                "attempts": 1,
                "rejections": {},
                "accepted_plan_steps": feasibility_result.path_length,
                "routing_difficulty": (
                    feasibility_result.difficulty.model_dump(mode="json")
                    if feasibility_result.difficulty is not None
                    else None
                ),
                "difficulty_band": feasibility_result.difficulty_band,
            }
            reset_seed = (
                int(seed)
                if seed is not None
                else int(self._config.get("seed", 42)) + self._reset_count + 1
            )
            self._record_accepted_task_manifest(
                reset_seed, geometry_result, feasibility_result
            )
        return self._reset_task_state(super().reset(seed=seed, options=options))

    def _record_accepted_task_manifest(
        self, reset_seed, geometry_result, feasibility_result
    ) -> None:
        """Capture portable identity and validation evidence for this reset."""
        from theseo_anysearch.environments.task_identity import accepted_task_manifest

        route = []
        if self._active_goal is not None:
            route.append(tuple(self._active_goal))
        route.extend(tuple(item) for item in self._route_remaining)
        manifest = accepted_task_manifest(
            coordinates=self._scenario_geometry,
            geometry_identity_sha256=self._config.get("world_identity_sha256"),
            seed=int(reset_seed),
            start=tuple(self._active_start or self._rust_env.cursor_pos()),
            route=route,
            action_mode=str(self._config.get("action_mode", "discrete_26")),
            transformations=dict(self._augmentation_config),
            planner_settings={
                key: self._validation_config.get(key)
                for key in (
                    "maximum_search_nodes",
                    "recovery_margin_steps",
                    "clearance_radius",
                    "difficulty_bands",
                    "accepted_difficulty_bands",
                )
            },
            geometry_validation=geometry_result,
            task_feasibility=feasibility_result,
        )
        self._last_accepted_task_manifest = manifest.model_dump(mode="json")

    def _geometry_validation_result(self):
        from theseo_anysearch.environments.validation import validate_geometry

        return validate_geometry(self._scenario_geometry, resolve_task_extent(self._config))

    def _task_feasibility_result(self, config: dict[str, Any]):
        from theseo_anysearch.environments.validation import (
            BoundedWorldRead,
            validate_task_feasibility,
        )
        from theseo_anysearch.settings.environment.geometry import RoutingDifficultyBand

        action_mode = self._config.get("action_mode", "discrete_26")
        directions = (
            ACTION_OFFSETS_26
            if action_mode == "vector_3"
            else offsets_for_mode(action_mode)
        )
        return validate_task_feasibility(
            BoundedWorldRead(self._rust_env.world_occupied),
            start=self._active_start,
            goal=self._active_goal,
            extent=resolve_task_extent(self._config),
            directions=directions,
            action_mode=action_mode,
            maximum_search_nodes=int(config["maximum_search_nodes"]),
            maximum_steps=int(self._config.get("max_steps", 200)),
            recovery_margin_steps=int(config.get("recovery_margin_steps", 0)),
            clearance_radius=config.get("clearance_radius"),
            difficulty_bands=tuple(
                RoutingDifficultyBand.model_validate(item)
                for item in config.get("difficulty_bands", ())
            ),
            accepted_difficulty_bands=tuple(
                config.get("accepted_difficulty_bands", ())
            ),
        )

    def _apply_scenario(self, seed: int | None) -> None:
        """Invoke the configured reset hook and install its validated route."""
        if self._scenario_provider is None:
            return
        from theseo_anysearch.experiments.custom_scenarios import (
            ScenarioContext,
            validate_scenario,
        )

        class _WorldView:
            def __init__(self, rust_env, extent, identity, maximum_queries, maximum_results):
                self.rust_env = rust_env
                self.extent = extent
                self.identity = identity
                self.maximum_queries = maximum_queries
                self.maximum_results = maximum_results
                self.queries = 0
                self.results = 0

            def _consume(self, results=0):
                self.queries += 1
                self.results += results
                if self.queries > self.maximum_queries or self.results > self.maximum_results:
                    raise RuntimeError("scenario world query budget exhausted")

            def occupied(self, coordinate: tuple[int, int, int]) -> bool:
                self._consume()
                return bool(self.rust_env.world_occupied(coordinate))

            def occupied_in_region(self, minimum, maximum_exclusive):
                remaining = self.maximum_results - self.results
                values = tuple(
                    self.rust_env.world_occupied_in_region(
                        minimum, maximum_exclusive, remaining
                    )
                )
                self._consume(len(values))
                return values

        episode_index = self._reset_count
        resolved_seed = (
            int(seed)
            if seed is not None
            else int(self._config.get("seed", 42)) + episode_index + 1
        )
        action_mode = self._config.get("action_mode", "discrete_26")
        offsets = (
            ACTION_OFFSETS_26
            if action_mode == "vector_3"
            else offsets_for_mode(action_mode)
        )
        extent = resolve_task_extent(self._config)
        candidates = None
        candidate_root = self._config.get("scenario_candidate_index")
        world_identity = self._config.get("world_identity_sha256")
        if candidate_root is not None:
            from theseo_anysearch.worlds.candidates import (
                CandidateIndexHandle,
                CandidateQueryBudget,
            )

            candidates = CandidateIndexHandle(
                Path(candidate_root),
                world_identity=world_identity,
                budget=CandidateQueryBudget(
                    maximum_queries=int(
                        self._config.get("scenario_maximum_candidate_queries", 64)
                    ),
                    maximum_results=int(
                        self._config.get("scenario_maximum_candidate_results", 4096)
                    ),
                ),
            )
            world_identity = candidates.world_identity
        world = _WorldView(
            self._rust_env,
            extent,
            world_identity,
            int(self._config.get("scenario_maximum_candidate_queries", 64)),
            int(self._config.get("scenario_maximum_candidate_results", 4096)),
        )
        if self._scenario_provider.native_abi == 2:
            raw = self._rust_env.generate_native_scenario_v2(
                str(self._scenario_provider.source_path),
                self._scenario_provider.name,
                resolved_seed,
                episode_index,
                self._scenario_scope,
                action_mode,
                json.dumps(offsets),
                json.dumps(self._previous_scenario),
                json.dumps(dict(self._config.get("waypoint_curriculum") or {})),
                json.dumps(self._scenario_parameters),
                str(candidate_root) if candidate_root is not None else None,
                world.identity,
            )
            from theseo_anysearch.experiments.custom_scenarios import ScenarioResult

            generated = ScenarioResult.model_validate_json(raw)
        else:
            context = ScenarioContext(
                seed=resolved_seed,
                episode_index=episode_index,
                scope=self._scenario_scope,
                extent=extent,
                world_identity=world.identity,
                world=world,
                candidates=candidates,
                action_mode=action_mode,
                action_offsets=offsets,
                previous_scenario=self._previous_scenario,
                curriculum=dict(self._config.get("waypoint_curriculum") or {}),
                parameters=self._scenario_parameters,
            )
            if self._scenario_provider.generate is None:
                raise RuntimeError("scenario provider has no executable implementation")
            generated = self._scenario_provider.generate(context)
        scenario = validate_scenario(
            generated,
            extent=extent,
            world=world,
        )
        self._activate_route({"start": scenario.start, "waypoints": scenario.waypoints})
        self._previous_scenario = {
            "scenario_id": scenario.scenario_id,
            "start": scenario.start,
            "waypoints": scenario.waypoints,
            "metadata": scenario.metadata,
        }

    def _segment_length(
        self,
        start: tuple[int, int, int],
        goal: tuple[int, int, int],
    ) -> int:
        return action_step_distance(
            start,
            goal,
            self._config.get("action_mode", "discrete_26"),
        )

    def queue_waypoints(
        self,
        start: tuple[int, int, int],
        goal: tuple[int, int, int],
    ) -> None:
        """Apply a trainer-broadcast waypoint pair on the next episode reset."""
        self._pending_waypoints = (tuple(start), tuple(goal))

    def _apply_pending_waypoints(self) -> None:
        if self._pending_waypoints is None:
            return
        start, goal = self._pending_waypoints
        self._rust_env.set_waypoints(start, goal, self._segment_length(start, goal))
        self._active_start = start
        self._active_goal = goal
        self._route_remaining = []
        self._route_waypoint_count = 1
        self._route_waypoints_reached = 0
        self._config["waypoints"] = {"start": start, "goal": goal}
        self._pending_waypoints = None

    def queue_waypoint_route(self, start, waypoints) -> None:
        """Apply an ordered waypoint route on the next episode reset."""
        self._pending_route = {
            "start": tuple(start),
            "waypoints": [tuple(waypoint) for waypoint in waypoints],
        }

    def _activate_route(self, route: dict[str, Any]) -> None:
        start = tuple(route["start"])
        waypoints = [tuple(waypoint) for waypoint in route["waypoints"]]
        if not waypoints:
            raise ValueError("waypoint route requires at least one goal")
        self._rust_env.set_waypoints(
            start, waypoints[0], self._segment_length(start, waypoints[0])
        )
        self._active_start = start
        self._active_goal = waypoints[0]
        self._route_remaining = waypoints[1:]
        self._route_waypoint_count = len(waypoints)
        self._route_waypoints_reached = 0
        self._config["waypoint_route"] = {"start": start, "waypoints": waypoints}

    def _apply_pending_route(self) -> None:
        if self._pending_route is None:
            return
        self._activate_route(self._pending_route)
        self._pending_route = None

    def set_waypoint_curriculum(
        self,
        stages: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
        probabilities: list[float],
    ) -> None:
        """Set the stage pool sampled by subsequent training resets."""
        if len(stages) != len(probabilities):
            raise ValueError("curriculum stages and probabilities must have equal length")
        if stages and not np.isclose(sum(probabilities), 1.0):
            raise ValueError("curriculum stage probabilities must sum to 1.0")
        self._curriculum_stages = list(stages)
        self._curriculum_stage_probabilities = [float(value) for value in probabilities]

    def _sample_curriculum_waypoints(self) -> None:
        if not self._curriculum_stages:
            return
        index = int(
            self._obs_rng.choice(
                len(self._curriculum_stages),
                p=self._curriculum_stage_probabilities,
            )
        )
        selected = self._curriculum_stages[index]
        if isinstance(selected, dict) and "waypoints" in selected:
            self._activate_route(selected)
            return
        start, goal = selected
        self._route_remaining = []
        self._route_waypoint_count = 1
        self._route_waypoints_reached = 0
        self._rust_env.set_waypoints(start, goal, self._segment_length(start, goal))
        self._active_start = tuple(start)
        self._active_goal = tuple(goal)
        self._config["waypoints"] = {"start": start, "goal": goal}

    def _reset_task_state(self, reset_result):
        """Initialize episode-level task metrics after a Rust reset."""

        observation, _ = reset_result
        cursor = tuple(self._rust_env.cursor_pos())
        fallback = self._rust_env.goal_pos()
        distance = goal_distance(self._task.goal, cursor, fallback)
        self._episode_steps = 0
        self._consecutive_collisions = 0
        self._initial_distance = distance
        self._minimum_distance = distance
        self._previous_task_distance = distance
        self._episode_reward_breakdown = {}
        self._initial_filled = (
            set()
            if self._config.get("compiled_world_path") is not None
            else {tuple(coord) for coord in self._rust_env.filled_voxels()}
        )
        self._last_observation = observation
        info = {
            "task_version": self._task.version,
            "initial_goal_distance": distance,
        }
        if self._previous_scenario is not None:
            info["scenario"] = dict(self._previous_scenario)
        if self._last_feasibility_diagnostics is not None:
            info["geometry_feasibility"] = dict(self._last_feasibility_diagnostics)
        if self._last_accepted_task_manifest is not None:
            info["accepted_task"] = dict(self._last_accepted_task_manifest)
            info["accepted_task_identity"] = self._last_accepted_task_manifest[
                "identity_sha256"
            ]
        return observation, info

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

        route_complete = not self._route_remaining
        final_success = success and route_complete
        waypoint_reached = success
        if waypoint_reached:
            self._route_waypoints_reached += 1
        if success and self._route_remaining:
            next_goal = self._route_remaining.pop(0)
            observation = self._obs_to_numpy(
                self._rust_env.set_goal(
                    next_goal, self._segment_length(cursor, next_goal)
                )
            )
            fallback = next_goal
            current_distance = goal_distance(self._task.goal, cursor, fallback)
            terminated = False
            reason = "in_progress"
        else:
            reason = str(result.termination_reason)
        from theseo_anysearch.environments.lifecycle import (
            LifecycleContext,
            evaluate_lifecycle,
        )

        lifecycle = evaluate_lifecycle(
            self._lifecycle_rules,
            LifecycleContext(
                step=self._episode_steps,
                action=action,
                action_index=int(action_index),
                cursor=cursor,
                goal=tuple(fallback) if fallback is not None else None,
                goal_distance=current_distance,
                collision=collision,
                invalid_action=invalid_action,
                native_success=success,
                native_terminated=terminated,
                native_truncated=truncated,
                native_reason=reason,
                route_complete=route_complete,
                diagnostics={
                    "consecutive_collisions": int(result.consecutive_collisions),
                    "construction_residual": residual,
                    "construction_overshoot": overshoot,
                },
            ),
        )
        final_success = lifecycle.success
        terminated = lifecycle.terminated
        truncated = lifecycle.truncated
        reason = lifecycle.reason
        self._minimum_distance = min(self._minimum_distance, current_distance)
        self._previous_task_distance = current_distance
        self._last_observation = observation
        info = {
            "task_version": self._task.version,
            "goal_reached": final_success,
            "waypoint_reached": waypoint_reached,
            "route_waypoints_total": self._route_waypoint_count,
            "route_waypoints_reached": self._route_waypoints_reached,
            "route_waypoint_completion_fraction": (
                self._route_waypoints_reached / self._route_waypoint_count
                if self._route_waypoint_count
                else 0.0
            ),
            "route_waypoints_remaining": len(self._route_remaining),
            "termination_reason": reason,
            "failure": lifecycle.failure,
            "lifecycle_diagnostics": lifecycle.diagnostics,
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
        if self._previous_scenario is not None:
            info["scenario"] = dict(self._previous_scenario)
        cache_metrics = self._rust_env.world_cache_metrics()
        if cache_metrics is not None:
            info["world_cache"] = dict(cache_metrics)
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
            or self._config.get("waypoints")
            or self._config.get("waypoint_route")
            or (self._config.get("waypoint_curriculum") or {}).get("enabled")
            or self._config.get("stl_path")
            or self._config.get("geometry_pool")
            or self._config.get("scenario_provider")
            or self._config.get("compiled_world_path")
        )

    def _observation_space(self) -> gymnasium.Space:
        mode = self._config.get("obs_mode", "scalar")
        action_mode = self._config.get("action_mode", "discrete_26")
        mask_size = 27 if action_mode == "vector_3" else build_action_space(action_mode).n
        def with_mask(items: dict[str, spaces.Space]) -> spaces.Dict:
            if self._config.get("action_masking_enabled", False):
                items["action_mask"] = spaces.Box(0, 1, (mask_size,), np.int8)
            return spaces.Dict(items)
        goal_space = (
            {
                "goal_distance": spaces.Box(0.0, 1.0, (1,), np.float32),
                "goal_direction": spaces.Box(-1.0, 1.0, (3,), np.float32),
            }
            if self._has_goal()
            else {}
        )

        if mode == "scalar":
            return with_mask(goal_space)
        if mode == "box":
            n = 2 * self._config.get("box_radius", 2) + 1
            return with_mask({
                "local_grid":      spaces.Box(0.0, 1.0,   (n**3,), np.float32),
                **goal_space,
            })
        if mode == "radial":
            return with_mask({
                "ray_hits":        spaces.Box(0.0, 1.0,   (26,), np.float32),
                "ray_hit_types":   spaces.Box(0.0, 1.0,   (26,), np.float32),
                **goal_space,
            })
        if mode == "hierarchical_box":
            radii = self._config.get("box_radii") or [1, 4]
            flat_size = sum((2 * r + 1) ** 3 for r in radii)
            return with_mask({
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
        base = {}
        if self._has_goal_flag and rust_obs.goal_distance is not None:
            self._buf_goal[0] = rust_obs.goal_distance * self._inv_max_manhattan
            base["goal_distance"] = self._buf_goal.copy()
            if rust_obs.goal_direction is not None:
                self._buf_goal_direction[:] = rust_obs.goal_direction
                base["goal_direction"] = self._buf_goal_direction.copy()

        if self._obs_mode == "scalar":
            return self._attach_action_mask(base)

        if self._obs_mode == "box":
            local_grid = rust_obs.local_grid
            if local_grid is None:
                local_grid = self._rust_env.box_obs(self._box_radius)
            self._buf_grid[:] = local_grid
            self._buf_grid *= 1.0 / MAX_VOXEL_KIND
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
            self._buf_grid *= 1.0 / MAX_VOXEL_KIND
            base["local_grid"] = self._buf_grid.copy()
        else:
            raise ValueError(
                f"Unknown obs_mode: {self._obs_mode!r}. "
                "Expected 'scalar', 'box', 'radial', or 'hierarchical_box'."
            )
        if self._obs_log_count <= 5:
            self._log_env_stage(f"obs_to_numpy done index={self._obs_log_count}")
        return self._attach_action_mask(base)

    def _attach_action_mask(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Attach predicate feasibility while enforcing the all-masked contract."""
        if not self._config.get("action_masking_enabled", False):
            return observation
        mask = self.action_mask()
        if np.any(mask):
            observation["action_mask"] = mask
            return observation
        raise RuntimeError(
            "all actions are masked by the configured predicates; the current "
            "discrete action spaces do not expose a no-op action"
        )
