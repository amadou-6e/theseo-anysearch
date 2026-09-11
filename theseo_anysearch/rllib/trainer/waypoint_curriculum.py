"""Centralized evaluation-driven waypoint curriculum state."""

from __future__ import annotations

import math
from functools import partial
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.models import WaypointCurriculumConfig
from theseo_anysearch.rllib.trainer.waypoint_routes import WaypointRoute, sample_route
from theseo_anysearch.worlds.extent import (
    WorldExtent,
    maximum_euclidean,
    resolve_task_extent,
    task_center,
)

Waypoint = tuple[int, int, int]


def configure_initial_waypoint_curriculum(
    env: Any,
    env_config: dict[str, Any],
) -> None:
    """Activate stage zero on a directly constructed environment."""

    raw_curriculum = env_config.get("waypoint_curriculum") or {}
    if not raw_curriculum.get("enabled", False):
        return
    curriculum = WaypointCurriculum(
        WaypointCurriculumConfig.model_validate(raw_curriculum),
        env_config,
    )
    env.set_waypoint_curriculum(curriculum.stages(), [1.0])


class WaypointTransition(BaseModel):
    """One recorded curriculum-stage transition."""

    model_config = ConfigDict(extra="forbid")
    iteration: int
    from_stage: int
    to_stage: int
    start: Waypoint
    goal: Waypoint
    waypoints: tuple[Waypoint, ...] = ()
    trigger: str


class WaypointStageEvaluation(BaseModel):
    """Most recent deterministic evaluation outcome for one stage."""

    model_config = ConfigDict(extra="forbid")
    attempts: int = Field(default=0, ge=0)
    successes: int = Field(default=0, ge=0)


class WaypointCurriculumState(BaseModel):
    """Serializable trainer-owned waypoint curriculum state."""

    model_config = ConfigDict(extra="forbid")
    stage: int = 0
    start: Waypoint
    goal: Waypoint
    waypoints: tuple[Waypoint, ...] = ()
    successes_in_stage: int = 0
    transitions: list[WaypointTransition] = Field(default_factory=list)
    stage_evaluations: dict[int, WaypointStageEvaluation] = Field(default_factory=dict)


def waypoint_distance(start: Waypoint, goal: Waypoint) -> float:
    """Return Euclidean waypoint separation in voxel units."""
    return math.dist(start, goal)


class WaypointCurriculum:
    """Decide curriculum transitions from deterministic evaluation results."""

    def __init__(self, config: WaypointCurriculumConfig, env_config: dict[str, Any] | None = None) -> None:
        if config.initial_start is None:
            raise ValueError("waypoint curriculum requires an initial start")
        self.config = config
        self._initial_route: WaypointRoute | None = None
        if config.completion_mode == "continue_route":
            if env_config is None:
                raise ValueError("continue_route curriculum requires environment settings")
            self._initial_route = self._sample_route(env_config, stage=0)
            self.state = WaypointCurriculumState(
                start=self._initial_route.start,
                goal=self._initial_route.goal,
                waypoints=self._initial_route.waypoints,
            )
        else:
            if config.initial_goal is None:
                raise ValueError("waypoint curriculum requires an initial waypoint pair")
            self.state = WaypointCurriculumState(start=config.initial_start, goal=config.initial_goal)

    def stages(self) -> list[Any]:
        """Return the initial stage followed by every visited stage."""
        if self._initial_route is not None:
            return [self._initial_route.model_dump(mode="python")] + [
                {"start": transition.start, "waypoints": transition.waypoints}
                for transition in self.state.transitions
            ]
        initial = (self.config.initial_start, self.config.initial_goal)
        assert initial[0] is not None and initial[1] is not None
        return [(initial[0], initial[1])] + [
            (transition.start, transition.goal) for transition in self.state.transitions
        ]

    @property
    def maximum_stage(self) -> int | None:
        """Return the last distinct configured difficulty stage, when bounded."""
        maximum = self.config.difficulty.maximum_distance
        if maximum is None:
            return None
        if self.config.completion_mode == "continue_route":
            initial = self.config.difficulty.initial_distance
            assert initial is not None
            initial_distance = float(initial)
        elif self.config.difficulty.mode == "monotonic_distance":
            assert self.config.initial_start is not None
            assert self.config.initial_goal is not None
            initial_distance = waypoint_distance(
                self.config.initial_start,
                self.config.initial_goal,
            )
        else:
            return None
        remaining = max(float(maximum) - initial_distance, 0.0)
        stage_span = remaining / self.config.difficulty.distance_increment
        rounded_span = round(stage_span)
        if math.isclose(stage_span, rounded_span, rel_tol=1e-12, abs_tol=1e-12):
            return int(rounded_span)
        return int(math.ceil(stage_span))

    @property
    def terminal(self) -> bool:
        """Whether the curriculum has reached its final distinct difficulty."""
        maximum_stage = self.maximum_stage
        return maximum_stage is not None and self.state.stage >= maximum_stage

    def clamp_restored_state(self) -> bool:
        """Clamp legacy state that advanced beyond the configured difficulty cap."""
        maximum_stage = self.maximum_stage
        if maximum_stage is None or self.state.stage <= maximum_stage:
            return False
        self.state.transitions = self.state.transitions[:maximum_stage]
        self.state.stage_evaluations = {
            stage: evaluation
            for stage, evaluation in self.state.stage_evaluations.items()
            if stage <= maximum_stage
        }
        self.state.stage = maximum_stage
        self.state.successes_in_stage = 0
        if maximum_stage == 0:
            if self._initial_route is not None:
                self.state.start = self._initial_route.start
                self.state.goal = self._initial_route.goal
                self.state.waypoints = self._initial_route.waypoints
            else:
                assert self.config.initial_start is not None
                assert self.config.initial_goal is not None
                self.state.start = self.config.initial_start
                self.state.goal = self.config.initial_goal
                self.state.waypoints = ()
        else:
            transition = self.state.transitions[-1]
            self.state.start = transition.start
            self.state.goal = transition.goal
            self.state.waypoints = transition.waypoints
        return True

    def observe(self, iteration: int, successes: int) -> bool:
        """Record evaluation successes and return whether the stage advances."""
        if self.terminal:
            return False
        self.state.successes_in_stage += max(int(successes), 0)
        advance = self.config.advance
        if advance.mode == "fixed":
            return False
        if advance.mode == "success":
            return self.state.successes_in_stage >= advance.successes_required

        interval_due = iteration % advance.interval_iterations == 0
        if not interval_due:
            return False
        if not advance.require_success:
            return True
        return self.state.successes_in_stage >= advance.successes_required

    def record_stage_evaluations(
        self,
        outcomes: list[tuple[int, int]],
    ) -> None:
        """Store the latest ``(attempts, successes)`` for each visited stage."""
        if len(outcomes) != len(self.stages()):
            raise ValueError("stage evaluation outcomes must cover every visited stage")
        for stage_index, (attempts, successes) in enumerate(outcomes):
            if attempts < 0 or successes < 0 or successes > attempts:
                raise ValueError("invalid stage evaluation outcome")
            self.state.stage_evaluations[stage_index] = WaypointStageEvaluation(
                attempts=attempts, successes=successes
            )

    def sampling_probabilities(self) -> list[float]:
        """Return normalized training probabilities for all visited stages."""
        from theseo_anysearch.rllib.trainer.stage_sampling import (
            StageSamplingContext,
            StageSamplingStage,
            stage_probabilities,
        )

        stages = self.stages()
        latest = len(stages) - 1
        context_stages = []
        for index, stage in enumerate(stages):
            if isinstance(stage, dict):
                start = stage["start"]
                goal = stage["waypoints"][-1]
            else:
                start, goal = stage
            evaluation = self.state.stage_evaluations.get(index)
            attempts = evaluation.attempts if evaluation is not None else 0
            successes = evaluation.successes if evaluation is not None else 0
            context_stages.append(
                StageSamplingStage(
                    index=index,
                    start=start,
                    goal=goal,
                    age=latest - index,
                    is_latest=index == latest,
                    evaluation_attempts=attempts,
                    evaluation_successes=successes,
                    evaluation_success_rate=(
                        successes / attempts if attempts else None
                    ),
                )
            )
        return stage_probabilities(
            StageSamplingContext(stages=tuple(context_stages)),
            self.config.training_sampling,
        )

    def advance(self, iteration: int, start: Waypoint, goal: Waypoint) -> WaypointTransition:
        """Commit a sampled waypoint pair as the next curriculum stage."""
        if self.terminal:
            raise RuntimeError("cannot advance a terminal waypoint curriculum")
        previous_stage = self.state.stage
        transition = WaypointTransition(
            iteration=iteration,
            from_stage=previous_stage,
            to_stage=previous_stage + 1,
            start=start,
            goal=goal,
            trigger=self.config.advance.mode,
        )
        self.state.stage += 1
        self.state.start = start
        self.state.goal = goal
        self.state.successes_in_stage = 0
        self.state.transitions.append(transition)
        return transition

    def target_distance(self, extent: WorldExtent | int) -> float:
        """Return the requested separation for the next curriculum stage."""
        assert self.config.initial_start is not None
        assert self.config.initial_goal is not None
        initial_distance = waypoint_distance(
            self.config.initial_start,
            self.config.initial_goal,
        )
        difficulty = self.config.difficulty
        if isinstance(extent, int):
            extent = (extent, extent, extent)
        grid_diagonal = maximum_euclidean(extent)
        maximum = min(difficulty.maximum_distance or grid_diagonal, grid_diagonal)
        next_stage = self.state.stage + 1
        return min(
            initial_distance + difficulty.distance_increment * next_stage,
            maximum,
        )

    def _sample_route(
        self,
        env_config: dict[str, Any],
        stage: int,
        *,
        seed: int | None = None,
    ) -> WaypointRoute:
        """Generate a route at a configured stage with an optional independent seed."""
        self._require_empty_geometry(env_config)
        difficulty = self.config.difficulty
        assert difficulty.initial_distance is not None
        assert self.config.route_length is not None
        segment_distance = int(min(
            difficulty.initial_distance + difficulty.distance_increment * stage,
            difficulty.maximum_distance or math.inf,
        ))
        return sample_route(
            start=self.config.initial_start,
            total_distance=self.config.route_length.resolve(int(env_config.get("max_steps", 200))),
            segment_distance=segment_distance,
            extent=resolve_task_extent(env_config),
            action_mode=str(env_config.get("action_mode", "discrete_26")),
            seed=self.config.seed + stage if seed is None else seed,
        )

    def route_for_stage(
        self,
        env_config: dict[str, Any],
        stage: int,
        *,
        seed: int | None = None,
    ) -> WaypointRoute:
        """Sample one route at an exact configured stage difficulty."""

        return self._sample_route(env_config, stage, seed=seed)

    def configured_route_stages(
        self, env_config: dict[str, Any]
    ) -> list[WaypointRoute]:
        """Return every configured segment-distance route stage."""

        if self.config.completion_mode != "continue_route":
            raise ValueError("all-stage collection requires continue_route mode")
        difficulty = self.config.difficulty
        if difficulty.initial_distance is None:
            raise ValueError("all-stage collection requires initial_distance")
        maximum = difficulty.maximum_distance
        if maximum is None:
            raise ValueError("all-stage collection requires maximum_distance")
        # Round to the nearest stage before falling back to ceil: floating-point
        # error can push an exact ratio (e.g. 9.0) just above the integer (e.g.
        # 9.000000000000002), and a naive ceil would silently add an extra stage.
        stage_span = (
            (maximum - difficulty.initial_distance) / difficulty.distance_increment
        )
        rounded_span = round(stage_span)
        if math.isclose(stage_span, rounded_span, rel_tol=1e-12, abs_tol=1e-12):
            stage_count = int(rounded_span) + 1
        else:
            stage_count = int(math.ceil(stage_span)) + 1
        return [
            self._sample_route(env_config, stage) for stage in range(stage_count)
        ]

    def sample_stage(self, env_config: dict[str, Any]) -> WaypointRoute | tuple[Waypoint, Waypoint]:
        if self.terminal:
            raise RuntimeError("cannot sample beyond the terminal waypoint curriculum stage")
        if self.config.completion_mode == "continue_route":
            return self._sample_route(env_config, self.state.stage + 1)
        return self.sample(env_config)

    def advance_stage(
        self,
        iteration: int,
        stage: WaypointRoute | tuple[Waypoint, Waypoint],
    ) -> WaypointTransition:
        if self.terminal:
            raise RuntimeError("cannot advance a terminal waypoint curriculum")
        if not isinstance(stage, WaypointRoute):
            return self.advance(iteration, *stage)
        previous_stage = self.state.stage
        transition = WaypointTransition(
            iteration=iteration,
            from_stage=previous_stage,
            to_stage=previous_stage + 1,
            start=stage.start,
            goal=stage.goal,
            waypoints=stage.waypoints,
            trigger=self.config.advance.mode,
        )
        self.state.stage += 1
        self.state.start = stage.start
        self.state.goal = stage.goal
        self.state.waypoints = stage.waypoints
        self.state.successes_in_stage = 0
        self.state.transitions.append(transition)
        return transition
    def sample(self, env_config: dict[str, Any]) -> tuple[Waypoint, Waypoint]:
        """Sample the next valid pair using the configured difficulty strategy."""
        if self.config.difficulty.mode == "monotonic_distance":
            return self._sample_monotonic_distance(env_config)
        return self._sample_environment_pair(env_config)

    def _sample_environment_pair(
        self,
        env_config: dict[str, Any],
    ) -> tuple[Waypoint, Waypoint]:
        """Preserve the original native environment sampling behavior."""
        from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv

        sample_config = dict(env_config)
        sample_config["waypoints_file"] = None
        sample_config["waypoints"] = None
        sample_config["waypoint_curriculum"] = {"enabled": False}
        env = VoxelEnv(sample_config)
        try:
            env.reset(seed=self.config.seed + self.state.stage + 1)
            raw_start = env._rust_env.cursor_pos()
            raw_goal = env._rust_env.goal_pos()
            if raw_goal is None:
                raise RuntimeError("environment did not provide a sampled goal")
            start = tuple(int(value) for value in raw_start)
            goal = tuple(int(value) for value in raw_goal)
            return start, goal
        finally:
            env.close()

    def _sample_monotonic_distance(
        self,
        env_config: dict[str, Any],
    ) -> tuple[Waypoint, Waypoint]:
        """Sample an empty-grid pair whose distance never decreases by stage."""
        self._require_empty_geometry(env_config)
        extent = resolve_task_extent(env_config)
        if maximum_euclidean(extent) < 1:
            raise ValueError("monotonic_distance curriculum requires a movable extent")

        rng = np.random.default_rng(self.config.seed + self.state.stage + 1)
        target = self.target_distance(extent)
        previous = waypoint_distance(self.state.start, self.state.goal)
        minimum = previous
        center = task_center(extent)
        grid_radius = float(
            min(
                min(center[index] - 1, extent[index] - center[index])
                for index in range(3)
            )
        )

        if target <= grid_radius:
            displacement = self._sample_displacement(
                rng,
                target,
                minimum,
                extent,
                center=center,
                excluded=(
                    tuple(
                        self.state.goal[index] - self.state.start[index]
                        for index in range(3)
                    )
                    if self.state.start == center
                    else None
                ),
            )
            goal = tuple(center[index] + displacement[index] for index in range(3))
            return center, goal

        displacement = self._sample_displacement(
            rng,
            target,
            minimum,
            extent,
            excluded=tuple(
                self.state.goal[index] - self.state.start[index]
                for index in range(3)
            ),
        )
        start_values = []
        for axis, delta in zip(extent, displacement):
            low = 1 if delta >= 0 else 1 - delta
            high = axis - delta if delta >= 0 else axis
            start_values.append(int(rng.integers(low, high + 1)))
        start = tuple(start_values)
        goal = tuple(start[index] + displacement[index] for index in range(3))
        return start, goal

    def _sample_displacement(
        self,
        rng: np.random.Generator,
        target: float,
        minimum: float,
        extent: WorldExtent,
        *,
        center: Waypoint | None = None,
        excluded: Waypoint | None = None,
    ) -> Waypoint:
        """Choose a lattice displacement near a uniformly sampled sphere direction."""
        candidates: list[tuple[float, Waypoint]] = []
        for _ in range(self.config.difficulty.sampling_attempts):
            direction = rng.normal(size=3)
            norm = float(np.linalg.norm(direction))
            if norm == 0.0:
                continue
            delta = tuple(int(value) for value in np.rint(direction * (target / norm)))
            distance = math.sqrt(sum(value * value for value in delta))
            if delta == excluded:
                continue
            if distance + 1e-9 < minimum or not self._displacement_fits(
                delta,
                extent,
                center,
            ):
                continue
            candidates.append((abs(distance - target), delta))

        if candidates:
            best_error = min(error for error, _ in candidates)
            best = [delta for error, delta in candidates if error == best_error]
            return best[int(rng.integers(0, len(best)))]

        return self._nearest_lattice_displacement(
            rng,
            target,
            minimum,
            extent,
            center,
            excluded,
        )

    @staticmethod
    def _displacement_fits(
        delta: Waypoint,
        extent: WorldExtent,
        center: Waypoint | None,
    ) -> bool:
        if delta == (0, 0, 0):
            return False
        if center is None:
            return all(
                abs(value) <= extent[index] - 1
                for index, value in enumerate(delta)
            )
        goal = tuple(center[index] + delta[index] for index in range(3))
        return all(1 <= value <= extent[index] for index, value in enumerate(goal))

    def _nearest_lattice_displacement(
        self,
        rng: np.random.Generator,
        target: float,
        minimum: float,
        extent: WorldExtent,
        center: Waypoint | None,
        excluded: Waypoint | None,
    ) -> Waypoint:
        """Guarantee a valid monotonic lattice result when spherical rounding misses."""
        best_error = math.inf
        best: list[Waypoint] = []

        # Search only a thin deterministic lattice band around the requested
        # distance. This avoids the old Cartesian enumeration of the full world.
        radius = max(1, int(math.ceil(target)) + 1)
        limits = tuple(min(axis - 1, radius) for axis in extent)
        candidates = (
            (dx, dy, dz)
            for dx in range(-limits[0], limits[0] + 1)
            for dy in range(-limits[1], limits[1] + 1)
            for dz in range(-limits[2], limits[2] + 1)
            if self._displacement_fits((dx, dy, dz), extent, center)
        )

        for delta in candidates:
            if delta == (0, 0, 0) or (center is not None and delta == excluded):
                continue
            distance = math.sqrt(sum(value * value for value in delta))
            if distance + 1e-9 < minimum:
                continue
            error = abs(distance - target)
            if error < best_error:
                best_error = error
                best = [delta]
            elif error == best_error:
                best.append(delta)
        if not best:
            raise RuntimeError("could not sample a non-decreasing waypoint distance")

        choices = sorted(set(best) - ({excluded} if excluded is not None else set()))
        if not choices:
            raise RuntimeError("could not sample a new waypoint direction at the distance cap")
        return choices[int(rng.integers(0, len(choices)))]

    @staticmethod
    def _require_empty_geometry(env_config: dict[str, Any]) -> None:
        populated = (
            env_config.get("stl_path")
            or env_config.get("stl_paths")
            or env_config.get("geometry_boxes")
            or env_config.get("geometry_pool")
        )
        if populated:
            raise ValueError(
                "monotonic_distance curriculum currently requires empty geometry"
            )


def _call_env_method(
    env_runner: Any,
    method_name: str,
    args: tuple[Any, ...],
) -> Any:
    """Invoke one environment method across a legacy or modern EnvRunner."""
    if hasattr(env_runner, "foreach_env"):
        return env_runner.foreach_env(
            lambda env: getattr(env, method_name)(*args)
        )
    vector_env = getattr(env_runner, "env", None)
    if vector_env is None:
        raise RuntimeError("RLlib EnvRunner has no environment")
    seen: set[int] = set()
    while not hasattr(vector_env, "call") and hasattr(vector_env, "env"):
        identity = id(vector_env)
        if identity in seen:
            break
        seen.add(identity)
        vector_env = vector_env.env
    if hasattr(vector_env, "call"):
        return vector_env.call(method_name, *args)
    environments = getattr(vector_env, "envs", (vector_env,))
    return [getattr(env, method_name)(*args) for env in environments]


def _broadcast_env_method(
    algo: Any,
    method_name: str,
    *args: Any,
) -> None:
    """Broadcast one method call through the configured RLlib execution stack."""
    env_runner_group = getattr(algo, "env_runner_group", None)
    if env_runner_group is None:
        raise RuntimeError("RLlib algorithm has no environment runner group")
    modern_stack = bool(
        getattr(
            getattr(algo, "config", None),
            "enable_env_runner_and_connector_v2",
            False,
        )
    )
    if not modern_stack:
        env_runner_group.foreach_env(
            lambda env: getattr(env, method_name)(*args)
        )
        return
    env_runner_group.foreach_env_runner(
        partial(_call_env_method, method_name=method_name, args=args),
        local_env_runner=False,
    )


def broadcast_waypoints(algo: Any, start: Waypoint, goal: Waypoint) -> None:
    """Queue a waypoint pair on every local and remote RLlib environment."""
    _broadcast_env_method(algo, "queue_waypoints", start, goal)


def broadcast_waypoint_curriculum(algo: Any, curriculum: WaypointCurriculum) -> None:
    """Broadcast the current and retained training stages to every environment."""
    _broadcast_env_method(
        algo,
        "set_waypoint_curriculum",
        curriculum.stages(),
        curriculum.sampling_probabilities(),
    )
