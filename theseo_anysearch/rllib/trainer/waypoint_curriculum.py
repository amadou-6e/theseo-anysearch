"""Centralized evaluation-driven waypoint curriculum state."""

from __future__ import annotations

import math
from itertools import product
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.models import WaypointCurriculumConfig
from theseo_anysearch.rllib.trainer.waypoint_routes import WaypointRoute, sample_route

Waypoint = tuple[int, int, int]


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


class WaypointCurriculumState(BaseModel):
    """Serializable trainer-owned waypoint curriculum state."""

    model_config = ConfigDict(extra="forbid")
    stage: int = 0
    start: Waypoint
    goal: Waypoint
    waypoints: tuple[Waypoint, ...] = ()
    successes_in_stage: int = 0
    transitions: list[WaypointTransition] = Field(default_factory=list)


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

    def observe(self, iteration: int, successes: int) -> bool:
        """Record evaluation successes and return whether the stage advances."""
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

    def advance(self, iteration: int, start: Waypoint, goal: Waypoint) -> WaypointTransition:
        """Commit a sampled waypoint pair as the next curriculum stage."""
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

    def target_distance(self, grid_size: int) -> float:
        """Return the requested separation for the next curriculum stage."""
        assert self.config.initial_start is not None
        assert self.config.initial_goal is not None
        initial_distance = waypoint_distance(
            self.config.initial_start,
            self.config.initial_goal,
        )
        difficulty = self.config.difficulty
        grid_diagonal = math.sqrt(3.0) * (grid_size - 1)
        maximum = min(difficulty.maximum_distance or grid_diagonal, grid_diagonal)
        next_stage = self.state.stage + 1
        return min(
            initial_distance + difficulty.distance_increment * next_stage,
            maximum,
        )

    def _sample_route(self, env_config: dict[str, Any], stage: int) -> WaypointRoute:
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
            grid_size=int(env_config.get("grid_size", 32)),
            action_mode=str(env_config.get("action_mode", "discrete_26")),
            seed=self.config.seed + stage,
        )

    def sample_stage(self, env_config: dict[str, Any]) -> WaypointRoute | tuple[Waypoint, Waypoint]:
        if self.config.completion_mode == "continue_route":
            return self._sample_route(env_config, self.state.stage + 1)
        return self.sample(env_config)

    def advance_stage(
        self,
        iteration: int,
        stage: WaypointRoute | tuple[Waypoint, Waypoint],
    ) -> WaypointTransition:
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
        grid_size = int(env_config.get("grid_size", 32))
        if grid_size < 2:
            raise ValueError("monotonic_distance curriculum requires grid_size >= 2")

        rng = np.random.default_rng(self.config.seed + self.state.stage + 1)
        target = self.target_distance(grid_size)
        previous = waypoint_distance(self.state.start, self.state.goal)
        minimum = previous
        center_value = (grid_size + 1) // 2
        center = (center_value, center_value, center_value)
        grid_radius = float(min(center_value - 1, grid_size - center_value))

        if target <= grid_radius:
            displacement = self._sample_displacement(
                rng,
                target,
                minimum,
                grid_size,
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
            grid_size,
            excluded=tuple(
                self.state.goal[index] - self.state.start[index]
                for index in range(3)
            ),
        )
        start_values = []
        for delta in displacement:
            low = 1 if delta >= 0 else 1 - delta
            high = grid_size - delta if delta >= 0 else grid_size
            start_values.append(int(rng.integers(low, high + 1)))
        start = tuple(start_values)
        goal = tuple(start[index] + displacement[index] for index in range(3))
        return start, goal

    def _sample_displacement(
        self,
        rng: np.random.Generator,
        target: float,
        minimum: float,
        grid_size: int,
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
                grid_size,
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
            grid_size,
            center,
            excluded,
        )

    @staticmethod
    def _displacement_fits(
        delta: Waypoint,
        grid_size: int,
        center: Waypoint | None,
    ) -> bool:
        if delta == (0, 0, 0):
            return False
        if center is None:
            return all(abs(value) <= grid_size - 1 for value in delta)
        goal = tuple(center[index] + delta[index] for index in range(3))
        return all(1 <= value <= grid_size for value in goal)

    def _nearest_lattice_displacement(
        self,
        rng: np.random.Generator,
        target: float,
        minimum: float,
        grid_size: int,
        center: Waypoint | None,
        excluded: Waypoint | None,
    ) -> Waypoint:
        """Guarantee a valid monotonic lattice result when spherical rounding misses."""
        best_error = math.inf
        best: list[Waypoint] = []

        if center is not None:
            candidates = (
                (
                    goal_x - center[0],
                    goal_y - center[1],
                    goal_z - center[2],
                )
                for goal_x in range(1, grid_size + 1)
                for goal_y in range(1, grid_size + 1)
                for goal_z in range(1, grid_size + 1)
            )
        else:
            limit = grid_size - 1
            candidates = (
                (dx, dy, dz)
                for dx in range(0, limit + 1)
                for dy in range(0, limit + 1)
                for dz in range(0, limit + 1)
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

        if center is not None:
            return best[int(rng.integers(0, len(best)))]

        signed = sorted({
            tuple(
                value * sign if value else 0
                for value, sign in zip(candidate, signs)
            )
            for candidate in best
            for signs in product((-1, 1), repeat=3)
        } - ({excluded} if excluded is not None else set()))
        if not signed:
            raise RuntimeError("could not sample a new waypoint direction at the distance cap")
        return signed[int(rng.integers(0, len(signed)))]

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


def broadcast_waypoints(algo: Any, start: Waypoint, goal: Waypoint) -> None:
    """Queue a waypoint pair on every local and remote RLlib environment."""
    env_runner_group = getattr(algo, "env_runner_group", None)
    if env_runner_group is None:
        raise RuntimeError("RLlib algorithm has no environment runner group")
    env_runner_group.foreach_env(lambda env: env.queue_waypoints(start, goal))


def broadcast_waypoint_curriculum(algo: Any, curriculum: WaypointCurriculum) -> None:
    """Broadcast the current and retained training stages to every environment."""
    env_runner_group = getattr(algo, "env_runner_group", None)
    if env_runner_group is None:
        raise RuntimeError("RLlib algorithm has no environment runner group")
    sampling = curriculum.config.training_sampling
    stages = curriculum.stages()
    env_runner_group.foreach_env(
        lambda env: env.set_waypoint_curriculum(
            stages,
            sampling.current_stage_probability,
            sampling.retained_stage_probability,
        )
    )
