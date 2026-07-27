"""Centralized evaluation-driven waypoint curriculum state."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.models import WaypointCurriculumConfig

Waypoint = tuple[int, int, int]


class WaypointTransition(BaseModel):
    """One recorded curriculum-stage transition."""

    model_config = ConfigDict(extra="forbid")
    iteration: int
    from_stage: int
    to_stage: int
    start: Waypoint
    goal: Waypoint
    trigger: str


class WaypointCurriculumState(BaseModel):
    """Serializable trainer-owned waypoint curriculum state."""

    model_config = ConfigDict(extra="forbid")
    stage: int = 0
    start: Waypoint
    goal: Waypoint
    successes_in_stage: int = 0
    transitions: list[WaypointTransition] = Field(default_factory=list)


class WaypointCurriculum:
    """Decide curriculum transitions from deterministic evaluation results."""

    def __init__(self, config: WaypointCurriculumConfig) -> None:
        if config.initial_start is None or config.initial_goal is None:
            raise ValueError("waypoint curriculum requires an initial waypoint pair")
        self.config = config
        self.state = WaypointCurriculumState(
            start=config.initial_start,
            goal=config.initial_goal,
        )

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

    def sample(self, env_config: dict[str, Any]) -> tuple[Waypoint, Waypoint]:
        """Sample a valid pair reproducibly using the environment's native reset."""
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


def broadcast_waypoints(algo: Any, start: Waypoint, goal: Waypoint) -> None:
    """Queue a waypoint pair on every local and remote RLlib environment."""
    env_runner_group = getattr(algo, "env_runner_group", None)
    if env_runner_group is None:
        raise RuntimeError("RLlib algorithm has no environment runner group")
    env_runner_group.foreach_env(lambda env: env.queue_waypoints(start, goal))
