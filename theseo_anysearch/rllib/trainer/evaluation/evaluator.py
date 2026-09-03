"""Standardized success metrics for deterministic policy evaluation batches."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvaluationMetrics(BaseModel):
    """Complete, serializable evaluation contract shared by all reporters."""

    model_config = ConfigDict(extra="forbid")

    evaluated_episodes: int
    successes: int
    success_rate: float
    steps_to_goal: list[int]
    steps_to_goal_min: float
    steps_to_goal_mean: float
    steps_to_goal_max: float
    final_goal_distance_mean: float
    minimum_goal_distance_mean: float
    terminated_count: int
    truncated_count: int
    shaped_return_mean: float
    unshaped_return_mean: float
    goal_progress_mean: float
    status: str
    reward_component_means: dict[str, float] = Field(default_factory=dict)
    route_success_rate: float | None = None
    waypoint_completion_fraction_mean: float | None = None
    route_efficiency_mean: float | None = None

    @classmethod
    def from_voxel_episodes(
        cls,
        episodes: list[Any],
        env_config: dict[str, Any],
        *,
        min_success_rate: float,
    ) -> "EvaluationMetrics":
        """Aggregate deterministic voxel episodes into evaluation metrics.

        Parameters
        ----------
        episodes : list[Any]
            Completed deterministic evaluation episodes.
        env_config : dict[str, Any]
            Runtime environment configuration.
        min_success_rate : float
            Required success rate used to assign evaluation status.

        Returns
        -------
        EvaluationMetrics
            Serializable aggregate evaluation metrics.

        Raises
        ------
        ValueError
            If no episodes are supplied.
        """
        if not episodes:
            raise ValueError("evaluation metrics require at least one episode")

        successful_steps = [
            len(episode.steps) for episode in episodes if episode.success
        ]
        final_distances = [
            _distance(_final_position(episode), episode.goal_pos)
            for episode in episodes
        ]
        minimum_distances = [
            min(
                [_distance(episode.start_pos, episode.goal_pos)]
                + [
                    _distance(
                        (step.cursor_x, step.cursor_y, step.cursor_z),
                        episode.goal_pos,
                    )
                    for step in episode.steps
                ]
            )
            for episode in episodes
        ]
        start_distances = [
            _distance(episode.start_pos, episode.goal_pos) for episode in episodes
        ]
        shaped_returns = [float(episode.total_reward) for episode in episodes]
        unshaped_returns = [
            _unshaped_return(episode, env_config) for episode in episodes
        ]
        count = len(episodes)
        component_names = set().union(*(episode.reward_breakdown or {} for episode in episodes))
        component_means = {
            name: sum((episode.reward_breakdown or {}).get(name, 0.0) for episode in episodes) / count
            for name in component_names
        }
        route_infos = [
            episode.final_info or {}
            for episode in episodes
            if (episode.final_info or {}).get("route_waypoints_total", 0)
        ]
        completion_mean = (
            sum(
                float(info.get("route_waypoint_completion_fraction", 0.0))
                for info in route_infos
            ) / len(route_infos)
            if route_infos else None
        )
        route_efficiency = None
        route = env_config.get("waypoint_route")
        if route_infos and route:
            from theseo_anysearch.environments.action_spaces import action_step_distance

            points = [
                tuple(route["start"]),
                *(tuple(item) for item in route["waypoints"]),
            ]
            route_length = sum(
                action_step_distance(
                    start,
                    goal,
                    env_config.get("action_mode", "discrete_26"),
                )
                for start, goal in zip(points, points[1:])
            )
            efficiencies = [
                route_length / len(episode.steps)
                for episode in episodes
                if episode.success and episode.steps
            ]
            route_efficiency = (
                sum(efficiencies) / len(efficiencies) if efficiencies else 0.0
            )
        successes = len(successful_steps)
        success_rate = successes / count
        progress = sum(
            start - final
            for start, final in zip(start_distances, final_distances)
        ) / count

        if successes and success_rate >= min_success_rate:
            status = "solved"
        elif successes:
            status = "below_success_threshold"
        elif progress > 0.0:
            status = "approaching_not_solved"
        else:
            status = "not_solved"

        return cls(
            evaluated_episodes=count,
            successes=successes,
            success_rate=success_rate,
            steps_to_goal=successful_steps,
            steps_to_goal_min=float(min(successful_steps)) if successful_steps else 0.0,
            steps_to_goal_mean=(
                sum(successful_steps) / successes if successful_steps else 0.0
            ),
            steps_to_goal_max=float(max(successful_steps)) if successful_steps else 0.0,
            final_goal_distance_mean=sum(final_distances) / count,
            minimum_goal_distance_mean=sum(minimum_distances) / count,
            terminated_count=successes,
            truncated_count=count - successes,
            shaped_return_mean=sum(shaped_returns) / count,
            unshaped_return_mean=sum(unshaped_returns) / count,
            goal_progress_mean=progress,
            status=status,
            reward_component_means=component_means,
            route_success_rate=success_rate if route_infos else None,
            waypoint_completion_fraction_mean=completion_mean,
            route_efficiency_mean=route_efficiency,
        )


    @classmethod
    def from_multi_voxel_episodes(
        cls,
        episodes: list[Any],
        env_config: dict[str, Any],
        *,
        min_success_rate: float,
    ) -> "EvaluationMetrics":
        """Summarize agent-level outcomes from multi-agent evaluation episodes."""
        if not episodes:
            raise ValueError("evaluation metrics require at least one episode")

        final_distances: list[float] = []
        minimum_distances: list[float] = []
        start_distances: list[float] = []
        successful_steps: list[int] = []
        shaped_returns: list[float] = []
        unshaped_returns: list[float] = []
        step_cost = float(env_config.get("step_cost", -0.01))
        goal_reward = float(env_config.get("goal_reward", 1.0))

        for episode in episodes:
            for agent_index in range(episode.agent_count):
                start = episode.start_positions[agent_index]
                goal = episode.goal_positions[agent_index]
                positions = [step.cursors[agent_index] for step in episode.steps]
                final = positions[-1] if positions else start
                success = goal is not None and final == goal
                if success:
                    successful_steps.append(len(episode.steps))
                start_distances.append(_distance(start, goal))
                final_distances.append(_distance(final, goal))
                minimum_distances.append(
                    min(
                        [_distance(start, goal)]
                        + [_distance(position, goal) for position in positions]
                    )
                )
                shaped_returns.append(float(episode.total_rewards[agent_index]))
                unshaped_returns.append(
                    step_cost * len(episode.steps)
                    + (goal_reward if success else 0.0)
                )

        count = len(final_distances)
        successes = len(successful_steps)
        success_rate = successes / count if count else 0.0
        progress = (
            sum(
                start - final
                for start, final in zip(start_distances, final_distances)
            ) / count
            if count
            else 0.0
        )
        if successes and success_rate >= min_success_rate:
            status = "solved"
        elif successes:
            status = "below_success_threshold"
        elif progress > 0.0:
            status = "approaching_not_solved"
        else:
            status = "not_solved"

        return cls(
            evaluated_episodes=count,
            successes=successes,
            success_rate=success_rate,
            steps_to_goal=successful_steps,
            steps_to_goal_min=float(min(successful_steps)) if successful_steps else 0.0,
            steps_to_goal_mean=(
                sum(successful_steps) / successes if successful_steps else 0.0
            ),
            steps_to_goal_max=float(max(successful_steps)) if successful_steps else 0.0,
            final_goal_distance_mean=sum(final_distances) / count if count else 0.0,
            minimum_goal_distance_mean=sum(minimum_distances) / count if count else 0.0,
            terminated_count=successes,
            truncated_count=count - successes,
            shaped_return_mean=sum(shaped_returns) / count if count else 0.0,
            unshaped_return_mean=sum(unshaped_returns) / count if count else 0.0,
            goal_progress_mean=progress,
            status=status,
        )

    def scalar_metrics(self) -> dict[str, float]:
        """Return canonical metrics shared by MLflow and TensorBoard."""
        metrics = {
            "eval/task/episodes": float(self.evaluated_episodes),
            "eval/task/success_rate": self.success_rate,
            "eval/task/steps_to_success_mean": self.steps_to_goal_mean,
            "eval/task/final_goal_distance_mean": self.final_goal_distance_mean,
            "eval/task/minimum_goal_distance_mean": self.minimum_goal_distance_mean,
            "eval/task/termination_rate": (
                self.terminated_count / self.evaluated_episodes
                if self.evaluated_episodes else 0.0
            ),
            "eval/task/truncation_rate": (
                self.truncated_count / self.evaluated_episodes
                if self.evaluated_episodes else 0.0
            ),
            "eval/task/return_mean": self.shaped_return_mean,
            "eval/task/goal_progress_mean": self.goal_progress_mean,
        }
        if not math.isclose(self.shaped_return_mean, self.unshaped_return_mean):
            metrics["eval/task/unshaped_return_mean"] = self.unshaped_return_mean
        if len(self.reward_component_means) > 1:
            metrics.update({
                f"eval/task/reward/{name}_mean": value
                for name, value in self.reward_component_means.items()
            })
        if self.route_success_rate is not None:
            metrics["eval/task/waypoint/route_success_rate"] = self.route_success_rate
            metrics["eval/task/waypoint/completion_fraction_mean"] = float(
                self.waypoint_completion_fraction_mean or 0.0
            )
            if self.route_efficiency_mean is not None:
                metrics["eval/task/waypoint/route_efficiency_mean"] = (
                    self.route_efficiency_mean
                )
        return metrics

    def tensorboard_metrics(self) -> dict[str, float]:
        """Return the same metrics under the TensorBoard evaluation namespace."""
        return self.scalar_metrics()


def _distance(
    position: tuple[int, int, int] | None,
    goal: tuple[int, int, int] | None,
) -> float:
    if position is None or goal is None:
        return 0.0
    return math.dist(position, goal)


def _final_position(episode: Any) -> tuple[int, int, int] | None:
    if not episode.steps:
        return episode.start_pos
    step = episode.steps[-1]
    return (step.cursor_x, step.cursor_y, step.cursor_z)


def _unshaped_return(episode: Any, env_config: dict[str, Any]) -> float:
    """Return recorded unshaped task return, with legacy reconstruction fallback."""
    if getattr(episode, "unshaped_return", None) is not None:
        return float(episode.unshaped_return)
    step_cost = float(env_config.get("step_cost", -0.01))
    collision_cost = float(env_config.get("collision_cost", 0.0))
    goal_reward = float(env_config.get("goal_reward", 1.0))
    collisions = 0
    previous = episode.start_pos
    for step in episode.steps:
        current = (step.cursor_x, step.cursor_y, step.cursor_z)
        if previous is not None and current == previous:
            collisions += 1
        previous = current
    return (
        step_cost * len(episode.steps)
        + collision_cost * collisions
        + (goal_reward if episode.success else 0.0)
    )
