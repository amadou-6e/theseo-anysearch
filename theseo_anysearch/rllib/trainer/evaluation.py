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

    @classmethod
    def from_voxel_episodes(
        cls,
        episodes: list[Any],
        env_config: dict[str, Any],
        *,
        min_success_rate: float,
    ) -> "EvaluationMetrics":
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
        """Return one flat metric map for Tune, MLflow, TensorBoard, and CLI."""
        return {
            "evaluation_episodes": float(self.evaluated_episodes),
            "evaluation_goals_reached": float(self.successes),
            "evaluation_success_rate": self.success_rate,
            "evaluation_steps_to_goal_min": self.steps_to_goal_min,
            "evaluation_steps_to_goal_mean": self.steps_to_goal_mean,
            "evaluation_steps_to_goal_max": self.steps_to_goal_max,
            "evaluation_final_goal_distance_mean": self.final_goal_distance_mean,
            "evaluation_minimum_goal_distance_mean": self.minimum_goal_distance_mean,
            "evaluation_terminated_count": float(self.terminated_count),
            "evaluation_truncated_count": float(self.truncated_count),
            "evaluation_shaped_return_mean": self.shaped_return_mean,
            "evaluation_unshaped_return_mean": self.unshaped_return_mean,
            "evaluation_goal_progress_mean": self.goal_progress_mean,
            **{
                f"evaluation_reward_{name}_mean": value
                for name, value in self.reward_component_means.items()
            },
        }

    def tensorboard_metrics(self) -> dict[str, float]:
        """Return the same metrics under the TensorBoard evaluation namespace."""
        return {
            "eval/" + key.removeprefix("evaluation_"): value
            for key, value in self.scalar_metrics().items()
        }


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
