"""Evaluation-driven early stopping for standard training runs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from theseo_anysearch.settings import TrainingEarlyStopConfig


class EarlyStopState(BaseModel):
    """Persistent consecutive-match state used across resumed runs."""

    model_config = ConfigDict(extra="forbid")

    consecutive_matches: int = 0
    last_iteration: int = 0


class EarlyStopDecision(BaseModel):
    """Result of checking one deterministic evaluation batch."""

    model_config = ConfigDict(extra="forbid")

    triggered: bool
    mode: str | None = None
    value: float | None = None
    threshold: float | None = None
    consecutive_matches: int = 0
    required_consecutive: int = 0
    iteration: int


class TrainingEarlyStopController:
    """Track and evaluate a configured training early-stop condition."""

    def __init__(
        self,
        config: TrainingEarlyStopConfig,
        state: EarlyStopState | None = None,
    ) -> None:
        self.config = config
        self.state = state or EarlyStopState()

    def evaluate(
        self,
        iteration: int,
        *,
        reward_mean: float,
        goal_finishes: int,
        heuristic_accuracy: float | None = None,
        heuristic_distance: float | None = None,
    ) -> EarlyStopDecision:
        if not self.config.enabled:
            return EarlyStopDecision(triggered=False, iteration=iteration)

        values: dict[str, float | None] = {
            "reward": float(reward_mean),
            "goal_finishes": float(goal_finishes),
            "heuristic_accuracy": heuristic_accuracy,
            "heuristic_distance": heuristic_distance,
        }
        thresholds: dict[str, float | int | None] = {
            "reward": self.config.min_reward,
            "goal_finishes": self.config.min_goal_finishes,
            "heuristic_accuracy": self.config.min_heuristic_accuracy,
            "heuristic_distance": self.config.max_heuristic_distance,
        }
        mode = str(self.config.mode)
        value = values[mode]
        threshold = thresholds[mode]
        matched = (
            iteration >= self.config.min_iterations
            and value is not None
            and threshold is not None
            and (
                value <= float(threshold)
                if mode == "heuristic_distance"
                else value >= float(threshold)
            )
        )
        self.state.consecutive_matches = self.state.consecutive_matches + 1 if matched else 0
        self.state.last_iteration = iteration
        triggered = self.state.consecutive_matches >= self.config.min_consecutive_evaluation
        return EarlyStopDecision(
            triggered=triggered,
            mode=mode,
            value=value,
            threshold=None if threshold is None else float(threshold),
            consecutive_matches=self.state.consecutive_matches,
            required_consecutive=self.config.min_consecutive_evaluation,
            iteration=iteration,
        )


def heuristic_action_accuracy(
    policy_episodes: list[Any],
    heuristic_episodes: list[Any],
) -> tuple[float, int]:
    """Return action-sequence agreement against same-seed heuristic episodes."""
    matches = 0
    compared = 0
    for policy_episode, heuristic_episode in zip(policy_episodes, heuristic_episodes):
        policy_actions = [step.action for step in policy_episode.steps]
        heuristic_actions = [step.action for step in heuristic_episode.steps]
        for index, heuristic_action in enumerate(heuristic_actions):
            compared += 1
            if index < len(policy_actions) and policy_actions[index] == heuristic_action:
                matches += 1
    return (matches / compared if compared else 0.0), compared

def heuristic_action_distance(
    policy_episodes: list[Any],
    heuristic_episodes: list[Any],
    *,
    metric: str,
) -> tuple[float | None, int]:
    """Return mean voxel-unit distance between policy and heuristic next moves."""
    from math import sqrt

    from theseo_anysearch.environments.action_spaces import ACTION_OFFSETS_26

    total_distance = 0.0
    compared = 0
    for policy_episode, heuristic_episode in zip(policy_episodes, heuristic_episodes):
        policy_actions = [step.action for step in policy_episode.steps]
        heuristic_actions = [step.action for step in heuristic_episode.steps]
        for policy_action, heuristic_action in zip(policy_actions, heuristic_actions):
            policy_offset = (
                ACTION_OFFSETS_26[policy_action]
                if 0 <= policy_action < len(ACTION_OFFSETS_26)
                else (0, 0, 0)
            )
            heuristic_offset = (
                ACTION_OFFSETS_26[heuristic_action]
                if 0 <= heuristic_action < len(ACTION_OFFSETS_26)
                else (0, 0, 0)
            )
            differences = tuple(
                abs(policy_axis - heuristic_axis)
                for policy_axis, heuristic_axis in zip(policy_offset, heuristic_offset)
            )
            total_distance += (
                float(sum(differences))
                if metric == "l1"
                else sqrt(sum(difference * difference for difference in differences))
            )
            compared += 1
    return (total_distance / compared if compared else None), compared
