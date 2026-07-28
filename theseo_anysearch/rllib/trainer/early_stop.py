"""Evaluation-driven early stopping for standard training runs."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from theseo_anysearch.models import TrainingEarlyStopConfig


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
    ) -> EarlyStopDecision:
        if not self.config.enabled:
            return EarlyStopDecision(triggered=False, iteration=iteration)

        values: dict[str, float | None] = {
            "reward": float(reward_mean),
            "goal_finishes": float(goal_finishes),
            "heuristic_accuracy": heuristic_accuracy,
        }
        thresholds: dict[str, float | int | None] = {
            "reward": self.config.reward_threshold,
            "goal_finishes": self.config.goal_finishes_threshold,
            "heuristic_accuracy": self.config.heuristic_accuracy_threshold,
        }
        mode = str(self.config.mode)
        value = values[mode]
        threshold = thresholds[mode]
        matched = (
            iteration >= self.config.minimum_iterations
            and value is not None
            and threshold is not None
            and value >= float(threshold)
        )
        self.state.consecutive_matches = self.state.consecutive_matches + 1 if matched else 0
        self.state.last_iteration = iteration
        triggered = self.state.consecutive_matches >= self.config.consecutive_evaluations
        return EarlyStopDecision(
            triggered=triggered,
            mode=mode,
            value=value,
            threshold=None if threshold is None else float(threshold),
            consecutive_matches=self.state.consecutive_matches,
            required_consecutive=self.config.consecutive_evaluations,
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