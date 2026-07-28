import pytest
from pydantic import ValidationError

from theseo_anysearch.models import TrainingConfig, TrainingEarlyStopConfig
from theseo_anysearch.rllib.trainer.early_stop import (
    EarlyStopState,
    TrainingEarlyStopController,
    heuristic_action_accuracy,
)


def _config(mode: str, **updates) -> TrainingEarlyStopConfig:
    threshold = {
        "reward": {"reward_threshold": 1.5},
        "goal_finishes": {"goal_finishes_threshold": 3},
        "heuristic_accuracy": {"heuristic_accuracy_threshold": 0.8},
    }[mode]
    return TrainingEarlyStopConfig(
        enabled=True,
        mode=mode,
        consecutive_evaluations=2,
        **threshold,
        **updates,
    )


@pytest.mark.parametrize(
    "mode,values",
    [
        ("reward", {"reward_mean": 1.5, "goal_finishes": 0}),
        ("goal_finishes", {"reward_mean": 0.0, "goal_finishes": 3}),
        (
            "heuristic_accuracy",
            {"reward_mean": 0.0, "goal_finishes": 0, "heuristic_accuracy": 0.8},
        ),
    ],
)
def test_condition_requires_consecutive_evaluations(mode: str, values: dict) -> None:
    controller = TrainingEarlyStopController(_config(mode))
    assert not controller.evaluate(1, **values).triggered
    assert controller.evaluate(2, **values).triggered


def test_failed_evaluation_resets_consecutive_count() -> None:
    controller = TrainingEarlyStopController(_config("reward"))
    controller.evaluate(1, reward_mean=2.0, goal_finishes=0)
    failed = controller.evaluate(2, reward_mean=1.0, goal_finishes=0)
    assert failed.consecutive_matches == 0
    assert not controller.evaluate(3, reward_mean=2.0, goal_finishes=0).triggered


def test_minimum_iteration_is_enforced() -> None:
    controller = TrainingEarlyStopController(_config("reward", minimum_iterations=3))
    controller.evaluate(1, reward_mean=2.0, goal_finishes=0)
    assert not controller.evaluate(2, reward_mean=2.0, goal_finishes=0).triggered
    assert not controller.evaluate(3, reward_mean=2.0, goal_finishes=0).triggered
    assert controller.evaluate(4, reward_mean=2.0, goal_finishes=0).triggered


def test_persisted_state_continues_after_resume() -> None:
    state = EarlyStopState(consecutive_matches=1, last_iteration=4)
    controller = TrainingEarlyStopController(_config("goal_finishes"), state)
    decision = controller.evaluate(5, reward_mean=0.0, goal_finishes=3)
    assert decision.triggered


def test_disabled_configuration_never_stops() -> None:
    controller = TrainingEarlyStopController(TrainingEarlyStopConfig())
    assert not controller.evaluate(100, reward_mean=999.0, goal_finishes=999).triggered


def test_configuration_requires_only_matching_threshold() -> None:
    with pytest.raises(ValidationError, match="exactly its matching threshold"):
        TrainingEarlyStopConfig(
            enabled=True,
            mode="reward",
            reward_threshold=1.0,
            goal_finishes_threshold=2,
        )


def test_goal_threshold_cannot_exceed_evaluation_batch() -> None:
    with pytest.raises(ValidationError, match="cannot exceed"):
        TrainingConfig(
            algorithm="ppo",
            evaluation_episodes=2,
            early_stop={
                "enabled": True,
                "mode": "goal_finishes",
                "goal_finishes_threshold": 3,
            },
        )


class _Step:
    def __init__(self, action: int) -> None:
        self.action = action


class _Episode:
    def __init__(self, actions: list[int]) -> None:
        self.steps = [_Step(action) for action in actions]


def test_heuristic_accuracy_uses_canonical_action_sequences() -> None:
    accuracy, compared = heuristic_action_accuracy(
        [_Episode([1, 2, 9]), _Episode([4])],
        [_Episode([1, 3, 9]), _Episode([4, 5])],
    )
    assert compared == 5
    assert accuracy == pytest.approx(3 / 5)