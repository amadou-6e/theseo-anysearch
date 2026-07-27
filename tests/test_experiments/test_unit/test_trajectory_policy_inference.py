from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from theseo_anysearch.experiments.trajectory import (
    collect_eval_episode,
    collect_eval_episodes,
)


def _observation() -> dict[str, np.ndarray]:
    return {"voxel_count": np.asarray([0.0], dtype=np.float32)}


class _OneStepEnv:
    def __init__(
        self,
        *,
        goal_reached: bool = False,
        terminated: bool = True,
        truncated: bool = False,
    ) -> None:
        self.goal_reached = goal_reached
        self.terminated = terminated
        self.truncated = truncated
        self.actions: list[Any] = []

    def reset(self, seed: int | None = None):
        return _observation(), {}

    def step(self, action: Any):
        self.actions.append(action)
        return (
            _observation(),
            1.0,
            self.terminated,
            self.truncated,
            {"goal_reached": self.goal_reached},
        )

    def close(self) -> None:
        return None


class _RecordingAlgorithm:
    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def compute_single_action(self, observation: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.result


def test_collect_eval_episode_uses_deterministic_policy_inference() -> None:
    algorithm = _RecordingAlgorithm(np.int64(7))
    environment = _OneStepEnv(goal_reached=True)

    episode = collect_eval_episode(
        algorithm,
        {"max_steps": 1},
        env=environment,
    )

    assert algorithm.calls == [
        {"policy_id": "default_policy", "explore": False}
    ]
    assert environment.actions == [np.int64(7)]
    assert episode.steps[0].action == 7
    assert episode.success is True


def test_collect_eval_episode_unwraps_legacy_rllib_action_tuple() -> None:
    algorithm = _RecordingAlgorithm((np.int64(9), [], {}))
    environment = _OneStepEnv(goal_reached=True)

    episode = collect_eval_episode(
        algorithm,
        {"max_steps": 1},
        env=environment,
    )

    assert environment.actions == [np.int64(9)]
    assert episode.steps[0].action == 9


def test_collect_eval_episode_surfaces_policy_inference_failure() -> None:
    class _FailingAlgorithm:
        def compute_single_action(self, observation: Any, **kwargs: Any) -> Any:
            raise RuntimeError("policy inference failed")

    with pytest.raises(RuntimeError, match="policy inference failed"):
        collect_eval_episode(
            _FailingAlgorithm(),
            {"max_steps": 1},
            env=_OneStepEnv(),
        )


def test_collect_eval_episode_does_not_treat_truncation_as_success() -> None:
    episode = collect_eval_episode(
        _RecordingAlgorithm(np.int64(3)),
        {"max_steps": 1},
        env=_OneStepEnv(terminated=False, truncated=True),
    )

    assert episode.success is False


def test_collect_eval_episodes_uses_stable_sequential_seeds(monkeypatch) -> None:
    seeds: list[int | None] = []
    sentinel = object()

    def _collect(algorithm, env_config, *, seed=None):
        seeds.append(seed)
        return sentinel

    monkeypatch.setattr(
        "theseo_anysearch.experiments.trajectory.collect_eval_episode",
        _collect,
    )

    episodes = collect_eval_episodes(object(), {}, 3, seed=12)

    assert episodes == [sentinel, sentinel, sentinel]
    assert seeds == [12, 13, 14]


def test_collect_eval_episodes_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="at least one"):
        collect_eval_episodes(object(), {}, 0)