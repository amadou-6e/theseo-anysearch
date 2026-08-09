"""Tests for policy scoring adapters."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from theseo_anysearch.rllib.explain.scoring import DQNPolicyScorer, MockPolicyScorer


class FakeDQNAlgorithm:
    """Small Algorithm-like object that returns DQN policy info."""

    def __init__(self, q_values: np.ndarray) -> None:
        self.q_values = q_values
        self.calls: list[dict] = []

    def compute_single_action(self, observation, policy_id, explore, full_fetch):
        """Return a deterministic full-fetch DQN response."""

        self.calls.append(
            {
                "observation": observation,
                "policy_id": policy_id,
                "explore": explore,
                "full_fetch": full_fetch,
            }
        )
        return 0, [], {"q_values": self.q_values}


def observation() -> dict[str, np.ndarray]:
    """Build a minimal radial observation for scorer tests."""

    return {
        "steps_remaining": np.asarray([1.0], dtype=np.float32),
        "goal_distance": np.asarray([0.2], dtype=np.float32),
        "goal_direction": np.asarray([0.1, 0.0, 0.2], dtype=np.float32),
        "cursor_pos": np.asarray([0.1, 0.1, 0.1], dtype=np.float32),
        "ray_hits": np.zeros(26, dtype=np.float32),
        "ray_hit_types": np.zeros(26, dtype=np.float32),
    }


def test_mock_policy_scorer_returns_26_scores_for_each_observation():
    scores = np.arange(26, dtype=np.float32)
    scorer = MockPolicyScorer(scores)

    table = scorer.score_all([observation(), observation()])

    assert table.values.shape == (2, 26)
    assert table.action_count() == 26
    assert table.row(1) == pytest.approx(scores)


def test_dqn_policy_scorer_extracts_q_values_from_full_fetch_info():
    q_values = np.linspace(-1.0, 1.0, 26, dtype=np.float32)
    algorithm = FakeDQNAlgorithm(q_values)
    scorer = DQNPolicyScorer(algorithm)

    table = scorer.score_all([observation()])

    assert table.score_type == "q_value"
    assert table.values.shape == (1, 26)
    assert table.values[0] == pytest.approx(q_values)
    assert algorithm.calls[0]["policy_id"] == "default_policy"
    assert algorithm.calls[0]["explore"] is False
    assert algorithm.calls[0]["full_fetch"] is True


def test_dqn_policy_scorer_rejects_missing_q_values():
    algorithm = FakeDQNAlgorithm(np.arange(25, dtype=np.float32))
    scorer = DQNPolicyScorer(algorithm)

    with pytest.raises(ValueError, match="26 Q-values"):
        scorer.score_all([observation()])


def test_dqn_policy_scorer_resolves_latest_project_checkpoint(tmp_path: Path):
    checkpoint_dir = tmp_path.joinpath("checkpoints", "iter_000005")
    checkpoint_dir.mkdir(parents=True)
    latest_path = tmp_path.joinpath("checkpoints", "latest.json")
    latest_path.write_text(
        json.dumps({"path": str(checkpoint_dir), "iteration": 5}),
        encoding="utf-8",
    )

    resolved = DQNPolicyScorer.resolve_checkpoint_dir(tmp_path, "latest")

    assert resolved == checkpoint_dir


def test_dqn_policy_scorer_finds_repo_runtime_geometry_pool_candidate():
    pool_dir = Path("..", "..", "..", "runtime", "geometry_pools", "preview")
    experiment_path = Path("runtime", "experiments", "dqn-maps-zones", "5932954b", "experiment.yaml")

    candidates = DQNPolicyScorer.geometry_pool_candidates(pool_dir, experiment_path)

    assert Path("runtime", "geometry_pools", "preview").resolve() in candidates
