"""Tests for policy-driven observation trace collection."""

from __future__ import annotations

import numpy as np

from theseo_anysearch.rllib.explain.scoring import MockPolicyScorer
from theseo_anysearch.rllib.explain.traces import PolicyEvaluationTraceCollector


ACTION_PLUS_X = 21


class TwoStepCollisionEnv:
    """Tiny Gymnasium-like env with one collision transition."""

    def __init__(self) -> None:
        self.step_count = 0

    def reset(self, seed=None):
        """Return the initial observation."""

        self.step_count = 0
        return self._observation(cursor=0.1), {}

    def step(self, action):
        """Keep cursor fixed for the first step, then terminate."""

        self.step_count += 1
        if self.step_count == 1:
            return self._observation(cursor=0.1), -0.5, False, False, {}
        return self._observation(cursor=0.2), -0.05, True, False, {}

    def _observation(self, cursor: float) -> dict[str, np.ndarray]:
        """Build one radial observation."""

        return {
            "steps_remaining": np.asarray([1.0], dtype=np.float32),
            "goal_distance": np.asarray([0.2], dtype=np.float32),
            "goal_direction": np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
            "cursor_pos": np.asarray([cursor, cursor, cursor], dtype=np.float32),
            "ray_hits": np.zeros(26, dtype=np.float32),
            "ray_hit_types": np.zeros(26, dtype=np.float32),
        }


def test_policy_evaluation_trace_collector_records_collision_step():
    scores = np.zeros(26, dtype=np.float32)
    scores[ACTION_PLUS_X] = 1.0
    collector = PolicyEvaluationTraceCollector(
        TwoStepCollisionEnv(),
        MockPolicyScorer(scores),
        algorithm="mock",
    )

    trace = collector.collect(seed=7, max_steps=2)

    assert len(trace) == 2
    assert trace.step(0).action == ACTION_PLUS_X
    assert trace.step(0).is_collision() is True
    assert trace.step(1).done is True
    assert trace.step(1).is_collision() is False
