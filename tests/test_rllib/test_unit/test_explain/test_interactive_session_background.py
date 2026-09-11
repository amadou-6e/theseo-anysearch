"""Regression test: InteractiveExplanationSession must not build its occlusion
background from a single frozen observation.

A single-observation occlusion background collapses to the observation being
explained, silently reporting a 0.0 attribution for every group the user
didn't happen to edit away from the frozen baseline -- exactly the bug
resolve_occlusion_background was added to reject for trace-based explanations
in an earlier fix. The interactive session bypassed that guard entirely by
calling OcclusionExplainer directly with one observation.
"""
from __future__ import annotations

import numpy as np
import pytest

from theseo_anysearch.rllib.explain.ui import session as session_module
from theseo_anysearch.rllib.explain.ui.session import InteractiveExplanationSession


class _StubService:
    """Records every initial_observation() seed and returns a distinct value per seed."""

    def __init__(self, run_dir, checkpoint="latest") -> None:
        self.checkpoint = checkpoint
        self.requested_seeds: list[int | None] = []

    def observation_space(self):
        return object()

    def initial_observation(self, seed: int | None = None) -> dict[str, np.ndarray]:
        self.requested_seeds.append(seed)
        # Vary the observation with the seed so a single frozen background is
        # trivially distinguishable from a diverse background pool.
        value = float(seed or 0)
        return {"goal_distance": np.asarray([value], dtype=np.float32)}


@pytest.fixture()
def session(monkeypatch, tmp_path) -> InteractiveExplanationSession:
    monkeypatch.setattr(session_module, "PolicyExplanationService", _StubService)
    return InteractiveExplanationSession(tmp_path, checkpoint="latest")


def test_background_pool_has_more_than_one_observation(session) -> None:
    assert len(session._background) >= 2, (
        "a single-observation background silently zeroes every group "
        "attribution for fields the user didn't edit"
    )


def test_background_pool_is_actually_diverse_not_the_same_observation_repeated(
    session,
) -> None:
    values = {obs["goal_distance"].item() for obs in session._background}
    assert len(values) > 1, "background pool must sample distinct states, not repeat one"


def test_background_pool_uses_distinct_seeds(session) -> None:
    assert len(set(session.service.requested_seeds)) == len(session.service.requested_seeds)
