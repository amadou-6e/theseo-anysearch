"""Tests for the native explainability JSON-lines bridge."""

import numpy as np
from gymnasium import spaces

from theseo_anysearch.rllib.explain.native_bridge import _json_observation, _schema


class SessionStub:
    """Minimal session exposing an authoritative observation space."""

    observation_space = spaces.Dict(
        {
            "goal_direction": spaces.Box(-1.0, 1.0, (3,), np.float32),
            "local_grid": spaces.Box(0.0, 1.0, (27,), np.float32),
        }
    )


def test_json_observation_flattens_network_fields() -> None:
    """Native UI payloads contain flat finite float lists."""

    payload = _json_observation(
        {
            "goal_direction": np.asarray([[0.0, 1.0, -1.0]], dtype=np.float32),
            "local_grid": np.zeros(27, dtype=np.float32),
        }
    )

    assert payload["goal_direction"] == [0.0, 1.0, -1.0]
    assert len(payload["local_grid"]) == 27


def test_schema_exposes_authoritative_bounds_and_shapes() -> None:
    """Rust controls receive bounds from the checkpoint environment schema."""

    payload = _schema(SessionStub())

    assert payload["goal_direction"]["shape"] == [3]
    assert payload["goal_direction"]["low"] == [-1.0, -1.0, -1.0]
    assert payload["local_grid"]["high"] == [1.0] * 27
