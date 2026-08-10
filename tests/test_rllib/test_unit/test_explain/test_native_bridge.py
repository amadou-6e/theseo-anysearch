"""Tests for the native explainability JSON-lines bridge."""

import json
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from theseo_anysearch.rllib.explain.native_bridge import (
    _json_observation,
    _load_observation_file,
    _schema,
)


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


def test_load_observation_file_detects_json(tmp_path: Path) -> None:
    """JSON mappings are detected and flattened against the policy schema."""

    source = Path(tmp_path, "observation.json")
    source.write_text(
        json.dumps(
            {
                "goal_direction": [0.0, 1.0, -1.0],
                "local_grid": np.zeros((3, 3, 3), dtype=np.float32).tolist(),
            }
        ),
        encoding="utf-8",
    )

    observation, detected_format = _load_observation_file(source, _schema(SessionStub()))

    assert detected_format == "JSON"
    assert observation["goal_direction"] == [0.0, 1.0, -1.0]
    assert len(observation["local_grid"]) == 27


def test_load_observation_file_detects_numpy_archive(tmp_path: Path) -> None:
    """NPZ field names provide an unambiguous dictionary observation."""

    source = Path(tmp_path, "observation.npz")
    np.savez(
        source,
        goal_direction=np.asarray([0.0, 1.0, -1.0], dtype=np.float32),
        local_grid=np.zeros(27, dtype=np.float32),
    )

    observation, detected_format = _load_observation_file(source, _schema(SessionStub()))

    assert detected_format == "NumPy NPZ"
    assert observation["goal_direction"] == [0.0, 1.0, -1.0]
    assert len(observation["local_grid"]) == 27


def test_load_observation_file_rejects_missing_fields(tmp_path: Path) -> None:
    """Imports never silently fill fields omitted by the source file."""

    source = Path(tmp_path, "incomplete.json")
    source.write_text(json.dumps({"goal_direction": [0.0, 1.0, -1.0]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing=.*local_grid"):
        _load_observation_file(source, _schema(SessionStub()))
