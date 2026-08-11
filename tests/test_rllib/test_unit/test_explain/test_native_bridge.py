"""Tests for the native explainability JSON-lines bridge."""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces

from theseo_anysearch.rllib.explain import native_bridge
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

    observation, detected_format = _load_observation_file(
        source, _schema(SessionStub()), SessionStub.observation_space
    )

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

    observation, detected_format = _load_observation_file(
        source, _schema(SessionStub()), SessionStub.observation_space
    )

    assert detected_format == "NumPy NPZ"
    assert observation["goal_direction"] == [0.0, 1.0, -1.0]
    assert len(observation["local_grid"]) == 27


def test_load_observation_file_rejects_missing_fields(tmp_path: Path) -> None:
    """Imports never silently fill fields omitted by the source file."""

    source = Path(tmp_path, "incomplete.json")
    source.write_text(json.dumps({"goal_direction": [0.0, 1.0, -1.0]}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing=.*local_grid"):
        _load_observation_file(
            source, _schema(SessionStub()), SessionStub.observation_space
        )


def test_load_observation_file_rejects_out_of_bounds_values(tmp_path: Path) -> None:
    """A same-total-size import must still be rejected outside declared bounds.

    Regression test: the previous implementation only checked total element
    count, so an out-of-range or wrongly-shaped-but-same-size import was
    silently accepted.
    """

    source = Path(tmp_path, "out_of_bounds.json")
    source.write_text(
        json.dumps(
            {
                "goal_direction": [0.0, 5.0, -1.0],
                "local_grid": np.zeros(27, dtype=np.float32).tolist(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside its declared bounds"):
        _load_observation_file(
            source, _schema(SessionStub()), SessionStub.observation_space
        )


def test_load_observation_file_rejects_wrong_size(tmp_path: Path) -> None:
    """Total element count must still match; unrelated-sized imports are rejected."""

    source = Path(tmp_path, "wrong_size.json")
    source.write_text(
        json.dumps(
            {
                "goal_direction": [0.0, 1.0],
                "local_grid": np.zeros(27, dtype=np.float32).tolist(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="goal_direction.*value"):
        _load_observation_file(
            source, _schema(SessionStub()), SessionStub.observation_space
        )


class _FakeInteractiveSession:
    """Stand-in for InteractiveExplanationSession used by the serve() regression test."""

    observation_space = SessionStub.observation_space

    def initial_observation(self, seed=None):
        return {
            "goal_direction": np.zeros(3, dtype=np.float32),
            "local_grid": np.zeros(27, dtype=np.float32),
        }


def test_serve_writes_reset_observation_response_to_real_stdout(
    monkeypatch, tmp_path: Path
) -> None:
    """Regression test: reset_observation/load_observation_file responses were
    previously written while stdout was redirected to stderr, so Rust's
    blocking read on the real stdout pipe would hang forever."""

    monkeypatch.setattr(native_bridge, "resolve_run_dir", lambda ref: tmp_path)
    monkeypatch.setattr(
        native_bridge,
        "InteractiveExplanationSession",
        lambda run_dir, checkpoint: _FakeInteractiveSession(),
    )
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps({"command": "reset_observation"}) + "\n")
    )
    captured_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_stdout)

    native_bridge.serve("dummy-run", "latest")

    lines = [line for line in captured_stdout.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2, "expected a 'ready' response and a reset_observation response"
    reset_response = json.loads(lines[1])
    assert reset_response["ok"] is True
    assert "observation" in reset_response


def test_serve_recovers_from_malformed_request_line(monkeypatch, tmp_path: Path) -> None:
    """Regression test: json.loads/.get('command') used to run outside the
    try/except, so a malformed line killed the whole persistent process
    instead of returning one error response and continuing."""

    monkeypatch.setattr(native_bridge, "resolve_run_dir", lambda ref: tmp_path)
    monkeypatch.setattr(
        native_bridge,
        "InteractiveExplanationSession",
        lambda run_dir, checkpoint: _FakeInteractiveSession(),
    )
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            "not valid json\n" + json.dumps({"command": "reset_observation"}) + "\n"
        ),
    )
    captured_stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured_stdout)

    native_bridge.serve("dummy-run", "latest")

    lines = [line for line in captured_stdout.getvalue().splitlines() if line.strip()]
    assert len(lines) == 3, "expected ready, one error response, then the recovered response"
    error_response = json.loads(lines[1])
    assert error_response["ok"] is False
    recovered_response = json.loads(lines[2])
    assert recovered_response["ok"] is True
