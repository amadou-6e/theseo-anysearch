"""Tests for PolicyExplanationService replay fidelity and occlusion background
selection, targeting bugs found by audit: seed=0 discarded, environment leaked
on divergence, and degenerate single-observation occlusion baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from theseo_anysearch.rllib.explain.service import (
    PolicyExplanationService,
    resolve_occlusion_background,
)
from theseo_anysearch.rllib.explain.traces import ObservationTrace, ObservationTraceStep


def _observation(cursor: float = 0.1) -> dict[str, np.ndarray]:
    return {
        "steps_remaining": np.asarray([1.0], dtype=np.float32),
        "goal_distance": np.asarray([0.2], dtype=np.float32),
        "goal_direction": np.asarray([0.1, 0.0, 0.2], dtype=np.float32),
        "cursor_pos": np.asarray([cursor, cursor, cursor], dtype=np.float32),
        "ray_hits": np.zeros(26, dtype=np.float32),
        "ray_hit_types": np.zeros(26, dtype=np.float32),
    }


def _one_step_trace() -> ObservationTrace:
    observation = _observation()
    return ObservationTrace(
        [
            ObservationTraceStep(
                step=0,
                observation=observation,
                action=0,
                reward=0.0,
                cursor_before=(0.1, 0.1, 0.1),
                cursor_after=(0.1, 0.1, 0.1),
                done=False,
                collision=False,
            )
        ],
        algorithm="dqn",
    )


def _two_step_trace() -> ObservationTrace:
    step0 = _observation(0.1)
    step1 = _observation(0.2)
    return ObservationTrace(
        [
            ObservationTraceStep(
                step=0,
                observation=step0,
                action=0,
                reward=0.0,
                cursor_before=(0.1, 0.1, 0.1),
                cursor_after=(0.2, 0.2, 0.2),
                done=False,
                collision=False,
            ),
            ObservationTraceStep(
                step=1,
                observation=step1,
                action=0,
                reward=0.0,
                cursor_before=(0.2, 0.2, 0.2),
                cursor_after=(0.2, 0.2, 0.2),
                done=True,
                collision=False,
            ),
        ],
        algorithm="dqn",
    )


class TestResolveOcclusionBackground:
    """`background=trace/mean` must never collapse to the observation itself."""

    def test_single_step_trace_background_raises(self) -> None:
        with pytest.raises(ValueError, match="at least two background observations|degenerate"):
            resolve_occlusion_background("trace", _one_step_trace())

    def test_auto_uses_zeros_for_single_step_trace(self) -> None:
        background = resolve_occlusion_background("auto", _one_step_trace())

        assert len(background) == 1
        assert all(np.all(value == 0.0) for value in background[0].values())

    def test_auto_uses_trace_for_multi_step_trace(self) -> None:
        background = resolve_occlusion_background("auto", _two_step_trace())

        assert len(background) == 2

    def test_single_step_trace_mean_background_raises(self) -> None:
        with pytest.raises(ValueError, match="degenerate"):
            resolve_occlusion_background("mean", _one_step_trace())

    def test_multi_step_trace_background_succeeds(self) -> None:
        background = resolve_occlusion_background("trace", _two_step_trace())

        assert len(background) == 2

    def test_zeros_background_allowed_even_for_single_step_trace(self) -> None:
        """`zeros` is a synthetic, non-degenerate baseline distinct from the
        observation under attribution, so a one-step trace is fine."""

        background = resolve_occlusion_background("zeros", _one_step_trace())

        assert len(background) == 1
        assert all(np.all(value == 0.0) for value in background[0].values())

    def test_unsupported_background_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported occlusion background"):
            resolve_occlusion_background("bogus", _two_step_trace())


class _StubActionConfig:
    mode = "discrete_26"


class _StubEnvConfig:
    seed = 999

    def __init__(self) -> None:
        self.action = _StubActionConfig()

    def to_runtime_dict(self) -> dict:
        return {}


class _StubExperiment:
    def __init__(self) -> None:
        self.env = _StubEnvConfig()


class _FakeRustEnv:
    def __init__(self, cursor: tuple[int, int, int]) -> None:
        self._cursor = cursor

    def cursor_pos(self) -> tuple[int, int, int]:
        return self._cursor


class _FakeVoxelEnv:
    """Minimal stand-in for VoxelEnv, tracking reset seed and close calls."""

    def __init__(self, cursor: tuple[int, int, int] = (0, 0, 0)) -> None:
        self._rust_env = _FakeRustEnv(cursor)
        self.reset_seeds: list[int | None] = []
        self.closed = False

    def reset(self, seed: int | None = None):
        self.reset_seeds.append(seed)
        return _observation(), {}

    def step(self, action: int):
        return _observation(), 0.0, True, False, {}

    def close(self) -> None:
        self.closed = True


def _make_service(run_dir: Path, env: _FakeVoxelEnv) -> PolicyExplanationService:
    service = PolicyExplanationService.__new__(PolicyExplanationService)
    service.run_dir = run_dir
    service.experiment = _StubExperiment()
    service.checkpoint = "mock"
    service.scorer = None
    service._build_env = lambda overrides=None: env  # type: ignore[method-assign]
    return service


def _write_trajectory(path: Path, *, start=(0, 0, 0), goal=(1, 1, 1), steps=()) -> None:
    path.write_text(
        json.dumps({"start_pos": list(start), "goal_pos": list(goal), "steps": list(steps)}),
        encoding="utf-8",
    )


class TestReplaySeedHandling:
    """seed=0 is a legitimate explicit seed, not "no seed requested"."""

    def test_explicit_seed_zero_is_forwarded_to_env_reset(self, tmp_path: Path) -> None:
        env = _FakeVoxelEnv(cursor=(0, 0, 0))
        service = _make_service(tmp_path, env)
        trajectory = tmp_path.joinpath("trace.json")
        _write_trajectory(trajectory, start=(0, 0, 0))

        with pytest.raises(ValueError, match="contains no steps"):
            service._replay_trajectory(trajectory, seed=0)

        assert env.reset_seeds == [0]

    def test_missing_seed_falls_back_to_experiment_default(self, tmp_path: Path) -> None:
        env = _FakeVoxelEnv(cursor=(0, 0, 0))
        service = _make_service(tmp_path, env)
        trajectory = tmp_path.joinpath("trace.json")
        _write_trajectory(trajectory, start=(0, 0, 0))

        with pytest.raises(ValueError, match="contains no steps"):
            service._replay_trajectory(trajectory, seed=None)

        assert env.reset_seeds == [999]

    def test_nested_trajectory_envelope_and_npy_geometry_are_loaded(
        self, tmp_path: Path
    ) -> None:
        env = _FakeVoxelEnv(cursor=(0, 0, 0))
        service = _make_service(tmp_path, env)
        captured: dict = {}
        service._build_env = lambda overrides=None: (captured.update(overrides or {}) or env)  # type: ignore[method-assign]
        np.save(tmp_path.joinpath("filled.npy"), np.asarray([[2, 3, 4]], dtype=np.int64))
        trajectory = tmp_path.joinpath("trace.json")
        trajectory.write_text(
            json.dumps(
                {
                    "experiment_name": "example",
                    "episode": {
                        "start_pos": [0, 0, 0],
                        "goal_pos": [1, 1, 1],
                        "init_filled_file": "filled.npy",
                        "steps": [],
                    },
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="contains no steps"):
            service._replay_trajectory(trajectory, seed=1)

        assert captured["geometry_boxes"] == [[2, 3, 4, 2, 3, 4]]


class TestActionAlignedFeatureSchema:
    """Reported directions must follow the selected policy action space."""

    def test_discrete_18_schema_uses_discrete_18_offsets(self, tmp_path: Path) -> None:
        service = _make_service(tmp_path, _FakeVoxelEnv())
        service.experiment.env.action.mode = "discrete_18"

        schema = service.feature_schema(_observation())

        assert len(schema.action_directions) == 18


class TestReplayEnvironmentCleanup:
    """The environment must be closed even when replay diverges."""

    def test_env_closed_when_reset_diverges(self, tmp_path: Path) -> None:
        env = _FakeVoxelEnv(cursor=(5, 5, 5))
        service = _make_service(tmp_path, env)
        trajectory = tmp_path.joinpath("trace.json")
        _write_trajectory(trajectory, start=(0, 0, 0))

        with pytest.raises(ValueError, match="diverged at reset"):
            service._replay_trajectory(trajectory, seed=1)

        assert env.closed is True

    def test_replay_limit_does_not_validate_unrequested_later_steps(
        self, tmp_path: Path
    ) -> None:
        env = _FakeVoxelEnv(cursor=(0, 0, 0))
        service = _make_service(tmp_path, env)
        trajectory = tmp_path.joinpath("trace.json")
        _write_trajectory(
            trajectory,
            start=(0, 0, 0),
            steps=(
                {"action": 0, "cursor_x": 0, "cursor_y": 0, "cursor_z": 0},
                {"action": 0, "cursor_x": 9, "cursor_y": 9, "cursor_z": 9},
            ),
        )

        trace = service._replay_trajectory(trajectory, seed=1, replay_limit=1)

        assert len(trace) == 1

    def test_env_closed_when_missing_start_pos_raises_before_build(self, tmp_path: Path) -> None:
        env = _FakeVoxelEnv()
        service = _make_service(tmp_path, env)
        trajectory = tmp_path.joinpath("trace.json")
        trajectory.write_text(json.dumps({"steps": []}), encoding="utf-8")

        with pytest.raises(ValueError, match="lacks start_pos or goal_pos"):
            service._replay_trajectory(trajectory, seed=1)
