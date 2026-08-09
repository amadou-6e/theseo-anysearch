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
    action = _StubActionConfig()

    def to_runtime_dict(self) -> dict:
        return {}


class _StubExperiment:
    env = _StubEnvConfig()


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

    def test_env_closed_when_missing_start_pos_raises_before_build(self, tmp_path: Path) -> None:
        env = _FakeVoxelEnv()
        service = _make_service(tmp_path, env)
        trajectory = tmp_path.joinpath("trace.json")
        trajectory.write_text(json.dumps({"steps": []}), encoding="utf-8")

        with pytest.raises(ValueError, match="lacks start_pos or goal_pos"):
            service._replay_trajectory(trajectory, seed=1)
