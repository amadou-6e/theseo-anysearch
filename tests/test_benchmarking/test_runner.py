"""Tests for resource benchmark measurement aggregation."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from theseo_anysearch.benchmarking.models import (
    BenchmarkSample,
    CandidateSummary,
)
from theseo_anysearch.benchmarking.runner import (
    ResourceBenchmarkRunner,
    _TeeTextIO,
    _effective_worker_limit,
    _sampled_steps,
)


def _sample(repeat: int, throughput: float) -> BenchmarkSample:
    return BenchmarkSample(
        phase="environments",
        candidate=2,
        repeat=repeat,
        num_env_runners=1,
        num_envs_per_env_runner=2,
        wall_seconds=2.0,
        sampled_steps=int(throughput * 2),
        steps_per_second=throughput,
        cpu_percent=90.0 + repeat,
        memory_mb=500.0 + repeat,
        gpu_utilization_percent=40.0 + repeat,
    )


def test_candidate_summary_uses_repeat_medians(
        monkeypatch: pytest.MonkeyPatch) -> None:
    runner = object.__new__(ResourceBenchmarkRunner)
    runner._repeats = 3
    runner._measure_iterations = 2
    runner._tracker = MagicMock()
    samples = iter([_sample(1, 100.0), _sample(2, 300.0), _sample(3, 200.0)])
    monkeypatch.setattr(runner, "_measure_repeat", lambda **_: next(samples))

    summary = runner._evaluate_candidate(
        phase="environments",
        candidate=2,
        num_env_runners=1,
        num_envs_per_env_runner=2,
    )

    assert summary.steps_per_second == 200.0
    assert summary.iteration_seconds == 1.0
    assert summary.cpu_percent == 92.0
    assert summary.gpu_utilization_percent == 42.0
    assert len(summary.samples) == 3
    runner._tracker.log_metrics.assert_called_once()


def test_cpu_only_config_disables_gpu_telemetry() -> None:
    config = MagicMock()
    config.training.algorithm = "ppo"
    config.training.require_gpu = False
    config.training.num_gpus = None

    runner = ResourceBenchmarkRunner(
        config,
        Path(__file__),
        output_dir=Path(__file__).parent,
        repeats=1,
        max_envs_per_worker=1,
        max_workers=1,
    )

    assert runner._uses_gpu is False


def test_completed_candidate_refreshes_report_and_notifies_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    runner = object.__new__(ResourceBenchmarkRunner)
    runner._environment_candidates = []
    runner._worker_candidates = []
    runner._output_dir = tmp_path
    runner._max_envs_per_worker = 16
    runner._max_workers = 20
    events = []
    runner._progress_callback = lambda *event: events.append(event)
    write_progress = MagicMock()
    monkeypatch.setattr(
        "theseo_anysearch.benchmarking.runner.write_progress_report",
        write_progress,
    )
    summary = CandidateSummary(
        phase="environments",
        candidate=1,
        num_env_runners=1,
        num_envs_per_env_runner=1,
        steps_per_second=100.0,
        iteration_seconds=1.0,
    )

    runner._candidate_completed(summary)

    assert runner._environment_candidates == [summary]
    write_progress.assert_called_once()
    assert events == [("completed", "environments", 1, summary)]


def test_quiet_mode_suppresses_foreground_ppo_stage_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from theseo_anysearch.rllib.trainer.ppo import _log_stage

    monkeypatch.setenv("ANYSEARCH_QUIET", "1")

    _log_stage("hidden startup detail")

    assert capsys.readouterr().out == ""


def test_quiet_mode_writes_ppo_stage_output_to_redirected_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from theseo_anysearch.rllib.trainer.ppo import _log_stage

    log_path = tmp_path / "benchmark.stdout.log"
    monkeypatch.setenv("ANYSEARCH_QUIET", "1")
    monkeypatch.setenv("ANYSEARCH_QUIET_LOG", str(log_path))

    with log_path.open("w", encoding="utf-8") as handle:
        with redirect_stdout(handle):
            _log_stage("captured startup detail")

    assert "[ppo]" in log_path.read_text(encoding="utf-8")
    assert "captured startup detail" in log_path.read_text(encoding="utf-8")


def test_debug_tee_writes_to_foreground_and_artifact() -> None:
    foreground = StringIO()
    artifact = StringIO()

    stream = _TeeTextIO(foreground, artifact)
    stream.write("Ray startup detail\n")
    stream.flush()

    assert foreground.getvalue() == "Ray startup detail\n"
    assert artifact.getvalue() == "Ray startup detail\n"


def test_rejects_algorithm_without_vector_rollout_controls() -> None:
    config = MagicMock()
    config.training.algorithm = "dqn"

    with pytest.raises(ValueError, match="supports PPO only"):
        ResourceBenchmarkRunner(
            config,
            Path(__file__),
            output_dir=Path(__file__).parent,
        )


def test_rejects_non_positive_duration_budget() -> None:
    config = MagicMock()
    config.training.algorithm = "ppo"

    with pytest.raises(ValueError, match="max_duration_minutes"):
        ResourceBenchmarkRunner(
            config,
            Path(__file__),
            output_dir=Path(__file__).parent,
            max_duration_minutes=0.0,
        )


def test_worker_limit_reserves_one_logical_cpu(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("psutil.cpu_count", lambda logical: 8)

    assert _effective_worker_limit(20) == 7
    assert _effective_worker_limit(4) == 4


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({
            "env_runners": {
                "num_env_steps_sampled_lifetime": 12
            }
        }, 12),
        ({
            "num_env_steps_sampled_lifetime": 13
        }, 13),
        ({
            "timesteps_total": 14
        }, 14),
        ({}, 0),
    ],
)
def test_sampled_steps_supports_rllib_result_shapes(result, expected) -> None:
    assert _sampled_steps(result) == expected
