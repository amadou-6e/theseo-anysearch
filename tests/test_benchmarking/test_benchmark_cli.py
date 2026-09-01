"""CLI tests for adaptive resource benchmarking."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from theseo_anysearch.benchmarking.models import (
    BenchmarkRecommendation,
    ResourceBenchmarkResult,
    SweepResult,
)
from theseo_anysearch.cli.main import app


def test_resource_benchmark_help_is_discoverable() -> None:
    result = CliRunner().invoke(app, ["benchmark", "resources", "--help"])

    assert result.exit_code == 0
    assert "--decline-patience" in result.output
    assert "--max-envs-per-worker" in result.output
    assert "--max-workers" in result.output
    assert "--max-gpu-utilization" in result.output
    assert "--max-duration-minutes" in result.output
    assert "--debug" in result.output
    assert "--open" in result.output


def test_resource_benchmark_uses_native_panels_and_human_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from rich.console import Console

    console_files = []

    def recording_console(*args, **kwargs):
        console_files.append(kwargs.get("file"))
        return Console(*args, **kwargs)

    monkeypatch.setattr("rich.console.Console", recording_console)
    result = ResourceBenchmarkResult(
        created_at="2026-07-29T00:00:00+00:00",
        config_path="experiment.yaml",
        machine={},
        decline_patience=3,
        decline_tolerance=0.02,
        max_duration_minutes=45.0,
        elapsed_seconds=90.0,
        environment_sweep=SweepResult(
            phase="environments",
            candidates=[],
            peak_candidate=4,
            peak_steps_per_second=180.0,
            stop_reason="stopped after 3 throughput declines",
        ),
        worker_sweep=SweepResult(
            phase="workers",
            candidates=[],
            peak_candidate=6,
            peak_steps_per_second=250.0,
            stop_reason="GPU target reached",
        ),
        recommendation=BenchmarkRecommendation(
            num_env_runners=6,
            num_envs_per_env_runner=4,
            steps_per_second=250.0,
            speedup=2.5,
        ),
    )

    class FakeRunner:

        def __init__(self, *args, **kwargs) -> None:
            self.progress_callback = kwargs["progress_callback"]

        def run(self):
            return result, {
                "html": tmp_path / "report.html",
                "stdout_log": tmp_path / "benchmark.stdout.log",
                "stderr_log": tmp_path / "benchmark.stderr.log",
            }

    monkeypatch.setattr(
        "theseo_anysearch.benchmarking.runner.ResourceBenchmarkRunner",
        FakeRunner,
    )
    config = Path("usage/experiments/train/ppo_tiny_overfit.yaml")

    invocation = CliRunner().invoke(
        app,
        [
            "benchmark",
            "resources",
            str(config),
            "--output-dir",
            str(tmp_path),
            "--max-duration-minutes",
            "45",
        ],
        color=False,
    )

    assert invocation.exit_code == 0, invocation.output
    assert "Starting resource benchmark" in invocation.output
    assert "initializing Ray" in invocation.output
    assert "Resource benchmark complete" in invocation.output
    assert "6 workers × 4 environments" in invocation.output
    assert "benchmark.stdout.log · benchmark.stderr.log" in invocation.output
    assert "Benchmark artifacts:" not in invocation.output
    assert '"stdout_log"' not in invocation.output
    assert console_files[0] is not None
