"""Tests for resource benchmark artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from theseo_anysearch.benchmarking.model import PredictedCandidate, StageCosts
from theseo_anysearch.benchmarking.models import (
    BenchmarkRecommendation,
    BenchmarkSample,
    CandidateSummary,
    PredictionSummary,
    ResourceBenchmarkResult,
    SweepResult,
)
from theseo_anysearch.benchmarking.report import (
    write_benchmark_artifacts,
    write_progress_report,
)


def _candidate(phase: str, candidate: int, steps: float) -> CandidateSummary:
    workers = candidate if phase == "workers" else 1
    envs = 2 if phase == "workers" else candidate
    sample = BenchmarkSample(
        phase=phase,
        candidate=candidate,
        repeat=1,
        num_env_runners=workers,
        num_envs_per_env_runner=envs,
        wall_seconds=2.0,
        sampled_steps=int(steps * 2),
        steps_per_second=steps,
        cpu_percent=95.0,
        memory_mb=512.0,
        gpu_utilization_percent=50.0,
    )
    return CandidateSummary(
        phase=phase,
        candidate=candidate,
        num_env_runners=workers,
        num_envs_per_env_runner=envs,
        steps_per_second=steps,
        speedup=steps / 100.0,
        iteration_seconds=2.0,
        cpu_percent=95.0,
        memory_mb=512.0,
        gpu_utilization_percent=50.0,
        samples=[sample],
    )


def test_writes_machine_readable_and_interactive_artifacts(
        tmp_path: Path) -> None:
    environments = [
        _candidate("environments", 1, 100.0),
        _candidate("environments", 2, 180.0)
    ]
    workers = [
        _candidate("workers", 1, 180.0),
        _candidate("workers", 2, 250.0)
    ]
    result = ResourceBenchmarkResult(
        created_at="2026-07-29T00:00:00+00:00",
        config_path="experiment.yaml",
        machine={"logical_cpus": 8},
        decline_patience=2,
        decline_tolerance=0.02,
        max_duration_minutes=12.0,
        elapsed_seconds=90.0,
        environment_sweep=SweepResult(
            phase="environments",
            candidates=environments,
            peak_candidate=2,
            peak_steps_per_second=180.0,
            stop_reason="maximum candidate 2 reached",
        ),
        worker_sweep=SweepResult(
            phase="workers",
            candidates=workers,
            peak_candidate=2,
            peak_steps_per_second=250.0,
            stop_reason="maximum candidate 2 reached",
        ),
        recommendation=BenchmarkRecommendation(
            num_env_runners=2,
            num_envs_per_env_runner=2,
            steps_per_second=250.0,
            speedup=2.5,
        ),
    )

    artifacts = write_benchmark_artifacts(result, tmp_path)

    assert set(artifacts) == {"json", "csv", "yaml", "html"}
    assert all(path.is_file() for path in artifacts.values())
    assert json.loads(artifacts["json"].read_text()
                      )["recommendation"]["num_env_runners"] == 2
    assert "num_envs_per_env_runner: 2" in artifacts["yaml"].read_text()
    assert "AnySearch resource benchmark" in artifacts["html"].read_text(
        encoding="utf-8")
    assert "GPU target" in artifacts["html"].read_text(encoding="utf-8")
    assert '"legend":"legend3"' in artifacts["html"].read_text(
        encoding="utf-8")
    assert '"legend":"legend4"' in artifacts["html"].read_text(
        encoding="utf-8")
    assert '"texttemplate":"%{text:.1f}%"' in artifacts["html"].read_text(
        encoding="utf-8")
    assert "Automatic environment sweep" in artifacts["html"].read_text(
        encoding="utf-8")
    assert "Automatic worker sweep" in artifacts["html"].read_text(
        encoding="utf-8")
    assert "Wall-clock budget: 12 minutes" in artifacts["html"].read_text(
        encoding="utf-8")
    assert 'href="benchmark.stdout.log"' in artifacts["html"].read_text(
        encoding="utf-8")
    assert 'href="benchmark.stderr.log"' in artifacts["html"].read_text(
        encoding="utf-8")
    with artifacts["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert rows[-1]["speedup"] == "2.5"


def test_html_includes_prediction_panel_when_calibration_ran(
        tmp_path: Path) -> None:
    environments = [_candidate("environments", 1, 100.0)]
    workers = [_candidate("workers", 1, 180.0)]
    result = ResourceBenchmarkResult(
        created_at="2026-07-29T00:00:00+00:00",
        config_path="experiment.yaml",
        machine={"logical_cpus": 8},
        decline_patience=2,
        decline_tolerance=0.02,
        max_duration_minutes=12.0,
        elapsed_seconds=90.0,
        environment_sweep=SweepResult(
            phase="environments",
            candidates=environments,
            peak_candidate=1,
            peak_steps_per_second=100.0,
            stop_reason="maximum candidate 1 reached",
        ),
        worker_sweep=SweepResult(
            phase="workers",
            candidates=workers,
            peak_candidate=1,
            peak_steps_per_second=180.0,
            stop_reason="maximum candidate 1 reached",
        ),
        recommendation=BenchmarkRecommendation(
            num_env_runners=1,
            num_envs_per_env_runner=1,
            steps_per_second=180.0,
            speedup=1.8,
        ),
        prediction=PredictionSummary(
            stage_costs=StageCosts(
                env_step_seconds=0.001,
                inference_seconds_per_env=0.0005,
                gil_contention_ratio=0.05,
                transfer_seconds_per_mb=0.002,
                avg_sample_mb=0.01,
                learner_seconds_per_batch=0.4,
                train_batch_size=512,
                scheduler_queue_seconds=0.003,
            ),
            correction_exponent=0.2,
            environment_predicted=PredictedCandidate(
                num_env_runners=1,
                num_envs_per_env_runner=1,
                predicted_steps_per_second=95.0,
                bottleneck="learner",
            ),
            worker_predicted=PredictedCandidate(
                num_env_runners=1,
                num_envs_per_env_runner=1,
                predicted_steps_per_second=170.0,
                bottleneck="learner",
            ),
            calibration_seconds=12.5,
        ),
    )

    artifacts = write_benchmark_artifacts(result, tmp_path)

    document = artifacts["html"].read_text(encoding="utf-8")
    assert "Calibration: predicted vs. confirmed throughput" in document
    assert "Calibration prediction" in document
    assert "Confirmed sweep peak" in document
    assert "bottleneck: learner" in document
    assert "oversubscription correction 0.20" in document


def test_progress_report_exposes_completed_ticks_and_refreshes(
        tmp_path: Path) -> None:
    path = write_progress_report(
        environment_candidates=[_candidate("environments", 1, 100.0)],
        worker_candidates=[],
        output_dir=tmp_path,
        max_envs_per_worker=16,
        max_workers=20,
    )

    document = path.read_text(encoding="utf-8")
    assert 'http-equiv="refresh" content="5"' in document
    assert ("Environment ticks: 1\\u002f16; worker ticks: 0\\u002f20"
            in document)
    assert "Waiting for the environment sweep to finish" in document
    assert '"texttemplate":"%{text:.1f}%"' in document
    assert 'href="benchmark.stdout.log"' in document
    assert 'href="benchmark.stderr.log"' in document
    assert not tmp_path.joinpath("report.progress.html").exists()
