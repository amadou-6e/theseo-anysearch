"""Tests for adaptive resource candidate selection."""

from __future__ import annotations

import pytest

from theseo_anysearch.benchmarking.models import CandidateSummary
from theseo_anysearch.benchmarking.search import (
    adaptive_sweep,
    gpu_saturation_sweep,
)


def _evaluator(values: list[float]):

    def evaluate(candidate: int) -> CandidateSummary:
        return CandidateSummary(
            phase="environments",
            candidate=candidate,
            num_env_runners=1,
            num_envs_per_env_runner=candidate,
            steps_per_second=values[candidate - 1],
            iteration_seconds=1.0,
        )

    return evaluate


def _worker_evaluator(throughput: list[float], gpu: list[float | None]):

    def evaluate(candidate: int) -> CandidateSummary:
        return CandidateSummary(
            phase="workers",
            candidate=candidate,
            num_env_runners=candidate,
            num_envs_per_env_runner=2,
            steps_per_second=throughput[candidate - 1],
            iteration_seconds=1.0,
            gpu_utilization_percent=gpu[candidate - 1],
        )

    return evaluate


def test_stops_after_consecutive_declines_and_selects_global_peak() -> None:
    result = adaptive_sweep(
        phase="environments",
        evaluate=_evaluator([100.0, 150.0, 140.0, 130.0, 200.0]),
        maximum=5,
        decline_patience=2,
        decline_tolerance=0.0,
    )

    assert [item.candidate for item in result.candidates] == [1, 2, 3, 4]
    assert result.peak_candidate == 2
    assert result.peak_steps_per_second == 150.0
    assert "2 consecutive" in result.stop_reason


def test_tolerance_ignores_small_regressions() -> None:
    result = adaptive_sweep(
        phase="environments",
        evaluate=_evaluator([100.0, 99.0, 98.5, 110.0]),
        maximum=4,
        decline_patience=2,
        decline_tolerance=0.02,
    )

    assert len(result.candidates) == 4
    assert result.peak_candidate == 4
    assert result.candidates[-1].speedup == pytest.approx(1.1)


def test_hard_limit_stops_non_declining_search() -> None:
    result = adaptive_sweep(
        phase="environments",
        evaluate=_evaluator([100.0, 110.0, 120.0]),
        maximum=3,
        decline_patience=2,
        decline_tolerance=0.0,
    )

    assert result.peak_candidate == 3
    assert result.stop_reason == "maximum candidate 3 reached"


def test_reports_each_completed_environment_candidate() -> None:
    completed: list[CandidateSummary] = []

    adaptive_sweep(
        phase="environments",
        evaluate=_evaluator([100.0, 150.0]),
        maximum=2,
        decline_patience=2,
        decline_tolerance=0.0,
        on_candidate_completed=completed.append,
    )

    assert [item.candidate for item in completed] == [1, 2]
    assert completed[-1].speedup == pytest.approx(1.5)


def test_environment_sweep_stops_before_candidate_when_budget_expires(
) -> None:
    checks = iter([False, True])
    result = adaptive_sweep(
        phase="environments",
        evaluate=_evaluator([100.0, 110.0, 120.0]),
        maximum=3,
        decline_patience=2,
        decline_tolerance=0.0,
        stop_requested=lambda: next(checks),
    )

    assert [item.candidate for item in result.candidates] == [1, 2]
    assert result.stop_reason == "wall-clock budget reached before next candidate"


def test_worker_sweep_stops_when_gpu_target_is_reached() -> None:
    result = gpu_saturation_sweep(
        evaluate=_worker_evaluator(
            [100.0, 190.0, 180.0, 220.0],
            [40.0, 75.0, 95.0, 99.0],
        ),
        maximum=4,
        max_gpu_utilization=90.0,
    )

    assert [item.candidate for item in result.candidates] == [1, 2, 3]
    assert result.peak_candidate == 2
    assert result.peak_steps_per_second == 190.0
    assert "target 90%" in result.stop_reason


def test_worker_sweep_uses_hard_limit_when_gpu_target_is_not_reached() -> None:
    result = gpu_saturation_sweep(
        evaluate=_worker_evaluator([100.0, 180.0], [None, 70.0]),
        maximum=2,
        max_gpu_utilization=95.0,
    )

    assert len(result.candidates) == 2
    assert result.peak_candidate == 2
    assert result.stop_reason == (
        "maximum candidate 2 reached before GPU utilization reached 95%")


def test_reports_each_completed_worker_candidate() -> None:
    completed: list[CandidateSummary] = []

    gpu_saturation_sweep(
        evaluate=_worker_evaluator([100.0, 180.0], [20.0, 40.0]),
        maximum=2,
        max_gpu_utilization=95.0,
        on_candidate_completed=completed.append,
    )

    assert [item.candidate for item in completed] == [1, 2]
    assert completed[-1].speedup == pytest.approx(1.8)


def test_worker_sweep_stops_before_candidate_when_budget_expires() -> None:
    result = gpu_saturation_sweep(
        evaluate=_worker_evaluator([100.0, 180.0], [20.0, 40.0]),
        maximum=2,
        max_gpu_utilization=95.0,
        stop_requested=lambda: True,
    )

    assert [item.candidate for item in result.candidates] == [1]
    assert result.stop_reason == "wall-clock budget reached before next candidate"


@pytest.mark.parametrize(
    ("maximum", "patience", "tolerance"),
    [(0, 1, 0.0), (1, 0, 0.0), (1, 1, -0.1), (1, 1, 1.0)],
)
def test_rejects_invalid_search_bounds(
    maximum: int,
    patience: int,
    tolerance: float,
) -> None:
    with pytest.raises(ValueError):
        adaptive_sweep(
            phase="environments",
            evaluate=_evaluator([100.0]),
            maximum=maximum,
            decline_patience=patience,
            decline_tolerance=tolerance,
        )
