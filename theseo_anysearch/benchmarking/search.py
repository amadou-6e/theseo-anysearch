"""Adaptive search utilities for resource benchmarks."""

from __future__ import annotations

from collections.abc import Callable

from theseo_anysearch.benchmarking.models import CandidateSummary, SweepResult

CandidateEvaluator = Callable[[int], CandidateSummary]
StopRequested = Callable[[], bool]
CandidateCompleted = Callable[[CandidateSummary], None]


def gpu_saturation_sweep(
    *,
    evaluate: CandidateEvaluator,
    maximum: int,
    max_gpu_utilization: float,
    stop_requested: StopRequested | None = None,
    on_candidate_completed: CandidateCompleted | None = None,
) -> SweepResult:
    """Increase workers until measured GPU utilization reaches the target."""
    if maximum < 1:
        raise ValueError("maximum must be at least 1")
    if not 0.0 < max_gpu_utilization <= 100.0:
        raise ValueError("max_gpu_utilization must be in (0, 100]")

    candidates: list[CandidateSummary] = []
    best: CandidateSummary | None = None
    baseline_steps_per_second: float | None = None
    stop_reason = (
        f"maximum candidate {maximum} reached before GPU utilization reached "
        f"{max_gpu_utilization:g}%")

    for candidate in range(1, maximum + 1):
        if candidates and stop_requested is not None and stop_requested():
            stop_reason = "wall-clock budget reached before next candidate"
            break
        summary = evaluate(candidate)
        if summary.phase != "workers" or summary.candidate != candidate:
            raise ValueError(
                "candidate evaluator returned mismatched metadata")

        if baseline_steps_per_second is None:
            baseline_steps_per_second = summary.steps_per_second
        speedup = (summary.steps_per_second / baseline_steps_per_second
                   if baseline_steps_per_second > 0.0 else 0.0)
        summary = summary.model_copy(update={"speedup": speedup})
        candidates.append(summary)
        if on_candidate_completed is not None:
            on_candidate_completed(summary)

        if best is None or summary.steps_per_second > best.steps_per_second:
            best = summary

        utilization = summary.gpu_utilization_percent
        if utilization is not None and utilization >= max_gpu_utilization:
            stop_reason = (
                f"GPU utilization reached {utilization:g}% at candidate "
                f"{candidate} (target {max_gpu_utilization:g}%)")
            break

    assert best is not None
    return SweepResult(
        phase="workers",
        candidates=candidates,
        peak_candidate=best.candidate,
        peak_steps_per_second=best.steps_per_second,
        stop_reason=stop_reason,
    )


def adaptive_sweep(
    *,
    phase: str,
    evaluate: CandidateEvaluator,
    maximum: int,
    decline_patience: int,
    decline_tolerance: float,
    stop_requested: StopRequested | None = None,
    on_candidate_completed: CandidateCompleted | None = None,
) -> SweepResult:
    """Evaluate increasing candidates until throughput declines repeatedly.

    A decline is measured against the running best. Values within
    ``decline_tolerance`` of that best are treated as noise rather than declines.
    The recommendation is always the highest-throughput candidate observed.
    """
    if maximum < 1:
        raise ValueError("maximum must be at least 1")
    if decline_patience < 1:
        raise ValueError("decline_patience must be at least 1")
    if not 0.0 <= decline_tolerance < 1.0:
        raise ValueError("decline_tolerance must be in [0, 1)")

    candidates: list[CandidateSummary] = []
    best: CandidateSummary | None = None
    baseline_steps_per_second: float | None = None
    consecutive_declines = 0
    stop_reason = f"maximum candidate {maximum} reached"

    for candidate in range(1, maximum + 1):
        if candidates and stop_requested is not None and stop_requested():
            stop_reason = "wall-clock budget reached before next candidate"
            break
        summary = evaluate(candidate)
        if summary.phase != phase or summary.candidate != candidate:
            raise ValueError(
                "candidate evaluator returned mismatched metadata")

        if baseline_steps_per_second is None:
            baseline_steps_per_second = summary.steps_per_second
        speedup = (summary.steps_per_second / baseline_steps_per_second
                   if baseline_steps_per_second > 0.0 else 0.0)
        summary = summary.model_copy(update={"speedup": speedup})
        candidates.append(summary)
        if on_candidate_completed is not None:
            on_candidate_completed(summary)

        if best is None or summary.steps_per_second > best.steps_per_second:
            best = summary
            consecutive_declines = 0
            continue

        decline_threshold = best.steps_per_second * (1.0 - decline_tolerance)
        if summary.steps_per_second < decline_threshold:
            consecutive_declines += 1
        else:
            consecutive_declines = 0

        if consecutive_declines >= decline_patience:
            stop_reason = (
                f"throughput declined for {decline_patience} consecutive candidates"
            )
            break

    assert best is not None
    return SweepResult(
        phase=phase,
        candidates=candidates,
        peak_candidate=best.candidate,
        peak_steps_per_second=best.steps_per_second,
        stop_reason=stop_reason,
    )
