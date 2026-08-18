"""Roofline-style predictive model for adaptive resource benchmark calibration.

Predicts steps/second for untested ``(num_env_runners, num_envs_per_env_runner)``
candidates from a handful of calibration measurements, so the adaptive sweeps in
``search.py`` only need to confirm a narrow band instead of scanning from 1 upward.
The model treats each pipeline stage (native env stepping, policy inference, GIL
contention, Ray object-store transfer, GPU learner throughput, Ray scheduler queuing)
as an independent rate; the slowest stage caps overall throughput and is reported as
the bottleneck.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

BottleneckStage = Literal[
    "env_step",
    "inference",
    "gil",
    "transfer",
    "learner",
    "scheduler",
]

# Top-level rates that compete in the roofline min(). "env_step", "inference",
# and "gil" are not independent rates here: env stepping and policy inference
# run sequentially on the same thread, so they only ever gate throughput
# together, as one "sampling" rate. When sampling is the overall bottleneck,
# _sampling_bottleneck_label() attributes it to whichever of the three raw
# per-env time components (step, inference, GIL-contention overhead) is
# largest, purely for reporting.
_RATE_STAGES = ("sampling", "transfer", "learner", "scheduler")

# Rate stages whose value scales with num_env_runners and is therefore subject
# to the CPU-oversubscription correction fit by fit_contention_correction().
_PARALLEL_RATE_STAGES = frozenset({"sampling", "scheduler"})


class StageCosts(BaseModel):
    """Per-unit costs for each pipeline stage, gathered during calibration."""

    model_config = ConfigDict(extra="forbid")

    env_step_seconds: float = Field(gt=0.0, description="Native env.step() wall time, single env, single thread.")
    inference_seconds_per_env: float = Field(gt=0.0, description="Policy forward-pass wall time per env in a batch.")
    gil_contention_ratio: float | None = Field(
        default=None, ge=0.0, lt=1.0,
        description="Fraction of wall time the sampling thread was starved of the GIL.")
    transfer_seconds_per_mb: float = Field(gt=0.0, description="Ray object-store round trip cost per MB.")
    avg_sample_mb: float = Field(gt=0.0, description="Average size of one rollout batch transfer.")
    learner_seconds_per_batch: float = Field(gt=0.0, description="GPU train_step() wall time for one batch.")
    train_batch_size: int = Field(ge=1)
    scheduler_queue_seconds: float | None = Field(
        default=None, ge=0.0,
        description="Ray task dispatch latency per queued env-runner task.")


class PredictedCandidate(BaseModel):
    """Predicted throughput and limiting stage for one candidate configuration."""

    model_config = ConfigDict(extra="forbid")

    num_env_runners: int = Field(ge=1)
    num_envs_per_env_runner: int = Field(ge=1)
    predicted_steps_per_second: float = Field(ge=0.0)
    bottleneck: BottleneckStage


def _sampling_components(costs: StageCosts) -> tuple[float, float, float]:
    """Return the three additive contributors to one env's sampling time.

    ``(env_step_seconds, inference_seconds_per_env, gil_overhead_seconds)``.
    GIL contention is modeled as extending the raw step+inference time rather
    than as an independent stage, since it is contention *for* that same CPU
    work, not separate work.
    """
    raw = costs.env_step_seconds + costs.inference_seconds_per_env
    if costs.gil_contention_ratio is not None:
        usable_fraction = max(1e-6, 1.0 - costs.gil_contention_ratio)
        gil_overhead = raw / usable_fraction - raw
    else:
        gil_overhead = 0.0
    return costs.env_step_seconds, costs.inference_seconds_per_env, gil_overhead


def _sampling_bottleneck_label(costs: StageCosts) -> BottleneckStage:
    """Attribute a sampling-bound prediction to its largest raw contributor."""
    env_step, inference, gil_overhead = _sampling_components(costs)
    contributions: dict[BottleneckStage, float] = {
        "env_step": env_step,
        "inference": inference,
        "gil": gil_overhead,
    }
    return max(contributions, key=lambda stage: contributions[stage])


def _rates(
    costs: StageCosts,
    num_env_runners: int,
    num_envs_per_env_runner: int,
) -> dict[str, float]:
    """Return each top-level stage's achievable steps/second, uncorrected."""
    total_envs = num_env_runners * num_envs_per_env_runner
    env_step, inference, gil_overhead = _sampling_components(costs)
    per_env_seconds = env_step + inference + gil_overhead

    rates: dict[str, float] = {
        "sampling": total_envs / per_env_seconds,
        "transfer": 1.0 / (costs.transfer_seconds_per_mb * costs.avg_sample_mb),
        "learner": costs.train_batch_size / (
            costs.learner_seconds_per_batch + (costs.scheduler_queue_seconds or 0.0)
        ),
    }
    if costs.scheduler_queue_seconds is not None and costs.scheduler_queue_seconds > 0.0:
        rates["scheduler"] = num_env_runners / costs.scheduler_queue_seconds
    else:
        rates["scheduler"] = math.inf
    return rates


def predict_throughput(
    costs: StageCosts,
    num_env_runners: int,
    num_envs_per_env_runner: int,
    *,
    correction: float = 0.0,
) -> PredictedCandidate:
    """Predict steps/second for an untested candidate via a roofline model.

    ``correction`` is the CPU-oversubscription penalty exponent fit by
    :func:`fit_contention_correction` (0 means no penalty). It scales down the
    rates that are proportional to ``num_env_runners`` (sampling, scheduler
    queuing) as more env-runners compete for the same host cores; transfer and
    learner rates are GPU/network-bound and independent of env-runner count in
    this model.
    """
    if num_env_runners < 1 or num_envs_per_env_runner < 1:
        raise ValueError("candidate counts must be at least 1")

    rates = _rates(costs, num_env_runners, num_envs_per_env_runner)
    if correction and num_env_runners > 1:
        penalty = num_env_runners**(-correction)
        for stage in _PARALLEL_RATE_STAGES:
            rates[stage] *= penalty

    limiting_stage = min(rates, key=lambda stage: rates[stage])
    bottleneck: BottleneckStage = (
        _sampling_bottleneck_label(costs)
        if limiting_stage == "sampling" else limiting_stage  # type: ignore[assignment]
    )
    return PredictedCandidate(
        num_env_runners=num_env_runners,
        num_envs_per_env_runner=num_envs_per_env_runner,
        predicted_steps_per_second=rates[limiting_stage],
        bottleneck=bottleneck,
    )


def fit_contention_correction(
    costs: StageCosts,
    probe_points: list[tuple[int, int, float]],
) -> float:
    """Fit an oversubscription-penalty exponent from measured multi-candidate probes.

    ``probe_points`` is a list of ``(num_env_runners, num_envs_per_env_runner,
    measured_steps_per_second)`` triples from a few short real sweep candidates.
    Returns 0.0 (no correction) if fewer than one probe has ``num_env_runners >
    1``, since the exponent cannot be estimated from a single data point.

    The fit is linear regression through the origin in log-log space:
    ``log(measured / naive) = -exponent * log(num_env_runners)``.
    """
    numerator = 0.0
    denominator = 0.0
    for num_env_runners, num_envs_per_env_runner, measured in probe_points:
        if num_env_runners <= 1 or measured <= 0.0:
            continue
        naive = predict_throughput(
            costs, num_env_runners, num_envs_per_env_runner).predicted_steps_per_second
        if naive <= 0.0:
            continue
        log_n = math.log(num_env_runners)
        log_ratio = math.log(measured / naive)
        numerator += log_n * -log_ratio
        denominator += log_n * log_n

    if denominator == 0.0:
        return 0.0
    return max(0.0, min(1.0, numerator / denominator))


SweepAxis = Literal["num_env_runners", "num_envs_per_env_runner"]


def recommend_search_range(
    costs: StageCosts,
    *,
    axis: SweepAxis,
    maximum: int,
    fixed: int = 1,
    correction: float = 0.0,
    band: int = 2,
) -> tuple[int, int]:
    """Return an inclusive ``(start, end)`` band of candidates to confirm.

    Evaluates the roofline prediction across the full ``1..maximum`` grid (cheap,
    no measurement) to find the predicted peak along ``axis``, then returns a
    small window around it for the real adaptive sweep to confirm. ``fixed`` is
    the other axis's held-constant value (e.g. ``num_envs_per_env_runner`` from
    the environment-sweep phase when narrowing the worker-sweep phase).
    """
    if maximum < 1:
        raise ValueError("maximum must be at least 1")
    if band < 0:
        raise ValueError("band must be non-negative")

    peak_candidate = 1
    peak_rate = -1.0
    for candidate in range(1, maximum + 1):
        num_env_runners, num_envs_per_env_runner = (
            (candidate, fixed) if axis == "num_env_runners" else (fixed, candidate)
        )
        predicted = predict_throughput(
            costs,
            num_env_runners,
            num_envs_per_env_runner,
            correction=correction,
        )
        if predicted.predicted_steps_per_second > peak_rate:
            peak_rate = predicted.predicted_steps_per_second
            peak_candidate = candidate

    start = max(1, peak_candidate - band)
    end = min(maximum, peak_candidate + band)
    return start, end
