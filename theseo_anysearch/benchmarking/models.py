"""Data contracts for adaptive resource benchmarks."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.benchmarking.model import PredictedCandidate, StageCosts

BenchmarkPhase = Literal["environments", "workers"]


class BenchmarkSample(BaseModel):
    """One measured benchmark repetition."""

    model_config = ConfigDict(extra="forbid")

    phase: BenchmarkPhase
    candidate: int = Field(ge=1)
    repeat: int = Field(ge=1)
    num_env_runners: int = Field(ge=1)
    num_envs_per_env_runner: int = Field(ge=1)
    wall_seconds: float = Field(gt=0.0)
    sampled_steps: int = Field(ge=0)
    steps_per_second: float = Field(ge=0.0)
    cpu_percent: float | None = Field(default=None, ge=0.0)
    memory_mb: float | None = Field(default=None, ge=0.0)
    gpu_utilization_percent: float | None = Field(default=None, ge=0.0)
    gpu_memory_mb: float | None = Field(default=None, ge=0.0)
    gpu_power_watts: float | None = Field(default=None, ge=0.0)


class CandidateSummary(BaseModel):
    """Median measurements for one resource candidate."""

    model_config = ConfigDict(extra="forbid")

    phase: BenchmarkPhase
    candidate: int = Field(ge=1)
    num_env_runners: int = Field(ge=1)
    num_envs_per_env_runner: int = Field(ge=1)
    steps_per_second: float = Field(ge=0.0)
    speedup: float = Field(default=1.0, ge=0.0)
    iteration_seconds: float = Field(gt=0.0)
    cpu_percent: float | None = Field(default=None, ge=0.0)
    memory_mb: float | None = Field(default=None, ge=0.0)
    gpu_utilization_percent: float | None = Field(default=None, ge=0.0)
    gpu_memory_mb: float | None = Field(default=None, ge=0.0)
    gpu_power_watts: float | None = Field(default=None, ge=0.0)
    samples: list[BenchmarkSample] = Field(default_factory=list)


class SweepResult(BaseModel):
    """Outcome of one adaptive candidate sweep."""

    model_config = ConfigDict(extra="forbid")

    phase: BenchmarkPhase
    candidates: list[CandidateSummary]
    peak_candidate: int = Field(ge=1)
    peak_steps_per_second: float = Field(ge=0.0)
    stop_reason: str


class BenchmarkRecommendation(BaseModel):
    """Recommended rollout configuration from both benchmark phases."""

    model_config = ConfigDict(extra="forbid")

    num_env_runners: int = Field(ge=1)
    num_envs_per_env_runner: int = Field(ge=1)
    steps_per_second: float = Field(ge=0.0)
    speedup: float = Field(ge=0.0)


class PredictionSummary(BaseModel):
    """Roofline calibration used to narrow the confirming sweeps.

    Derived from two cheap real candidate probes (reusing the same
    machinery the sweeps use) rather than isolated per-stage RLlib timers,
    so treat ``environment_predicted``/``worker_predicted`` as a hint the
    confirming sweep validates, not a certainty.
    """

    model_config = ConfigDict(extra="forbid")

    stage_costs: StageCosts
    correction_exponent: float = Field(ge=0.0, le=1.0)
    environment_predicted: PredictedCandidate
    worker_predicted: PredictedCandidate
    calibration_seconds: float = Field(ge=0.0)


class ResourceBenchmarkResult(BaseModel):
    """Serializable result document for a complete resource benchmark."""

    model_config = ConfigDict(extra="forbid")

    created_at: str
    config_path: str
    machine: dict[str, str | int | float]
    decline_patience: int = Field(ge=1)
    decline_tolerance: float = Field(ge=0.0, lt=1.0)
    max_gpu_utilization: float = Field(default=95.0, gt=0.0, le=100.0)
    max_duration_minutes: float = Field(default=30.0, gt=0.0)
    elapsed_seconds: float = Field(default=0.0, ge=0.0)
    environment_sweep: SweepResult
    worker_sweep: SweepResult
    recommendation: BenchmarkRecommendation
    prediction: PredictionSummary | None = Field(
        default=None,
        description="Roofline calibration, absent if calibration was skipped or failed.")
