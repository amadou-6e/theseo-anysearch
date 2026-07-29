"""Ray-backed adaptive resource benchmark runner."""

from __future__ import annotations

import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from theseo_anysearch.benchmarking.models import (
    BenchmarkRecommendation,
    BenchmarkSample,
    CandidateSummary,
    ResourceBenchmarkResult,
)
from theseo_anysearch.benchmarking.report import write_benchmark_artifacts
from theseo_anysearch.benchmarking.search import (
    adaptive_sweep,
    gpu_saturation_sweep,
)
from theseo_anysearch.benchmarking.telemetry import (
    GpuSampler,
    machine_metadata,
    process_snapshot,
    rollout_worker_pids,
)
from theseo_anysearch.experiments.models import ExperimentConfig


class ResourceBenchmarkRunner:
    """Find efficient environment-vectorization and rollout-worker settings."""

    def __init__(
        self,
        config: ExperimentConfig,
        config_path: Path,
        *,
        output_dir: Path,
        decline_patience: int = 3,
        decline_tolerance: float = 0.02,
        warmup_iterations: int = 1,
        measure_iterations: int = 3,
        repeats: int = 3,
        max_envs_per_worker: int = 16,
        max_workers: int = 20,
        max_gpu_utilization: float = 95.0,
        max_duration_minutes: float = 30.0,
    ) -> None:
        if config.training.algorithm.lower() != "ppo":
            raise ValueError(
                "resource benchmarking currently supports PPO only")
        for name, value in {
                "warmup_iterations": warmup_iterations,
                "measure_iterations": measure_iterations,
                "repeats": repeats,
                "max_envs_per_worker": max_envs_per_worker,
                "max_workers": max_workers,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0.0 < max_gpu_utilization <= 100.0:
            raise ValueError("max_gpu_utilization must be in (0, 100]")
        if max_duration_minutes <= 0.0:
            raise ValueError("max_duration_minutes must be greater than zero")

        self._config = config
        self._config_path = config_path.resolve()
        self._output_dir = output_dir.resolve()
        self._decline_patience = decline_patience
        self._decline_tolerance = decline_tolerance
        self._warmup_iterations = warmup_iterations
        self._measure_iterations = measure_iterations
        self._repeats = repeats
        self._max_envs_per_worker = max_envs_per_worker
        self._max_workers = _effective_worker_limit(max_workers)
        self._max_gpu_utilization = max_gpu_utilization
        self._max_duration_minutes = max_duration_minutes
        self._uses_gpu = bool(config.training.require_gpu
                              or (config.training.num_gpus is not None
                                  and config.training.num_gpus > 0))
        self._tracker: Any = None

    @property
    def output_dir(self) -> Path:
        """Directory receiving benchmark artifacts."""
        return self._output_dir

    def run(self) -> tuple[ResourceBenchmarkResult, dict[str, Path]]:
        """Execute both adaptive phases and write report artifacts."""
        from theseo_anysearch.experiments.tracking import MLflowTracker

        benchmark_started = time.perf_counter()
        deadline = benchmark_started + self._max_duration_minutes * 60.0
        stop_requested = lambda: time.perf_counter() >= deadline
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._tracker = MLflowTracker(
            self._config.mlflow,
            f"benchmark/{self._config.experiment.name}",
        )
        self._tracker.start_run(
            run_name=self._output_dir.name,
            tags={"anysearch.run_type": "resource_benchmark"},
        )
        self._tracker.log_params({
            "decline_patience":
            self._decline_patience,
            "decline_tolerance":
            self._decline_tolerance,
            "warmup_iterations":
            self._warmup_iterations,
            "measure_iterations":
            self._measure_iterations,
            "repeats":
            self._repeats,
            "max_envs_per_worker":
            self._max_envs_per_worker,
            "max_workers":
            self._max_workers,
            "max_gpu_utilization":
            self._max_gpu_utilization,
            "max_duration_minutes":
            self._max_duration_minutes,
        })

        ray_was_initialized = False
        try:
            import ray

            ray_was_initialized = ray.is_initialized()
            if not ray_was_initialized:
                from theseo_anysearch.rllib.trainer.ppo import _ensure_ray_runtime

                _ensure_ray_runtime(
                    str(self._output_dir),
                    num_env_runners=self._max_workers,
                )
            cluster_worker_limit = max(
                1,
                int(ray.cluster_resources().get("CPU", 1.0)) - 1,
            )
            self._max_workers = min(self._max_workers, cluster_worker_limit)
            environment_sweep = adaptive_sweep(
                phase="environments",
                evaluate=lambda candidate: self._evaluate_candidate(
                    phase="environments",
                    candidate=candidate,
                    num_env_runners=1,
                    num_envs_per_env_runner=candidate,
                ),
                maximum=self._max_envs_per_worker,
                decline_patience=self._decline_patience,
                decline_tolerance=self._decline_tolerance,
                stop_requested=stop_requested,
            )
            best_envs = environment_sweep.peak_candidate
            worker_sweep = gpu_saturation_sweep(
                evaluate=lambda candidate: self._evaluate_candidate(
                    phase="workers",
                    candidate=candidate,
                    num_env_runners=candidate,
                    num_envs_per_env_runner=best_envs,
                ),
                maximum=self._max_workers,
                max_gpu_utilization=self._max_gpu_utilization,
                stop_requested=stop_requested,
            )
            final_steps = worker_sweep.peak_steps_per_second
            baseline_steps = environment_sweep.candidates[0].steps_per_second
            result = ResourceBenchmarkResult(
                created_at=datetime.now(timezone.utc).isoformat(),
                config_path=str(self._config_path),
                machine=machine_metadata(),
                decline_patience=self._decline_patience,
                decline_tolerance=self._decline_tolerance,
                max_gpu_utilization=self._max_gpu_utilization,
                max_duration_minutes=self._max_duration_minutes,
                elapsed_seconds=time.perf_counter() - benchmark_started,
                environment_sweep=environment_sweep,
                worker_sweep=worker_sweep,
                recommendation=BenchmarkRecommendation(
                    num_env_runners=worker_sweep.peak_candidate,
                    num_envs_per_env_runner=best_envs,
                    steps_per_second=final_steps,
                    speedup=(final_steps /
                             baseline_steps if baseline_steps > 0 else 0.0),
                ),
            )
            artifacts = write_benchmark_artifacts(result, self._output_dir)
            for artifact in artifacts.values():
                self._tracker.log_artifact(artifact)
            self._tracker.set_tag("benchmark.stop.environments",
                                  environment_sweep.stop_reason)
            self._tracker.set_tag("benchmark.stop.workers",
                                  worker_sweep.stop_reason)
            self._tracker.end_run("FINISHED")
            return result, artifacts
        except BaseException:
            self._tracker.end_run("FAILED")
            raise
        finally:
            if not ray_was_initialized:
                try:
                    import ray

                    ray.shutdown()
                except Exception:
                    pass

    def _evaluate_candidate(
        self,
        *,
        phase: str,
        candidate: int,
        num_env_runners: int,
        num_envs_per_env_runner: int,
    ) -> CandidateSummary:
        samples = [
            self._measure_repeat(
                phase=phase,
                candidate=candidate,
                repeat=repeat,
                num_env_runners=num_env_runners,
                num_envs_per_env_runner=num_envs_per_env_runner,
            ) for repeat in range(1, self._repeats + 1)
        ]

        def median_optional(field: str) -> float | None:
            values = [
                float(value) for sample in samples
                if (value := getattr(sample, field)) is not None
            ]
            return statistics.median(values) if values else None

        summary = CandidateSummary(
            phase=phase,
            candidate=candidate,
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=num_envs_per_env_runner,
            steps_per_second=statistics.median(sample.steps_per_second
                                               for sample in samples),
            iteration_seconds=statistics.median(sample.wall_seconds /
                                                self._measure_iterations
                                                for sample in samples),
            cpu_percent=median_optional("cpu_percent"),
            memory_mb=median_optional("memory_mb"),
            gpu_utilization_percent=median_optional("gpu_utilization_percent"),
            gpu_memory_mb=median_optional("gpu_memory_mb"),
            gpu_power_watts=median_optional("gpu_power_watts"),
            samples=samples,
        )
        prefix = f"benchmark/{phase}"
        self._tracker.log_metrics(
            {
                f"{prefix}/steps_per_second":
                summary.steps_per_second,
                f"{prefix}/iteration_seconds":
                summary.iteration_seconds,
                f"{prefix}/cpu_percent":
                summary.cpu_percent or 0.0,
                f"{prefix}/gpu_utilization_percent":
                (summary.gpu_utilization_percent or 0.0),
            },
            step=candidate,
        )
        return summary

    def _measure_repeat(
        self,
        *,
        phase: str,
        candidate: int,
        repeat: int,
        num_env_runners: int,
        num_envs_per_env_runner: int,
    ) -> BenchmarkSample:
        from theseo_anysearch.rllib.trainer import Trainer

        candidate_dir = self._output_dir / "runs" / phase / str(
            candidate) / str(repeat)
        training = self._config.training.model_copy(
            update={
                "iterations": 1,
                "output_dir": candidate_dir,
                "num_env_runners": num_env_runners,
                "num_envs_per_env_runner": num_envs_per_env_runner,
                "num_gpus_per_env_runner": 0.0,
            })
        settings = self._config.to_settings().model_copy(
            update={
                "training":
                training,
                "evaluation":
                self._config.evaluation.model_copy(
                    update={"num_env_runners": 0}),
            })
        trainer = Trainer.from_settings(settings)
        algo = trainer._build_algorithm()
        try:
            last_result: dict[str, Any] = {}
            for _ in range(self._warmup_iterations):
                last_result = algo.train()
            starting_steps = _sampled_steps(last_result)
            pids = rollout_worker_pids(algo)
            process_before = process_snapshot(pids)
            started = time.perf_counter()
            with GpuSampler(enabled=self._uses_gpu) as gpu_sampler:
                for _ in range(self._measure_iterations):
                    last_result = algo.train()
            wall_seconds = time.perf_counter() - started
            gpu_samples = gpu_sampler.samples
            process_after = process_snapshot(pids)
            ending_steps = _sampled_steps(last_result)
            sampled_steps = max(0, ending_steps - starting_steps)
            if sampled_steps == 0:
                batch_size = int(
                    getattr(self._config.algorithm_config, "train_batch_size",
                            0))
                sampled_steps = batch_size * self._measure_iterations
            worker_cpu_seconds = max(
                0.0,
                process_after.cpu_seconds - process_before.cpu_seconds,
            )

            def gpu_average(field: str) -> float | None:
                values = [
                    float(value) for snapshot in gpu_samples
                    if (value := getattr(snapshot, field)) is not None
                ]
                return statistics.fmean(values) if values else None

            return BenchmarkSample(
                phase=phase,
                candidate=candidate,
                repeat=repeat,
                num_env_runners=num_env_runners,
                num_envs_per_env_runner=num_envs_per_env_runner,
                wall_seconds=wall_seconds,
                sampled_steps=sampled_steps,
                steps_per_second=sampled_steps / wall_seconds,
                cpu_percent=worker_cpu_seconds / wall_seconds * 100.0,
                memory_mb=process_after.memory_mb,
                gpu_utilization_percent=gpu_average("utilization_percent"),
                gpu_memory_mb=gpu_average("memory_mb"),
                gpu_power_watts=gpu_average("power_watts"),
            )
        finally:
            algo.stop()


def _sampled_steps(result: dict[str, Any]) -> int:
    """Extract lifetime sampled environment steps across RLlib API stacks."""
    env_runners = result.get("env_runners") or {}
    for value in (
            env_runners.get("num_env_steps_sampled_lifetime"),
            result.get("num_env_steps_sampled_lifetime"),
            result.get("timesteps_total"),
    ):
        if value is not None:
            return int(value)
    return 0


def _effective_worker_limit(requested: int) -> int:
    """Cap rollout workers to logical CPUs while reserving one driver CPU."""
    import psutil

    available = max(1, (psutil.cpu_count(logical=True) or 2) - 1)
    return min(requested, available)
