"""Ray-backed adaptive resource benchmark runner."""

from __future__ import annotations

import logging
import os
import statistics
import sys
import threading
import time
from collections.abc import Callable
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import TextIOBase
from pathlib import Path
from typing import Any

from theseo_anysearch.benchmarking.models import (
    BenchmarkRecommendation,
    BenchmarkSample,
    CandidateSummary,
    PredictionSummary,
    ResourceBenchmarkResult,
)
from theseo_anysearch.benchmarking.report import (
    write_benchmark_artifacts,
    write_progress_report,
)
from theseo_anysearch.benchmarking.model import recommend_search_range
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

ProgressCallback = Callable[
    [str, str, int, CandidateSummary | None],
    None,
]


class _TeeTextIO(TextIOBase):
    """Write diagnostics to both the foreground stream and an artifact file."""

    def __init__(self, foreground: Any, artifact: Any) -> None:
        self._foreground = foreground
        self._artifact = artifact

    def write(self, text: str) -> int:
        self._foreground.write(text)
        self._artifact.write(text)
        return len(text)

    def flush(self) -> None:
        self._foreground.flush()
        self._artifact.flush()

    def isatty(self) -> bool:
        return bool(getattr(self._foreground, "isatty", lambda: False)())


@contextmanager
def _capture_stderr_fd(artifact: Any, foreground: Any, *, tee: bool):
    """Capture Python, logging-handler, and native writes to stderr."""
    try:
        stderr_fd = foreground.fileno()
        artifact_fd = artifact.fileno()
    except (AttributeError, OSError):
        from contextlib import redirect_stderr

        stream = _TeeTextIO(foreground, artifact) if tee else artifact
        with redirect_stderr(stream):
            yield
        return

    foreground.flush()
    artifact.flush()
    saved_fd = os.dup(stderr_fd)
    read_fd: int | None = None
    reader: threading.Thread | None = None
    try:
        if tee:
            read_fd, write_fd = os.pipe()

            def copy_stderr() -> None:
                try:
                    while chunk := os.read(read_fd, 8192):
                        os.write(saved_fd, chunk)
                        os.write(artifact_fd, chunk)
                except OSError:
                    return

            reader = threading.Thread(
                target=copy_stderr,
                name="anysearch-stderr-capture",
                daemon=True,
            )
            reader.start()
            os.dup2(write_fd, stderr_fd)
            os.close(write_fd)
        else:
            os.dup2(artifact_fd, stderr_fd)
        yield
    finally:
        foreground.flush()
        os.dup2(saved_fd, stderr_fd)
        if reader is not None:
            reader.join(timeout=5.0)
        if read_fd is not None:
            os.close(read_fd)
        if reader is not None and reader.is_alive():
            reader.join(timeout=1.0)
        os.close(saved_fd)
        artifact.flush()


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
        debug: bool = False,
        enable_prediction: bool = True,
        progress_callback: ProgressCallback | None = None,
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
        self._debug = debug
        self._enable_prediction = enable_prediction
        self._progress_callback = progress_callback
        self._environment_candidates: list[CandidateSummary] = []
        self._worker_candidates: list[CandidateSummary] = []
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
        stdout_log = self._output_dir / "benchmark.stdout.log"
        stderr_log = self._output_dir / "benchmark.stderr.log"
        output_stack = ExitStack()
        try:
            stdout_handle = output_stack.enter_context(
                stdout_log.open("w", encoding="utf-8", buffering=1))
            stderr_handle = output_stack.enter_context(
                stderr_log.open("w", encoding="utf-8", buffering=1))
            native_stderr = sys.stderr
            try:
                self._foreground_stderr = output_stack.enter_context(
                    os.fdopen(
                        os.dup(native_stderr.fileno()),
                        "w",
                        encoding=getattr(sys.stderr, "encoding", None) or "utf-8",
                        errors="replace",
                        buffering=1,
                    ))
            except (AttributeError, OSError):
                self._foreground_stderr = sys.stderr
            stdout_stream = (_TeeTextIO(sys.stdout, stdout_handle)
                             if self._debug else stdout_handle)
            stderr_stream = (_TeeTextIO(self._foreground_stderr, stderr_handle)
                             if self._debug else stderr_handle)
            output_stack.enter_context(redirect_stdout(stdout_stream))
            output_stack.enter_context(redirect_stderr(stderr_stream))
            output_stack.enter_context(
                _capture_stderr_fd(
                    stderr_handle,
                    native_stderr,
                    tee=self._debug,
                ))
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
            previous_quiet = os.environ.get("ANYSEARCH_QUIET")
            previous_quiet_log = os.environ.get("ANYSEARCH_QUIET_LOG")
            ray_logger = logging.getLogger("ray")
            previous_ray_level = ray_logger.level
            if not self._debug:
                os.environ["ANYSEARCH_QUIET"] = "1"
                os.environ["ANYSEARCH_QUIET_LOG"] = str(stdout_log)
                ray_logger.setLevel(logging.INFO)
            try:
                import ray

                ray_was_initialized = ray.is_initialized()
                if not ray_was_initialized:
                    from theseo_anysearch.rllib.algorithms.ppo import _ensure_ray_runtime

                    _ensure_ray_runtime(
                        str(self._output_dir),
                        num_env_runners=self._max_workers,
                    )
                cluster_worker_limit = max(
                    1,
                    int(ray.cluster_resources().get("CPU", 1.0)) - 1,
                )
                self._max_workers = min(self._max_workers, cluster_worker_limit)

                prediction = self._calibrate() if self._enable_prediction else None
                env_start = 1
                if prediction is not None:
                    env_start, _ = recommend_search_range(
                        prediction.stage_costs,
                        axis="num_envs_per_env_runner",
                        maximum=self._max_envs_per_worker,
                    )
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
                    start=env_start,
                    stop_requested=stop_requested,
                    on_candidate_completed=self._candidate_completed,
                )
                best_envs = environment_sweep.peak_candidate
                worker_start = 1
                if prediction is not None:
                    worker_start, _ = recommend_search_range(
                        prediction.stage_costs,
                        axis="num_env_runners",
                        maximum=self._max_workers,
                        fixed=best_envs,
                        correction=prediction.correction_exponent,
                    )
                worker_sweep = gpu_saturation_sweep(
                    evaluate=lambda candidate: self._evaluate_candidate(
                        phase="workers",
                        candidate=candidate,
                        num_env_runners=candidate,
                        num_envs_per_env_runner=best_envs,
                    ),
                    maximum=self._max_workers,
                    max_gpu_utilization=self._max_gpu_utilization,
                    start=worker_start,
                    stop_requested=stop_requested,
                    on_candidate_completed=self._candidate_completed,
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
                    prediction=prediction,
                )
                artifacts = write_benchmark_artifacts(result, self._output_dir)
                stdout_handle.flush()
                stderr_handle.flush()
                artifacts["stdout_log"] = stdout_log
                artifacts["stderr_log"] = stderr_log
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
                ray_logger.setLevel(previous_ray_level)
                if previous_quiet is None:
                    os.environ.pop("ANYSEARCH_QUIET", None)
                else:
                    os.environ["ANYSEARCH_QUIET"] = previous_quiet
                if previous_quiet_log is None:
                    os.environ.pop("ANYSEARCH_QUIET_LOG", None)
                else:
                    os.environ["ANYSEARCH_QUIET_LOG"] = previous_quiet_log
                if not ray_was_initialized:
                    try:
                        import ray

                        ray.shutdown()
                    except Exception:
                        pass
        finally:
            output_stack.close()

    def _measure_transfer_seconds_per_mb(self) -> float | None:
        """Time a Ray object-store round trip for a 1 MB payload."""
        try:
            import numpy as np
            import ray
        except ImportError:
            return None
        if not ray.is_initialized():
            return None

        payload = np.zeros((256 * 1024,), dtype=np.float32)
        payload_mb = payload.nbytes / (1024 * 1024)
        started = time.perf_counter()
        ray.get(ray.put(payload))
        elapsed = time.perf_counter() - started
        return elapsed / payload_mb if payload_mb > 0 else None

    def _calibrate(self) -> PredictionSummary | None:
        """Roofline calibration from two cheap real candidate probes.

        Reuses the same candidate-evaluation machinery the sweeps use, so this
        costs no more than the first couple of sweep steps a full scan would
        run anyway. Per-stage costs are read directly from RLlib's own
        env_step_timer / rlmodule_inference_timer / learner_update_timer EMAs
        (see theseo_anysearch.rllib.trainer.results.IterationTimings) when
        the running RLlib version reports them; where a timer is absent (an
        older API stack, or a metric key RLlib renamed), the corresponding
        cost falls back to a coarser estimate derived from aggregate
        steps_per_second, and the bottleneck label should be treated as a
        weaker hint in that case. Returns None on any failure (missing
        gilknocker, a probe erroring, etc.), so the caller falls back to an
        unrestricted sweep -- calibration only ever makes this faster, never
        less correct.
        """
        from theseo_anysearch.benchmarking.model import (
            StageCosts,
            fit_contention_correction,
            predict_throughput,
        )
        from theseo_anysearch.benchmarking.telemetry import (
            gil_contention,
            scheduler_queue_delay,
        )

        started = time.perf_counter()
        try:
            probe_single = self._evaluate_candidate(
                phase="environments",
                candidate=1,
                num_env_runners=1,
                num_envs_per_env_runner=1,
            )
            probe_double = self._evaluate_candidate(
                phase="environments",
                candidate=2,
                num_env_runners=1,
                num_envs_per_env_runner=2,
            )
            probe_workers = self._evaluate_candidate(
                phase="workers",
                candidate=2,
                num_env_runners=2,
                num_envs_per_env_runner=2,
            )

            transfer_seconds_per_mb = self._measure_transfer_seconds_per_mb()
            train_batch_size = int(
                getattr(self._config.algorithm_config, "train_batch_size", 0)) or 4000

            env_step_seconds = (
                _median_sample_field(probe_double, "env_step_seconds")
                or 2.0 / max(probe_double.steps_per_second, 1e-9))
            inference_seconds_per_env = (
                _median_sample_field(probe_double, "inference_seconds") or 1e-9)
            learner_seconds_per_batch = (
                _median_sample_field(probe_single, "learner_update_seconds")
                or train_batch_size / max(probe_single.steps_per_second, 1e-9))

            costs = StageCosts(
                env_step_seconds=env_step_seconds,
                inference_seconds_per_env=inference_seconds_per_env,
                gil_contention_ratio=gil_contention(0.2),
                transfer_seconds_per_mb=transfer_seconds_per_mb or 1e-4,
                avg_sample_mb=max(1e-6, train_batch_size * 4 / (1024 * 1024)),
                learner_seconds_per_batch=learner_seconds_per_batch,
                train_batch_size=train_batch_size,
                scheduler_queue_seconds=scheduler_queue_delay(),
            )
            correction = fit_contention_correction(
                costs,
                [
                    (1, 1, probe_single.steps_per_second),
                    (2, 2, probe_workers.steps_per_second),
                ],
            )
            return PredictionSummary(
                stage_costs=costs,
                correction_exponent=correction,
                environment_predicted=predict_throughput(
                    costs, 1, self._max_envs_per_worker),
                worker_predicted=predict_throughput(
                    costs, self._max_workers, 2, correction=correction),
                calibration_seconds=time.perf_counter() - started,
            )
        except Exception:
            return None

    def _evaluate_candidate(
        self,
        *,
        phase: str,
        candidate: int,
        num_env_runners: int,
        num_envs_per_env_runner: int,
    ) -> CandidateSummary:
        self._notify_progress("started", phase, candidate, None)
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

    def _candidate_completed(self, summary: CandidateSummary) -> None:
        candidates = (self._environment_candidates if summary.phase
                      == "environments" else self._worker_candidates)
        candidates.append(summary)
        write_progress_report(
            environment_candidates=self._environment_candidates,
            worker_candidates=self._worker_candidates,
            output_dir=self._output_dir,
            max_envs_per_worker=self._max_envs_per_worker,
            max_workers=self._max_workers,
        )
        self._notify_progress(
            "completed",
            summary.phase,
            summary.candidate,
            summary,
        )

    def _notify_progress(
        self,
        event: str,
        phase: str,
        candidate: int,
        summary: CandidateSummary | None,
    ) -> None:
        callback = getattr(self, "_progress_callback", None)
        if callback is not None:
            callback(event, phase, candidate, summary)

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
        print(
            f"[{phase} candidate {candidate}, repeat {repeat}] "
            f"workers={num_env_runners} environments={num_envs_per_env_runner}",
            flush=True,
        )
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
        algo: Any | None = None
        try:
            trainer = Trainer.from_settings(settings)
            algo = trainer._build_algorithm()
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
            timings = _iteration_timings(last_result)
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
                env_step_seconds=timings.env_step_ema_s,
                inference_seconds=timings.inference_ema_s,
                learner_update_seconds=timings.learner_update_ema_s,
                sync_weights_seconds=timings.sync_weights_ema_s,
            )
        except BaseException:
            if not self._debug:
                log_path = candidate_dir / "debug_stage.log"
                print(
                    f"Benchmark candidate failed; diagnostics: {log_path}",
                    file=getattr(self, "_foreground_stderr", sys.stderr),
                )
                if log_path.is_file():
                    print(
                        log_path.read_text(encoding="utf-8"),
                        file=getattr(self, "_foreground_stderr", sys.stderr),
                    )
            raise
        finally:
            if algo is not None:
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


def _median_sample_field(candidate: CandidateSummary, field: str) -> float | None:
    """Median of a per-sample timing field across a candidate's repeats."""
    values = [
        float(value) for sample in candidate.samples
        if (value := getattr(sample, field, None)) is not None
    ]
    return statistics.median(values) if values else None


def _iteration_timings(result: dict[str, Any]) -> "IterationTimings":
    """Parse RLlib's own per-stage timers from a train() result, if present."""
    from theseo_anysearch.rllib.trainer.results import IterationTimings

    return IterationTimings.from_rllib(result)


def _effective_worker_limit(requested: int) -> int:
    """Cap rollout workers to logical CPUs while reserving one driver CPU."""
    import psutil

    available = max(1, (psutil.cpu_count(logical=True) or 2) - 1)
    return min(requested, available)
