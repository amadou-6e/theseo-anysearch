"""Cross-platform CPU, memory, and NVIDIA GPU telemetry."""

from __future__ import annotations

import csv
import io
import os
import platform
import subprocess
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class ProcessSnapshot:
    """Aggregate CPU time and resident memory for rollout worker processes."""

    cpu_seconds: float
    memory_mb: float


@dataclass(frozen=True)
class GpuSnapshot:
    """Instantaneous NVIDIA device metrics."""

    utilization_percent: float | None = None
    memory_mb: float | None = None
    power_watts: float | None = None


class GpuSampler:
    """Poll NVIDIA telemetry while a measured training region is active."""

    def __init__(self, enabled: bool, interval_seconds: float = 0.1) -> None:
        self._enabled = enabled
        self._interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[GpuSnapshot] = []

    def __enter__(self) -> "GpuSampler":
        if self._enabled:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self._interval_seconds + 5.0)

    def _poll(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            snapshot = gpu_snapshot()
            if snapshot.utilization_percent is not None:
                self.samples.append(snapshot)


def machine_metadata() -> dict[str, str | int | float]:
    """Return stable host metadata for benchmark reproducibility."""
    import psutil

    memory = psutil.virtual_memory()
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "logical_cpus": psutil.cpu_count(logical=True) or 0,
        "physical_cpus": psutil.cpu_count(logical=False) or 0,
        "memory_gb": round(memory.total / (1024**3), 2),
    }


def rollout_worker_pids(algo: object) -> list[int]:
    """Return process IDs for healthy remote RLlib rollout workers."""
    group = getattr(algo, "env_runner_group")
    return [
        int(pid) for pid in group.foreach_env_runner(
            lambda _: os.getpid(),
            local_env_runner=False,
        )
    ]


def process_snapshot(pids: list[int]) -> ProcessSnapshot:
    """Read aggregate process CPU time and resident memory."""
    import psutil

    cpu_seconds = 0.0
    memory_bytes = 0
    for pid in pids:
        try:
            process = psutil.Process(pid)
            times = process.cpu_times()
            cpu_seconds += float(times.user + times.system)
            memory_bytes += int(process.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return ProcessSnapshot(
        cpu_seconds=cpu_seconds,
        memory_mb=memory_bytes / (1024**2),
    )


def gpu_snapshot() -> GpuSnapshot:
    """Read average NVIDIA metrics, returning empty telemetry when unavailable."""
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return GpuSnapshot()

    rows = list(csv.reader(io.StringIO(completed.stdout)))
    parsed: list[tuple[float, float, float]] = []
    for row in rows:
        if len(row) < 3:
            continue
        try:
            parsed.append(tuple(float(value.strip()) for value in row[:3]))
        except ValueError:
            continue
    if not parsed:
        return GpuSnapshot()

    count = float(len(parsed))
    return GpuSnapshot(
        utilization_percent=sum(row[0] for row in parsed) / count,
        memory_mb=sum(row[1] for row in parsed),
        power_watts=sum(row[2] for row in parsed),
    )


def gil_contention(duration_seconds: float = 0.5) -> float | None:
    """Measure the fraction of ``duration_seconds`` this process held the GIL.

    Uses ``gilknocker`` (a canary thread that reports how much of the wall
    clock it was starved of the GIL) when available. Returns ``None`` on
    platforms or interpreters where GIL contention cannot be measured (e.g.
    PyPy, or ``gilknocker`` not installed), so callers must treat the signal
    as optional.
    """
    try:
        from gilknocker import KnockKnock
    except ImportError:
        return None

    knocker = KnockKnock(polling_interval_micros=100)
    knocker.start()
    try:
        time.sleep(duration_seconds)
    finally:
        knocker.stop()
    return float(knocker.contention_metric)


def scheduler_queue_delay(samples: int = 5) -> float | None:
    """Measure Ray task-dispatch latency as a proxy for CPU oversubscription.

    Submits trivial no-op remote tasks and measures the wall time between
    submission and completion, minus the task's own (near-zero) execution
    time. A rising delay across candidates indicates the Ray scheduler is
    queuing work because more tasks are competing for CPU than are
    available. Returns ``None`` if Ray is not initialized.
    """
    try:
        import ray
    except ImportError:
        return None
    if not ray.is_initialized():
        return None

    @ray.remote(num_cpus=0)
    def _noop() -> None:
        return None

    delays: list[float] = []
    for _ in range(max(1, samples)):
        started = time.perf_counter()
        ray.get(_noop.remote())
        delays.append(time.perf_counter() - started)
    return sum(delays) / len(delays)
