"""Tests for benchmark telemetry collection."""

from __future__ import annotations

import builtins
import time

import pytest

from theseo_anysearch.benchmarking.telemetry import (
    GpuSampler,
    GpuSnapshot,
    gil_contention,
    scheduler_queue_delay,
)


def test_disabled_gpu_sampler_does_not_poll(monkeypatch) -> None:
    monkeypatch.setattr(
        "theseo_anysearch.benchmarking.telemetry.gpu_snapshot",
        lambda: GpuSnapshot(utilization_percent=99.0),
    )

    with GpuSampler(enabled=False) as sampler:
        pass

    assert sampler.samples == []


def test_gpu_sampler_collects_during_measured_region(monkeypatch) -> None:
    monkeypatch.setattr(
        "theseo_anysearch.benchmarking.telemetry.gpu_snapshot",
        lambda: GpuSnapshot(utilization_percent=75.0, memory_mb=512.0),
    )

    with GpuSampler(enabled=True, interval_seconds=0.01) as sampler:
        time.sleep(0.03)

    assert sampler.samples
    assert sampler.samples[0].utilization_percent == 75.0


def test_gpu_sampler_waits_before_first_poll(monkeypatch) -> None:
    polls = []
    monkeypatch.setattr(
        "theseo_anysearch.benchmarking.telemetry.gpu_snapshot",
        lambda: polls.append(True) or GpuSnapshot(utilization_percent=75.0),
    )

    with GpuSampler(enabled=True, interval_seconds=1.0):
        time.sleep(0.01)

    assert polls == []


def test_gil_contention_returns_a_ratio_between_zero_and_one() -> None:
    pytest.importorskip("gilknocker")

    ratio = gil_contention(duration_seconds=0.05)

    assert ratio is not None
    assert 0.0 <= ratio <= 1.0


def test_gil_contention_returns_none_without_gilknocker(monkeypatch) -> None:
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "gilknocker":
            raise ImportError("simulated: gilknocker not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)

    assert gil_contention(duration_seconds=0.01) is None


def test_scheduler_queue_delay_returns_none_without_ray_initialized() -> None:
    assert scheduler_queue_delay() is None
