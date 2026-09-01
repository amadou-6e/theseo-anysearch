"""Tests for benchmark telemetry collection."""

from __future__ import annotations

import time

from theseo_anysearch.benchmarking.telemetry import GpuSampler, GpuSnapshot


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
