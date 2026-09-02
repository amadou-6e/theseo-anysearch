"""Contract tests for the P3 profiler and decision rule."""
from __future__ import annotations

from experiments.perception_encoder.p3_profile import (
    CANDIDATES,
    RADII,
    ProfileCell,
    _pyramid_strides,
    decide,
)


def _cell(candidate: str, radius: int) -> ProfileCell:
    return ProfileCell(
        candidate=candidate,
        radius=radius,
        status="completed",
        parameters=1,
        checkpoint_bytes=1,
        flops=1,
        latency_p50_ms=1,
        latency_p95_ms=1,
        peak_inference_allocated_bytes=1,
        peak_inference_reserved_bytes=1,
        peak_training_allocated_bytes=1,
        peak_training_reserved_bytes=1,
        training_examples_per_second=1,
        output_shapes={"global": [1, 192]},
        output_contract="legacy_global_v1" if candidate == "current_dense" else "global_scale_local_v1",
    )


def test_pyramid_levels_cover_each_profiled_radius() -> None:
    assert _pyramid_strides(8) == (1,)
    assert _pyramid_strides(16) == (1, 2)
    assert _pyramid_strides(32) == (1, 2, 4)


def test_decision_retains_complete_dense_profiles_and_reports_sparse_unavailable() -> None:
    cells = [_cell(candidate, radius) for candidate in CANDIDATES for radius in RADII]
    decision = decide(cells)
    assert decision["retained"] == list(CANDIDATES)
    assert "sparse_residual" in decision["rejected"]
    assert decision["quality_claim"] is False
