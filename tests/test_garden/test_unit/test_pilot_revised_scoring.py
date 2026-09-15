"""Tests for per-metric revised pilot reporting (F7)."""
from __future__ import annotations

from theseo_anysearch.garden.pilots.benchmark import (
    CALIBRATION_TEMPLATE_BARS,
    pareto_retained_candidates,
    revised_metric_report,
)
from theseo_anysearch.garden.pilots.contracts import RevisedScoreAnchor, TrivialityCheck


def _triviality() -> TrivialityCheck:
    return TrivialityCheck(
        null_input="coordinates_only",
        pvi_embedding=0.4,
        pvi_null=0.1,
        pvi_gain=0.3,
        mdl_embedding_bits=100,
        mdl_null_bits=200,
        min_pvi_gain=0.05,
        passes=True,
    )


def _anchor(name: str) -> RevisedScoreAnchor:
    deferred = name == "geodesic_nmae"
    error = name in {"clearance_nmae", "geodesic_nmae"}
    return RevisedScoreAnchor(
        higher_is_better=not error,
        floor=1.0 if error else 0.0,
        ceiling=0.5 if error else 1.0,
        floor_source="fixture-floor",
        ceiling_source="fixture-model-free-ceiling",
        ceiling_method="knn_residual" if error else "bayes_error_knn",
        ceiling_non_collapse_verified=True,
        triviality=_triviality(),
        status="deferred" if deferred else "active",
        deferral_reason="Stage 2 wide-context evaluation" if deferred else None,
    )


def test_revised_report_has_iqm_and_profile_without_composite() -> None:
    rows = [
        {
            "components": {
                "occupied_iou": 0.4 + 0.1 * index,
                "boundary_f1": 0.5 + 0.1 * index,
                "clearance_nmae": 0.8 - 0.1 * index,
                "reachability_auprc": 0.6 + 0.1 * index,
                "geodesic_nmae": 0.2,
            }
        }
        for index in range(3)
    ]
    report = revised_metric_report(
        rows, {name: _anchor(name) for name in rows[0]["components"]}
    )
    assert "pilot_score" not in report
    assert all(
        "normalized_iqm" in report[name]
        for name in report
        if name != "geodesic_nmae"
    )
    assert report["geodesic_nmae"]["status"] == "deferred"
    assert "performance_profile" not in report["geodesic_nmae"]
    assert CALIBRATION_TEMPLATE_BARS == {
        "boundary_f1": {"minimum_absolute_headroom": 0.10},
        "clearance_nmae": {"minimum_relative_error_reduction": 0.20},
    }


def test_pareto_retention_does_not_average_away_a_weak_metric() -> None:
    retained = pareto_retained_candidates(
        {
            "balanced": {"occupancy": 0.7, "reachability": 0.7},
            "spiky": {"occupancy": 1.0, "reachability": 0.1},
            "dominated": {"occupancy": 0.6, "reachability": 0.6},
        }
    )
    assert retained == ("balanced", "spiky")
