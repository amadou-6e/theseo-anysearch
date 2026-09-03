"""Tests for the recorded F5 geodesic deferral decision."""
from __future__ import annotations

import numpy as np

from theseo_anysearch.garden.evaluation.geodesic import decide_geodesic_pilot_metric


def test_low_variance_p0c_geodesic_metric_is_deferred_to_stage_2() -> None:
    targets = np.linspace(0.02, 0.08, 100)
    decision = decide_geodesic_pilot_metric(
        targets,
        frequency_nmae=0.022515243950901476,
        supervised_nmae=0.04816410017751241,
    )
    assert decision.disposition == "deferred"
    assert not decision.active_in_p0c
    assert decision.revisit_stage == "Stage 2 wide-context evaluation"
    assert "below the 0.15 noise floor" in decision.reason


def test_geodesic_metric_can_only_remain_active_with_real_headroom() -> None:
    targets = np.linspace(0, 1, 100)
    decision = decide_geodesic_pilot_metric(
        targets, frequency_nmae=0.30, supervised_nmae=0.20
    )
    assert decision.disposition == "redesign"
    assert decision.active_in_p0c


def test_geodesic_decision_rejects_nonfinite_targets() -> None:
    targets = np.array([0.0, np.nan])
    try:
        decide_geodesic_pilot_metric(
            targets, frequency_nmae=0.3, supervised_nmae=0.2
        )
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite targets must be rejected")
