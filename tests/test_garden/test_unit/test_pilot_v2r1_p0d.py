"""Tests for the v2r1 P0D prerequisite transition."""
from __future__ import annotations

import pytest

from experiments.perception_encoder.v2r1_p0d import blocked_p0d_report


SHA = "a" * 64


def _config() -> dict[str, object]:
    return {
        "run_id": "voxel-encoder-pilot-v2r1-p0d-1",
        "integration_base_sha": "b" * 40,
        "spec_commit": "c" * 40,
        "p0c_report_payload_sha256": SHA,
    }


def _p0c(**overrides) -> dict[str, object]:
    value = {
        "run_id": "voxel-encoder-pilot-v2r1-p0c-1",
        "status": "blocked",
        "report_payload_sha256": SHA,
        "denominator_failures": {"reachability_auprc": "insufficient headroom"},
    }
    value.update(overrides)
    return value


def test_blocked_p0d_does_not_open_observations_or_start_p1() -> None:
    report = blocked_p0d_report(_config(), _p0c())
    assert report["status"] == "blocked"
    assert report["decision"]["reason"] == "no_retained_bundle"
    assert report["observations_opened"] is False
    assert report["trials_started"] == 0
    assert report["replacement_p1"]["status"] == "not_started"
    assert len(report["report_payload_sha256"]) == 64


def test_blocked_p0d_rejects_a_changed_p0c_payload() -> None:
    with pytest.raises(ValueError, match="payload differs"):
        blocked_p0d_report(_config(), _p0c(report_payload_sha256="d" * 64))


def test_blocked_transition_rejects_a_passed_prerequisite() -> None:
    with pytest.raises(ValueError, match="only records"):
        blocked_p0d_report(_config(), _p0c(status="passed"))
