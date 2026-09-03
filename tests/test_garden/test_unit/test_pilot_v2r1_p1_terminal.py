"""Tests for the terminal v2r1 replacement-P1 transition."""
from __future__ import annotations

import pytest

from experiments.perception_encoder.v2r1_p1_terminal import blocked_p1_report


SHA = "a" * 64


def _config() -> dict[str, object]:
    return {
        "run_id": "voxel-encoder-pilot-v2r1-p1-1",
        "integration_base_sha": "b" * 40,
        "spec_commit": "c" * 40,
        "p0d_report_payload_sha256": SHA,
        "v1_p0c_report_payload_sha256": "d" * 64,
    }


def _p0d(**overrides) -> dict[str, object]:
    value = {
        "run_id": "voxel-encoder-pilot-v2r1-p0d-1",
        "status": "blocked",
        "report_payload_sha256": SHA,
        "decision": {
            "decision": "blocked",
            "reason": "no_retained_bundle",
            "detail": "E1 did not freeze a valid active denominator set",
            "next": None,
        },
    }
    value.update(overrides)
    return value


def test_terminal_report_records_no_new_p1_results() -> None:
    report = blocked_p1_report(_config(), _p0d())
    assert report["status"] == "not_started"
    assert report["decision"]["reason"] == "p0d_no_retained_bundle"
    assert report["execution"] == {
        "trials_started": 0,
        "optimizer_updates": 0,
        "accelerator_hours": 0.0,
        "new_training_results": False,
    }
    assert report["v1_p1_supersede"]["status"] == "superseded"
    assert report["v1_p1_supersede"]["replacement_run_executed"] is False
    assert len(report["report_payload_sha256"]) == 64


def test_terminal_report_rejects_a_changed_p0d_payload() -> None:
    with pytest.raises(ValueError, match="payload differs"):
        blocked_p1_report(_config(), _p0d(report_payload_sha256="e" * 64))


def test_terminal_report_requires_no_retained_bundle_decision() -> None:
    changed = dict(_p0d()["decision"])
    changed["reason"] = "other"
    with pytest.raises(ValueError, match="no-retained-bundle"):
        blocked_p1_report(_config(), _p0d(decision=changed))
