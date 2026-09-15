"""Integrity and decision replay of the completed, immutable R0 evidence."""
import hashlib
import json
from pathlib import Path

import pytest

from theseo_anysearch.garden.pilots.io import contract_sha256, payload_sha256, read_contract
from theseo_anysearch.garden.pilots.v2r2_audit import CounterfactualContext, PredictionFold, assess_r0
from theseo_anysearch.garden.pilots.v2r2_protocol import V2R2AuditProtocol, comparative_from_reports


ROOT = Path("experiments/perception_encoder/results/v2r2_r0")


def load(name):
    return json.loads((ROOT / name).read_text())


def test_completed_r0_hash_chain():
    report = load("r0-report.json")
    protocol = read_contract(ROOT / "audit-protocol.json", V2R2AuditProtocol)
    assert report["protocol_sha256"] == contract_sha256(protocol)
    assert report["report_payload_sha256"] == payload_sha256({key: value for key, value in report.items() if key != "report_payload_sha256"})
    for file, key in (("data-manifest.json", "data_manifest_sha256"), ("predictions.json", "predictions_sha256"), ("control-artifacts.json", "control_artifacts_sha256")):
        assert hashlib.sha256((ROOT / file).read_bytes()).hexdigest() == report[key]


def test_r0_decision_reproduces_without_retraining():
    protocol = read_contract(ROOT / "audit-protocol.json", V2R2AuditProtocol)
    predictions = load("predictions.json")
    rows = tuple(CounterfactualContext.model_validate(row) for row in predictions["contexts"])
    folds = tuple(PredictionFold.model_validate(row) for row in predictions["folds"])
    assert assess_r0(protocol.assessment, rows, folds) == load("r0-report.json")["assessment"]


def test_deferred_r0_does_not_authorize_downstream_experiments():
    report = load("r0-report.json")
    assert report["status"] == "completed"
    assert report["decision"] == "no_topology_identifiable"
    assert report["r0_decision"] == "defer" and report["next_stage"] is None
    assert not any(report[name] for name in ("p0c_started", "p0d_started", "p1_started"))
    assert report["candidate_training_updates"] == 0
    protocol = read_contract(ROOT / "audit-protocol.json", V2R2AuditProtocol)
    fake_calibration = {"dataset_id": protocol.dataset_id, "protocol_sha256": contract_sha256(protocol)}
    fake_calibration["report_payload_sha256"] = payload_sha256(fake_calibration)
    with pytest.raises(ValueError, match="R0 has not qualified"):
        comparative_from_reports(protocol, report, fake_calibration)
