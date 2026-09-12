import json

import pytest

from scripts import verify_compact_structured as verifier


def test_report_tamper_rejected(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"report_payload_sha256": "0" * 64, "status": "completed"}))
    with pytest.raises(ValueError, match="report hash"):
        verifier.checked_report(tmp_path)


def test_missing_candidates_rejected(tmp_path):
    study = verifier.study
    payload = {"plan": study.PLAN, "configs": study.configs()}
    report = {"registration": {"payload": payload, "identity_sha256": study.d.base.payload_sha256(payload)},
              "records": [], "artifacts": {}}
    report["report_payload_sha256"] = study.d.base.payload_sha256(report)
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="candidate set"):
        verifier.checked_report(tmp_path)
