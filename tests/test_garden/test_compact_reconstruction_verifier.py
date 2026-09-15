import json

import pytest

from scripts import verify_compact_reconstruction as verifier


def test_rejects_report_tamper_before_loading_models(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"report_payload_sha256": "0" * 64, "status": "completed"}))
    with pytest.raises(ValueError, match="report hash"):
        verifier.checked_report(tmp_path)


def test_rejects_missing_candidate_even_with_rehashed_envelope(tmp_path):
    study = verifier.study
    payload = {"plan": study.PLAN, "configs": study.configs()}
    report = {"registration": {"payload": payload, "identity_sha256": study.d.base.payload_sha256(payload)},
              "records": [], "artifacts": {}}
    report["report_payload_sha256"] = study.d.base.payload_sha256(report)
    (tmp_path / "report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="candidate set"):
        verifier.checked_report(tmp_path)
