import json

import pytest

from scripts import verify_compact_nonlinear as verifier


def test_incomplete_run_cannot_open_evaluation_data(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier.experiment, "data", lambda *args: pytest.fail("premature evaluation access"))
    with pytest.raises(FileNotFoundError):
        verifier.verify(tmp_path)


def test_tampered_report_rejected_before_evaluation(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier.experiment, "data", lambda *args: pytest.fail("premature evaluation access"))
    (tmp_path / "report.json").write_text(json.dumps({"report_payload_sha256": "invalid"}))
    with pytest.raises(ValueError, match="report hash mismatch"):
        verifier.verify(tmp_path)
