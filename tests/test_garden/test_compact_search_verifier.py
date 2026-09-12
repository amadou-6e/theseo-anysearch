import json

import pytest

from scripts import verify_compact_search as verifier


def test_incomplete_search_is_not_verified_or_opened(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier.corpus, "data", lambda *a: pytest.fail("opened data before completion"))
    with pytest.raises(FileNotFoundError):
        verifier.verify(tmp_path)


def test_tampered_report_is_rejected_before_data_access(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier.corpus, "data", lambda *a: pytest.fail("opened data before hash validation"))
    (tmp_path / "report.json").write_text(json.dumps({"report_payload_sha256": "bad"}))
    with pytest.raises(ValueError, match="report hash mismatch"):
        verifier.verify(tmp_path)
