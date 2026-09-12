import json

import pytest
import torch

from theseo_anysearch.garden.pilots import compact_diagnosis as diagnosis


def test_constant_vectors_have_zero_rank():
    result = diagnosis.spectrum(torch.ones(12, 8))
    assert result["effective_rank"] == 0
    assert result["leading_fraction"] == 0


def test_repeated_single_direction_has_rank_one():
    x = torch.arange(12).float()[:, None] * torch.tensor([[1., 2., 3.]])
    result = diagnosis.spectrum(x)
    assert result["effective_rank"] == pytest.approx(1., abs=1e-10)
    assert result["leading_fraction"] == pytest.approx(1.)


def test_trained_decoder_uses_logits_and_clamps_distance():
    vectors = torch.zeros(2, 4)
    bias = torch.cat((torch.zeros(2 * 4913), torch.full((4913,), 2.)))
    state = {"weight": torch.zeros(3 * 4913, 4), "bias": bias}
    prediction = diagnosis.trained_prediction(state, vectors)
    assert prediction.shape == (2, 3, 4913)
    assert (prediction[:, :2] == .5).all()
    assert (prediction[:, 2] == 1).all()


def test_started_analysis_cannot_silently_restart(tmp_path):
    diagnosis.ensure_unstarted(tmp_path)
    (tmp_path / "started.json").write_text("{}")
    with pytest.raises(ValueError, match="already started"):
        diagnosis.ensure_unstarted(tmp_path)


def test_search_tampering_rejected(tmp_path):
    (tmp_path / "report.json").write_text(json.dumps({"report_payload_sha256": "bad"}))
    with pytest.raises(ValueError, match="search report mismatch"):
        diagnosis.read_search(tmp_path)
