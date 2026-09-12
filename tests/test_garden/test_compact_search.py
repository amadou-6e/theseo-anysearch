import json
import pytest
import torch
from theseo_anysearch.garden.pilots import compact_search as s
from theseo_anysearch.garden.pilots import compact_search_data as c


def test_structure_and_hyperparameter_templates_are_frozen():
    configs = s.structural_configs()
    assert len(configs) == 18 and len({(x["mode"], x["dimension"], x["joint"]) for x in configs}) == 18
    assert s.templates() == s.templates() and len(s.templates()) == 18
    for t in s.templates():
        assert 1e-4 <= t["lr"] <= .003
        assert 1e-5 <= t["weight_decay"] <= .01
    assert s.PLAN["protected"] == [5, 16]


def test_budget_reservation_idempotence_and_cap(tmp_path):
    path = tmp_path/"budget.json"
    first = s.reserve_budget(path, "a", 8*3600)
    assert s.reserve_budget(path, "a", 8*3600) == first
    with pytest.raises(ValueError): s.reserve_budget(path, "a", 7*3600)
    with pytest.raises(ValueError): s.reserve_budget(path, "b", 40*3600)
    assert json.loads(path.read_text()) == first


def test_ridge_recovers_linear_fixture_with_train_normalization():
    torch.manual_seed(7)
    x = torch.randn(64, 4, dtype=torch.float64)
    weights = torch.randn(4, 6, dtype=torch.float64)*.01
    y = (x@weights+.5).reshape(64, 2, 3)
    readout = c.ridge_fit(x, y, alpha=1e-6)
    assert torch.allclose(readout["mean"], x.mean(0))
    assert torch.allclose(c.ridge_predict(readout, x), y, atol=1e-7)
    with pytest.raises(ValueError): c.ridge_fit(x, y, alpha=0)


def test_ranking_cannot_hide_failed_component():
    good = {"config": {"id": 1, "dimension": 128}, "selection": {t: {"score": b} for t, b in c.BARS.items()}}
    bad = {"config": {"id": 0, "dimension": 64}, "selection": {t: {"score": 1 if i < 2 else .0001} for i, t in enumerate(c.BARS)}}
    bad["selection"]["boundary_f1"]["score"] = .1
    assert c.rank(good) > c.rank(bad)


def test_loss_mask_and_gradients():
    logits = torch.randn(2, 3, 16, requires_grad=True)
    target = torch.zeros_like(logits); target[:, :2, :4] = 1
    hidden = torch.zeros(2, 16, dtype=torch.bool); hidden[:, :8] = 1
    loss = s.objective(logits, target, hidden, torch.ones(2), {"boundary_weight": 1, "distance_weight": 10})
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert not logits.grad[:, :2, 8:].any()
    with pytest.raises(ValueError): s.objective(logits, target, torch.zeros_like(hidden), torch.ones(2), {"boundary_weight": 1, "distance_weight": 10})


def test_corpus_family_support_and_identity(monkeypatch):
    monkeypatch.setitem(c.COUNTS, "selection", 24)
    rows = c.data(["selection"])
    s.check_support(rows)
    assert rows["selection"]["targets"].shape == (24, 3, 4913)
    assert len(set(rows["selection"]["parents"])) == 24
    original = c.identity(rows)
    rows["selection"]["targets"][0, 0, 0] += 1
    assert original != c.identity(rows)
