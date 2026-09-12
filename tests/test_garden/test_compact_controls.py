import time
import numpy as np
import pytest
import torch
from theseo_anysearch.garden.pilots import compact_controls as s


def test_raw_features_isolate_hidden_truth():
    occ = torch.rand(2, 33, 33, 33) > .7
    hidden = torch.rand(2, 33, 33, 33) > .8
    idx = torch.tensor([[0, 2456, 4912], [1, 289, 4000]])
    a = s.raw_features(occ, hidden, idx)
    b = s.raw_features(torch.where(hidden, ~occ, occ), hidden, idx)
    assert a.shape == (2, 3, 57) and torch.equal(a, b)
    xyz = torch.stack((idx//289, idx//17 % 17, idx % 17), -1)+8
    for batch in range(2):
        for q in range(3):
            x, y, z = xyz[batch, q]
            assert a[batch, q, 13] == (occ[batch, x, y, z] & ~hidden[batch, x, y, z]).float()
            assert a[batch, q, 40] == hidden[batch, x, y, z].float()


def test_metrics_separate_ranking_and_threshold():
    target = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    pred = np.array([[.1, .2, .3, .4], [.1, .2, .3, .4]])
    scores = s.metrics(pred, target, .25, "occupied_iou")
    assert scores["fixed05_score"] == 0
    assert scores["selected_score"] == scores["auprc"] == scores["auroc"] == 1


def test_constant_ranking_uses_prevalence():
    y = np.array([[0, 0, 0, 1]])
    scores = s.metrics(np.full_like(y, .2, dtype=float), y, .1, "boundary_f1")
    assert scores["auprc"] == .25 and scores["auroc"] == .5


def test_balancing_is_train_only_and_finite():
    torch.set_num_threads(2)
    x = torch.randn(2, 16, 3)
    y = torch.zeros(2, 16); y[:, :4] = 1
    fitted, _, weight = s.fit(x, y, 2, True, time.monotonic()+30)
    assert weight == 3
    assert torch.allclose(fitted[1], x.flatten(0, 1).mean(0))
    assert torch.isfinite(s.d.base.predict(fitted, x, "occupied_iou")).all()
    with pytest.raises(ValueError):
        s.fit(x, torch.zeros_like(y), 1, True, time.monotonic()+30)


def test_new_data_identity_and_source_disjointness():
    import json
    from pathlib import Path
    rows = s.data()
    hashes = {h for v in rows.values() for h in v["hashes"]}
    assert len(hashes) == 120
    root = Path(__file__).resolve().parents[2]/"docs/perception-encoder-local-geometry"
    smoke = json.loads((root/"compact-smoke-v2-preregistration.json").read_text())["payload"]["data"]
    assert not hashes.intersection(h for v in smoke.values() for h in v["parents"])
    for name in ("context65-preregistration.json", "tiles-preregistration.json", "scale-preregistration.json"):
        identity = json.loads((root/name).read_text())["payload"]["identity"]
        assert not hashes.intersection(h for v in identity.values() for h in v["hashes"])


def test_completed_report_integrity():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]/"docs/perception-encoder-local-geometry"
    report = json.loads((root/"compact-controls-report.json").read_text())
    sha = report.pop("report_payload_sha256")
    assert s.d.base.payload_sha256(report) == sha
    env = json.loads((root/"compact-controls-preregistration.json").read_text())
    assert env == report["registration"]
    assert s.d.base.payload_sha256(env["payload"]) == env["identity_sha256"]
    assert env["payload"]["plan"] == s.PLAN
    assert len(report["records"]) == len(report["artifacts"]) == 60
    assert not report["promotion_eligible"]
    assert {r["recipe"] for r in report["records"]} == set(s.PLAN["recipes"])
