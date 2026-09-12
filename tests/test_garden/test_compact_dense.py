import torch
from theseo_anysearch.garden.pilots import compact_dense as s


def test_fresh_data_and_linear_oracle_capacity():
    rows = s.data()
    old = s.previous.data()
    assert set(rows["parents"]).isdisjoint(old["parents"])
    x = torch.cat((rows["codes"], torch.ones(4, 1, dtype=torch.float64)), 1)
    assert torch.linalg.matrix_rank(x) == 4
    # This is a mathematical positive control, not a deployable fitted encoder.
    targets = rows["targets"].double()*2-1
    weights = torch.linalg.lstsq(x, targets, driver="gelsd").solution
    assert torch.equal((x@weights)>0, rows["targets"].bool())


def test_dense_prediction_metric_replay():
    y = torch.zeros(4, 16); y[:, :4] = 1
    scores = s.scores(y*.98+.01, y)
    assert scores["iou"] == scores["f1"] == scores["auprc"] == 1
    assert len(scores["per_scene"]) == 4
