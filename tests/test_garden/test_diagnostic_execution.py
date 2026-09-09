import time

import numpy as np
import pytest
import torch

from theseo_anysearch.garden.pilots import diagnostic_execution as d


def test_independent_fixtures():
    d.fixture_checks()


def test_masked_truth_is_not_visible():
    x = torch.zeros(2, 17, 17, 17)
    hidden = torch.zeros(2, 1, 17, 17, 17, dtype=torch.bool)
    hidden[:, :, 3:8] = True
    other = x.clone()
    other[:, 3:8] = 1
    assert torch.equal(d.visible({"occupancy": x, "hidden": hidden}),
                       d.visible({"occupancy": other, "hidden": hidden}))


def test_threshold_and_metrics():
    y = np.array([[0, 1, 0, 1]])
    p = np.array([[.1, .4, .1, .4]])
    t = d.threshold(p, y)
    assert d.f1(p, y, t) == 1
    assert d.f1(p, y, .5) == 0
    assert d.metrics(p, y, t)["auprc"] == 1
    assert np.isfinite(d.logloss_rows(np.array([[0., 1.]]), np.array([[0, 1]]))).all()


@pytest.mark.parametrize("scores,gain,expected", [
    ([.8]*3, .2, "pass"), ([.6]*3, .2, "fail"),
    ([.8,.6,.8], .2, "inconclusive"), ([.8]*3, -.1, "fail"),
])
def test_three_valued_evidence(scores, gain, expected):
    rows = [{"seed": s, "assessment": {"f1_calibrated": score,
             "geometry_log_loss": [1-gain]*48}} for s, score in enumerate(scores)]
    assert d.classify(rows, np.ones(48))["outcome"] == expected


def test_incomplete_seeds_and_deadline():
    assert d.classify([], np.ones(48))["outcome"] == "inconclusive"
    with pytest.raises(TimeoutError):
        d.check_deadline(time.monotonic()-1)


def test_reference_dimensions():
    assert d.Reference()(torch.zeros(1, 3, 17, 17, 17)).shape == (1, 1, 17, 17, 17)
