import numpy as np
import pytest
import torch
from torch.nn import functional as F

from theseo_anysearch.garden.collision_transfer import CollisionHead, decision_threshold, probability_metrics
from theseo_anysearch.garden.pilots import compact_validation_data as data
from theseo_anysearch.garden.pilots import compact_validation as common
from theseo_anysearch.garden.pilots import compact_collision_transfer as transfer


@pytest.mark.parametrize("family", data.FAMILIES)
def test_generators_repeat_and_density(family):
    a = data.scene("fixture-geometry", family, .16)
    b = data.scene("fixture-geometry", family, .16)
    assert np.array_equal(a, b) and a.shape == (49, 49, 49)
    assert abs(data.crop(a, 17).mean() - .16) < .07  # box distance ties are retained


def test_observation_noise_preserves_hidden_truth_and_nested_masks():
    occupancy = np.zeros((33, 33, 33), dtype=bool)
    noisy, mask = data.observations("fixture", occupancy, .2, 1.)
    assert not noisy[mask].any() and noisy[~mask].all()
    _, more = data.observations("fixture", occupancy, .6)
    assert (more | ~mask).all()
    _, slab = data.observations("fixture", occupancy, slab=True)
    assert slab.sum() == 7 * 33 * 33


def test_paths_connected_padded_and_do_not_use_hidden_labels():
    occupancy = np.zeros((17, 17, 17), dtype=bool); hidden = np.zeros_like(occupancy)
    hidden[5:12] = True
    a, valid = data.paths("fixture", occupancy, hidden)
    occupancy[hidden] = True
    b, vb = data.paths("fixture", occupancy, hidden)
    assert np.array_equal(a, b) and np.array_equal(valid, vb)
    for path, mask in zip(a, valid):
        actual = path[mask]
        assert not hidden[tuple(actual[0])]
        assert np.all(np.abs(np.diff(actual, axis=0)).sum(1) == 1)
        assert len(np.unique(actual, axis=0)) == len(actual)
        assert np.all(path[~mask] == actual[-1])
    assert set(valid.sum(1)) == {2, 3, 5, 9}


def test_paths_raise_when_no_visible_free_start():
    with pytest.raises(ValueError, match="free start"):
        data.paths("none", np.zeros((17, 17, 17), bool), np.ones((17, 17, 17), bool))


@pytest.mark.parametrize("channels,side", [(1, 9), (2, 33), (8, 33)])
def test_head_shapes_gradients(channels, side):
    head = CollisionHead(channels)
    grids = torch.randn(2, channels, side, side, side)
    paths = torch.zeros(4, 9, 3, dtype=torch.long); paths[..., 0] = torch.arange(9)
    valid = torch.ones(4, 9, dtype=torch.bool)
    logits = head(grids, paths, valid, torch.tensor([0, 0, 1, 1]))
    assert logits.shape == (4,)
    logits.square().mean().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in head.parameters())


def test_deterministic_interpolation_matches_aligned_trilinear():
    head = CollisionHead(1); x = torch.randn(2, 8, 9, 9, 9)
    y = torch.einsum("oi,bcijk->bcojk", head.interpolation, x)
    y = torch.einsum("oj,bcijk->bciok", head.interpolation, y)
    y = torch.einsum("ok,bcijk->bcijo", head.interpolation, y)
    torch.testing.assert_close(y, F.interpolate(x, size=(17, 17, 17), mode="trilinear", align_corners=True), rtol=1e-5, atol=1e-6)


def test_padding_does_not_change_head_predictions():
    head = CollisionHead(1).eval(); grids = torch.randn(1, 1, 9, 9, 9)
    paths = torch.zeros(1, 9, 3, dtype=torch.long); valid = torch.arange(9)[None] < 2
    a = head(grids, paths, valid, torch.tensor([0]))
    paths[:, 2:] = 16
    torch.testing.assert_close(a, head(grids, paths, valid, torch.tensor([0])))


def test_threshold_has_required_recall_with_ties():
    p = np.linspace(0, 1, 100); y = np.ones(100, dtype=bool)
    threshold = decision_threshold(p, y)
    assert (p >= threshold).mean() >= .95
    assert (p > threshold).mean() < .95
    assert decision_threshold(np.ones(10) * .5, np.ones(10)) == .5
    with pytest.raises(ValueError): decision_threshold(p, np.zeros(100))


def test_probability_support_and_orientation():
    m = probability_metrics([.1, .9], [False, True], .5)
    assert m["auprc"] == 1 and m["auroc"] == 1 and m["false_safe"] == 0 and m["false_alarm"] == 0
    assert probability_metrics([.5], [True], .6)["false_safe"] == 1
    assert probability_metrics([.5], [True], .6)["auroc"] is None
    assert probability_metrics([], [], .5)["log_loss"] is None


def test_zero_missing_geometry_metrics_unavailable_not_perfect():
    row = {"hidden": torch.zeros(2, 33, 33, 33, dtype=torch.bool), "targets": torch.zeros(2, 3, 4913)}
    output = common.geometry_metrics(torch.zeros(2, 3, 4913), row, {"occupied_iou": .5, "boundary_f1": .5})
    assert output["occupied_iou"]["score"] is None and output["occupied_iou"]["count"] == 0
    assert output["recovery_nmae"]["score"] is None
    assert output["clearance_nmae"]["score"] == 0


def test_engineering_screen_does_not_average_away_one_seed():
    records = []
    for seed in transfer.PLAN["seeds"]:
        for kind in ("compact", "raw", "spatial", "null"):
            records.append({"seed": seed, "kind": kind, "split": "test",
                            "metrics": {"all": {"auprc": .95 if kind != "null" else .5, "false_safe": .05, "false_alarm": .1}}})
    assert transfer.engineering_screen(records)["competitive_id_collision_head"]
    records[4]["metrics"]["all"]["false_safe"] = .11
    assert not transfer.engineering_screen(records)["competitive_id_collision_head"]
