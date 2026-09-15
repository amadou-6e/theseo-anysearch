import numpy as np
import pytest
import torch
from torch.nn import functional as F

from theseo_anysearch.garden.collision_readout import NativeCollisionHead, UpsamplingCollisionHead, thresholds, uncertain
from theseo_anysearch.garden.collision_transfer import CollisionHead
from theseo_anysearch.garden.pilots import compact_collision_readout as study


@pytest.mark.parametrize("kind,channels,side,parameters", [
    ("original", 1, 9, 4521), ("resize_conv", 1, 9, 6257), ("transpose", 1, 9, 6257),
    ("native", 3, 17, 4953), ("raw", 2, 33, 4737), ("spatial", 8, 33, 6033), ("null", 1, 9, 6257)])
def test_readout_shape_parameters_and_gradients(kind, channels, side, parameters):
    model = study.make_head(401, kind)
    grid = torch.randn(2, channels, side, side, side)
    paths = torch.randint(17, (4, 9, 3)); valid = torch.ones(4, 9, dtype=torch.bool)
    result = model(grid, paths, valid, torch.tensor([0, 0, 1, 1]))
    assert result.shape == (4,) and torch.isfinite(result).all()
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.numel() for p in model.parameters()) == parameters


def test_original_feature_path_is_unchanged():
    head = CollisionHead(1); grid = torch.randn(2, 1, 9, 9, 9)
    expected = F.interpolate(head.grid(grid), size=(17, 17, 17), mode="trilinear", align_corners=True)
    torch.testing.assert_close(head.feature_grid(grid), expected, rtol=1e-5, atol=1e-6)


def test_null_initialization_matches_resize_conv():
    a = study.make_head(403, "resize_conv").state_dict(); b = study.make_head(403, "null").state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_new_readout_shapes_are_enforced():
    with pytest.raises(ValueError): NativeCollisionHead().feature_grid(torch.zeros(1, 3, 9, 9, 9))
    with pytest.raises(ValueError): UpsamplingCollisionHead("transpose").feature_grid(torch.zeros(1, 1, 17, 17, 17))
    with pytest.raises(ValueError): UpsamplingCollisionHead("unknown")


def calibration_fixture():
    y = torch.ones(1, 4200, dtype=torch.bool); y[:, -100:] = False
    unknown = torch.zeros_like(y); unknown[:, -200:] = True
    p = torch.full(y.shape, .9); p[:, -200:-100] = torch.linspace(.1, .8, 100); p[:, -100:] = torch.linspace(0, .4, 100)
    return p, {"labels": y, "unknown_path": unknown, "visible_hit": ~unknown}


def test_uncertain_calibration_does_not_hide_rare_collisions():
    p, row = calibration_fixture(); cuts = thresholds(p, row); mask = uncertain(row)
    assert cuts["pooled"] > cuts["uncertain"]
    assert int(((p < cuts["pooled"]) & row["labels"] & mask).sum()) == 100
    assert int(((p < cuts["uncertain"]) & row["labels"] & mask).sum()) == 5


def test_insufficient_uncertain_support_is_rejected():
    p, row = calibration_fixture(); row["unknown_path"][:, -101] = False
    with pytest.raises(ValueError, match="support"): thresholds(p, row)


def test_native_zero_ablation_decodes_zero_codes_not_zero_geometry():
    memo = {"inputs-test-native-401.pt": torch.rand(4, 3, 17, 17, 17),
            "native-zero-401.pt": torch.full((1, 3, 17, 17, 17), .2)}
    result = study.variants(memo, "test", "native", 401)
    assert torch.equal(result["native-zeroed"], torch.full((4, 3, 17, 17, 17), .2))
    assert torch.equal(result["native-shuffled"].sort(dim=0).values, result["native"].sort(dim=0).values)


def test_primary_geometry_weighting_excludes_only_empty_support():
    row = {"labels": torch.tensor([[True, False], [True, False], [False, False]]),
           "unknown_path": torch.tensor([[True, False], [True, True], [False, False]]),
           "visible_hit": torch.zeros(3, 2, dtype=torch.bool)}
    scores, keep = study.uncertain_scores(torch.full((3, 2), .5), row)
    assert keep.tolist() == [True, True, False]
    np.testing.assert_allclose(scores[:2], -np.log(2))


def test_improvement_screen_does_not_average_away_a_failed_seed():
    records = []; contrasts = []
    for seed in study.PLAN["seeds"]:
        for kind in study.PLAN["representations"]:
            metric = {"auprc": .9 if kind != "original" else .7, "false_safe": .05, "false_alarm": .1}
            records.append({"seed": seed, "split": "test", "kind": kind,
                            "metrics": {"pooled": {"all": metric}, "uncertain_operating_point": metric}})
        for kind in study.CANDIDATES:
            contrasts.append({"seed": seed, "split": "test", "kind": kind, "reference": "original",
                              "negative_uncertain_log_loss_difference": {"lower_95": .1}})
    assert all(v["competitive"] for v in study.assessment(records, contrasts)["candidates"].values())
    contrasts[-1]["negative_uncertain_log_loss_difference"]["lower_95"] = -.01
    result = study.assessment(records, contrasts)
    assert not result["candidates"]["native"]["id_improvement_candidate"]
    assert result["candidates"]["native"]["improvement_failures"] == [{"seed": 403, "check": "primary_ci"}]


def test_invalid_fresh_dataset_program_rejected():
    with pytest.raises(ValueError, match="program"):
        study.common.data.collision_data({}, program="")
