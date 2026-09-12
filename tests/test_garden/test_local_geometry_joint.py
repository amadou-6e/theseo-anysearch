"""LG2 fixtures are synthetic and never open study geometries."""
import copy
import time

import pytest
import torch

from theseo_anysearch.garden.pilots import local_geometry_joint as lg


def test_fresh_disjoint_balanced_geometry_identities():
    ids = [r.geometry_id for s in lg.PLAN["geometries"] for r in lg.records(s)]
    assert len(ids) == len(set(ids)) == 192
    old = {r.geometry_id for s in lg.base.PLAN["geometries"] for r in lg.base.geometry_records(s)}
    assert not set(ids) & old
    for split in lg.PLAN["geometries"]:
        groups = [(r.family, r.occupancy_band) for r in lg.records(split)]
        assert len(set(groups)) == 12
        assert len({groups.count(g) for g in groups}) == 1


def test_rank_fixtures_distinguish_information_from_covariance_scale():
    f = lg.rank_fixture_report()["fixtures"]
    assert f["independent"]["effective_rank_fraction"] > .95
    assert f["invertibly_scaled"]["effective_rank_fraction"] < .25
    assert f["invertibly_scaled"]["correlation_rank_fraction"] > .95
    assert f["invertibly_scaled"]["heldout_linear_mse"] < 1e-12
    assert f["useful_rank_one"]["heldout_linear_mse"] < 1e-12
    assert f["constant"]["near_dead_fraction"] == 1
    assert f["constant"]["heldout_linear_mse"] > 1


@pytest.mark.parametrize("objective,active", [("occupancy", (True, False)), ("esdf", (False, True)), ("joint", (True, True))])
def test_joint_losses_and_masked_gradients(objective, active):
    logits = torch.zeros(1, 2, 3, 3, 3, requires_grad=True)
    mask = torch.zeros(1, 1, 3, 3, 3, dtype=torch.bool)
    mask[:, :, 1, 1, 1] = True
    target = torch.ones(1, 3, 3, 3)
    loss, occ, sdf = lg.objective_loss(logits, target, target, mask, objective)
    if objective == "joint":
        assert torch.equal(loss, occ + 10 * sdf)
    loss.backward()
    assert logits.grad[~mask.expand_as(logits)].abs().sum() == 0
    for channel, expected in enumerate(active):
        assert bool(logits.grad[:, channel].abs().sum() > 0) == expected


def test_development_training_and_query_path():
    torch.set_num_threads(2)
    torch.manual_seed(66)
    occupancy = (torch.rand(4, 9, 9, 9) > .5).float()
    data = {"occupancy": occupancy, "boundary": occupancy.clone(), "distance": 1 - 2 * occupancy}
    model, result = lg.train(data, "joint", 0, time.monotonic() + 60, steps=2)
    assert result["updates"] == 2
    assert result["initial_state_sha256"] != result["final_state_sha256"]
    assert result["initial_state_sha256"] == lg.base.encoder_state_sha256(lg.base.make_encoder(0, torch.device("cpu")))
    bank = lg.query_bank(data, "probe")
    assert lg.base.extract(model, data, bank)[lg.base.TASKS[0]].shape == (4, 256, 8)
    assert bank["sha256"] == lg.query_bank(data, "probe")["sha256"]


def fake_trials():
    result = []
    for objective in lg.PLAN["objectives"]:
        for seed in lg.PLAN["seeds"]:
            stats = {}
            for task in lg.base.TASKS:
                good = [90, 10, 10] if task in lg.base.TASKS[:2] else [5, 100]
                bad = [20, 80, 80] if task in lg.base.TASKS[:2] else [20, 100]
                stats[task] = {c: [good if c == "trained" else bad for _ in range(48)] for c in ("trained", *lg.CONTROLS)}
            result.append({"objective": objective, "seed": seed, "statistics": stats, "integrity_ok": True,
                           "mask_isolation_max_abs": 0., "rank": {"effective_rank_fraction": .125, "near_dead_fraction": 0.}})
    return result


def test_all_seeds_and_joint_preservation_required(monkeypatch):
    monkeypatch.setitem(lg.PLAN, "bootstrap_replicates", 10)
    trials = fake_trials()
    result = lg.assess(trials)
    assert result["decision"] == "joint_preserves_specialists"
    trials[-1]["statistics"]["boundary_f1"]["trained"] = [[50, 50, 50] for _ in range(48)]
    assert lg.assess(trials)["decision"] == "joint_hypothesis_not_validated"
    with pytest.raises(ValueError, match="nine"):
        lg.assess(trials[:-1])


def test_null_control_and_dead_features_remain_hard_gates(monkeypatch):
    monkeypatch.setitem(lg.PLAN, "bootstrap_replicates", 5)
    trials = fake_trials()
    trials[-1]["statistics"]["occupied_iou"]["zero"] = copy.deepcopy(trials[-1]["statistics"]["occupied_iou"]["trained"])
    assert lg.assess(trials)["recipes"]["joint"]["decision"] == "not_qualified"
    trials = fake_trials()
    trials[-1]["rank"]["near_dead_fraction"] = 1.
    assert not lg.assess(trials)["recipes"]["joint"]["integrity_pass"]


def test_paired_comparison_seed_and_geometry_validation():
    with pytest.raises(ValueError, match="3 paired"):
        lg.paired_comparison([], [], "occupied_iou")


def test_freeze_integrity_and_plan_preservation(tmp_path, monkeypatch):
    old = copy.deepcopy(lg.base.PLAN)
    monkeypatch.setattr(lg.base, "git", lambda *args: "a" * 40 if args == ("rev-parse", "HEAD") else "")
    p = tmp_path / "registration.json"
    envelope = lg.freeze(p, "b" * 40)
    assert lg.read_frozen(p) == envelope
    assert lg.base.PLAN == old
    altered = copy.deepcopy(envelope)
    altered["payload"]["plan"]["joint_esdf_weight"] = 1.
    q = tmp_path / "bad.json"
    lg.base.write_json(q, altered)
    with pytest.raises(ValueError, match="mismatch"):
        lg.read_frozen(q)
    with pytest.raises(FileExistsError):
        lg.freeze(p, "b" * 40)
