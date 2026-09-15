"""Development-only fixtures; these tests never materialize LG1 study data."""
import copy
import json

import pytest
import torch

from theseo_anysearch.garden.masking import mask_isolation_max_abs
from theseo_anysearch.garden.models.outputs import VoxelLevel
from theseo_anysearch.garden.pilots import local_geometry as lg


def test_geometry_identities_are_balanced_disjoint_and_fresh():
    pools = [lg.geometry_records(split) for split in lg.PLAN["geometries"]]
    ids = [record.geometry_id for pool in pools for record in pool]
    assert len(ids) == len(set(ids)) == 192
    for pool in pools:
        counts = {(family, band): sum(r.family == family and r.occupancy_band == band for r in pool)
                  for family in lg.FAMILIES for band in lg.BANDS}
        assert len(set(counts.values())) == 1
    assert not set(ids) & {r.geometry_id for r in lg.geometry_records("train", development=True)}


def test_masks_do_not_depend_on_occupancy():
    zeros, ones = torch.zeros(2, 9, 9, 9), torch.ones(2, 9, 9, 9)
    a = lg.independent_mask(zeros, torch.Generator().manual_seed(7))
    b = lg.independent_mask(ones, torch.Generator().manual_seed(7))
    assert torch.equal(a, b)
    assert a.any() and (~a).any()


def test_local_mask_isolation_and_global_projection_exclusion():
    torch.set_num_threads(2)
    model = lg.make_encoder(0, torch.device("cpu"))
    occupancy = torch.rand(1, 9, 9, 9)
    hidden = lg.independent_mask(occupancy, torch.Generator().manual_seed(3))
    assert mask_isolation_max_abs(model, VoxelLevel.from_occupancy(occupancy), hidden) == 0
    assert all(not p.requires_grad for p in model.projection.parameters())


@pytest.mark.parametrize("objective", ["occupancy", "esdf"])
def test_pretraining_only_supervises_hidden_cells(objective):
    logits = torch.zeros(1, 1, 3, 3, 3, requires_grad=True)
    target = torch.ones(1, 3, 3, 3)
    hidden = torch.zeros_like(logits, dtype=torch.bool)
    hidden[:, :, 1, 1, 1] = True
    lg.pretraining_loss(logits, target, hidden, objective).backward()
    assert logits.grad[~hidden].abs().sum() == 0
    assert logits.grad[hidden].abs().sum() > 0


def test_empty_pretraining_mask_rejected():
    with pytest.raises(ValueError, match="empty"):
        lg.pretraining_loss(torch.zeros(1, 1, 3, 3, 3), torch.zeros(1, 3, 3, 3),
                            torch.zeros(1, 1, 3, 3, 3, dtype=torch.bool), "occupancy")


def test_gather_features_uses_per_geometry_query_indices():
    volume = torch.arange(16).reshape(2, 1, 2, 2, 2)
    actual = lg.gather_features(volume, torch.tensor([[0, 7], [1, 2]]))
    assert actual[:, :, 0].tolist() == [[0, 7], [9, 10]]


def test_exact_f1_and_iou_have_no_query_neighbor_tolerance():
    truth = torch.tensor([[0., 1., 0., 0.]])
    prediction = torch.tensor([[1., 0., 0., 0.]])
    for task in lg.TASKS[:2]:
        assert lg.score(lg.sufficient_statistics(prediction, truth, task), task) == 0
    assert lg.score(lg.sufficient_statistics(truth, truth, lg.TASKS[1]), lg.TASKS[1]) == 1


def test_regression_normalization_not_reestimated_at_evaluation():
    truth = torch.tensor([[0.25, 0.5]])
    predictions = torch.tensor([[0.125, 0.375]])
    assert lg.score(lg.sufficient_statistics(predictions, truth, lg.TASKS[2]), lg.TASKS[2]) == .125


def test_probe_does_not_modify_features_or_use_evaluation_statistics():
    torch.manual_seed(0)
    features = torch.randn(3, 16, 8)
    original = features.clone()
    labels = (features[:, :, 0] > 0).float()
    fitted = lg.fit_probe(features, labels, lg.TASKS[0], 0, steps=2)
    mean = fitted[1].clone()
    predictions = lg.predict(fitted, features + 100, lg.TASKS[0])
    assert predictions.shape == labels.shape
    assert torch.equal(features, original) and torch.equal(mean, fitted[1])


def test_rank_detects_collapse():
    result = lg.rank_diagnostics(torch.zeros(2, 10, 8))
    assert result == {"effective_rank_fraction": 0.0, "near_dead_fraction": 1.0}


def trial_fixture():
    trials = []
    for objective in lg.PLAN["objectives"]:
        for seed in lg.PLAN["seeds"]:
            stats = {}
            for task in lg.TASKS:
                good = [90, 10, 10] if task in lg.TASKS[:2] else [5, 100]
                bad = [20, 80, 80] if task in lg.TASKS[:2] else [20, 100]
                stats[task] = {name: [good if name == "trained" else bad for _ in range(48)]
                               for name in ("trained", "random", "zero", "shuffle", "shuffled_labels")}
            trials.append({"objective": objective, "seed": seed, "statistics": stats,
                           "integrity_ok": True, "mask_isolation_max_abs": 0.0,
                           "rank": {"effective_rank_fraction": .8, "near_dead_fraction": 0.0}})
    return trials


def test_joint_gate_requires_every_metric_seed_and_control(monkeypatch):
    monkeypatch.setitem(lg.PLAN, "bootstrap_replicates", 10)
    trials = trial_fixture()
    assert lg.assess(trials)["occupancy"]["decision"] == "qualifies_local_followup"
    trials[0]["statistics"][lg.TASKS[0]]["zero"] = copy.deepcopy(trials[0]["statistics"][lg.TASKS[0]]["trained"])
    assert lg.assess(trials)["occupancy"]["decision"] == "not_qualified"
    with pytest.raises(ValueError, match="every preregistered seed"):
        lg.assess(trials[:1])


def test_registration_tamper_rejected_without_git_or_data(tmp_path):
    path = tmp_path / "registration.json"
    lg.write_json(path, {"payload": {"plan": lg.PLAN}, "identity_sha256": "tampered"})
    with pytest.raises(ValueError, match="mismatch"):
        lg.read_frozen(path)
    with pytest.raises(FileExistsError):
        lg.write_json(path, {})


def test_freeze_and_read_use_exact_source_and_identities(tmp_path, monkeypatch):
    def fake_git(*args):
        return "a" * 40 if args == ("rev-parse", "HEAD") else ""
    monkeypatch.setattr(lg, "git", fake_git)
    path = tmp_path / "registration.json"
    envelope = lg.freeze(path, "b" * 40)
    assert lg.read_frozen(path) == envelope
    changed = json.loads(path.read_text())
    changed["payload"]["geometry_ids"]["train"][0] = "old-pilot-id"
    changed["identity_sha256"] = lg.payload_sha256(changed["payload"])
    altered = tmp_path / "altered.json"
    lg.write_json(altered, changed)
    with pytest.raises(ValueError, match="split identities"):
        lg.read_frozen(altered)


def test_local_program_materializes_only_development_identity():
    record = lg.geometry_records("train", development=True)[0]
    observation = lg.make_pilot_observation(record, 1, radius=8, program=lg.LOCAL_PROGRAM)
    assert observation.geometry_id.startswith("lg1-development-")
    assert observation.occupancy.shape == (17, 17, 17)
    assert not observation.unknown_mask.any()


def test_development_training_query_and_raw_mask_path(monkeypatch):
    import time
    monkeypatch.setitem(lg.PLAN, "encoder_steps", 2)
    monkeypatch.setitem(lg.PLAN, "encoder_batch", 2)
    torch.manual_seed(99)
    occupancy = (torch.rand(4, 9, 9, 9) > .5).float()
    data = {"occupancy": occupancy, "boundary": occupancy.clone(),
            "distance": 1 - 2 * occupancy}
    model, result = lg.train_encoder(data, "esdf", 0, time.monotonic() + 60)
    assert result["updates"] == 2
    assert result["initial_state_sha256"] != result["final_state_sha256"]
    assert lg.encoder_state_sha256(lg.make_encoder(0, torch.device("cpu"))) == result["initial_state_sha256"]
    bank = lg.query_bank(data, "probe")
    features = lg.extract(model, data, bank)
    assert features[lg.TASKS[0]].shape == (4, 256, 8)
    raw = lg.extract(None, data, bank)
    altered = {**data, "occupancy": torch.where(bank["hidden"][:, 0], 1 - occupancy, occupancy)}
    changed_raw = lg.extract(None, altered, bank)
    for task in lg.TASKS:
        assert torch.equal(raw[task], changed_raw[task])
