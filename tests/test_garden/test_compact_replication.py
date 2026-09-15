import copy

import pytest
import torch

from theseo_anysearch.garden.pilots import compact_replication as study


def metrics():
    families = ("random_field", "oblique_sheets", "sphere_shells", "box_shells")
    return {k: {"score": v, "families": dict.fromkeys(families, v)} for k, v in study.TARGETS.items()}


def records():
    result = []
    for seed in study.SEEDS:
        row = {"seed": seed, "native": metrics(), "shuffled": metrics(), "null": metrics()}
        for control in ("shuffled", "null"):
            row[control]["boundary_f1"]["families"] = dict.fromkeys(row["native"]["boundary_f1"]["families"], .1)
        result.append(row)
    return result


def test_exact_targets_pass_without_promotion():
    outcome = study.assess(records())
    assert outcome["experimental_packaging_ready"]
    assert not outcome["promotion_eligible"]


@pytest.mark.parametrize("task,value", [("occupied_iou", .599999), ("boundary_f1", .699999),
                                        ("clearance_nmae", .100001), ("recovery_nmae", .100001)])
def test_single_seed_family_failure_cannot_be_averaged_away(task, value):
    rows = records(); rows[1]["native"][task]["families"]["oblique_sheets"] = value
    result = study.assess(rows)
    assert not result["experimental_packaging_ready"]
    assert result["failures"][0]["seed"] == 402


@pytest.mark.parametrize("control", ["null", "shuffled"])
def test_embedding_necessity(control):
    rows = records(); rows[2][control]["boundary_f1"]["families"]["box_shells"] = .65
    assert study.assess(rows)["failures"][0]["task"] == "embedding_necessity"


def test_all_seeds_families_and_finite_values_required():
    with pytest.raises(ValueError):
        study.assess(records()[:2])
    rows = records(); del rows[0]["native"]["occupied_iou"]["families"]["box_shells"]
    with pytest.raises(ValueError):
        study.assess(rows)
    rows = records(); rows[0]["native"]["occupied_iou"]["families"]["box_shells"] = float("nan")
    with pytest.raises(ValueError):
        study.assess(rows)


def test_seeded_models_do_not_reset_to_predecessor_seed():
    a = study.make_model(401); b = study.make_model(402); repeated = study.make_model(401)
    assert any(not torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters()))
    assert all(torch.equal(x, y) for x, y in zip(a.parameters(), repeated.parameters()))
    head = study.make_head(401); null = study.make_head(401)
    assert all(torch.equal(x, y) for x, y in zip(head.parameters(), null.parameters()))
    assert study.CONFIG == {k: v for k, v in study.parent.configs()[8].items() if k != "id"}


def test_fixed_recipe_shapes_and_gradients():
    model = study.make_model(401)
    values = torch.randn(2, 8, 33, 33, 33); indices = torch.tensor([[0, 4912], [18, 289]])
    code = model.aggregation(values)
    assert code.shape == (2, 729)
    output = model.head(code, indices)
    assert output.shape == (2, 3, 2)
    output.sum().backward()
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters())


def test_registration_tamper_and_threshold_lock():
    payload = {"plan": study.PLAN, "config": study.CONFIG}
    env = {"payload": payload, "identity_sha256": study.d.base.payload_sha256(payload)}
    assert study.check_registration(env) == payload
    wrong = copy.deepcopy(env); wrong["payload"]["config"]["lr"] = .001
    with pytest.raises(ValueError):
        study.check_registration(wrong)
    rows = [{"seed": s, **{k: {"thresholds": {"occupied_iou": .5, "boundary_f1": .6}}
                           for k in ("native", "spatial", "null")}} for s in study.SEEDS]
    assert study.lock_thresholds(env, rows)["seeds"] == list(study.SEEDS)
    with pytest.raises(ValueError):
        study.lock_thresholds(env, rows[::-1])


def test_fresh_data_arguments_and_legacy_plan_unchanged(monkeypatch):
    seen = {}
    monkeypatch.setattr(study.corpus, "data", lambda splits, **kwargs: seen.update(kwargs))
    study.data(["selection"])
    assert seen["study_id"] == "compact-replication-v1"
    assert seen["bank_sizes"] == {"train": 2, "probe": 4}
    assert study.parent.PLAN["run_id"] == "compact-structured-v1-run1"
    assert study.parent.PLAN["sampling_seed"] == 39900
