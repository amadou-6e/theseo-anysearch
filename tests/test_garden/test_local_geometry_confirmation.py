"""Only development identities are generated in these tests."""
from pathlib import Path

import numpy as np
import pytest

from theseo_anysearch.garden.pilots import local_geometry_confirmation as lg


@pytest.mark.parametrize("family", lg.FAMILIES)
def test_independent_generators_reproducible_and_nontrivial(family):
    gid = lg.identity("probe", 0, development=True)
    a = lg.generate(gid, family, .16)
    assert a.shape == (17, 17, 17) and a.dtype == np.bool_
    assert np.array_equal(a, lg.generate(gid, family, .16))
    assert abs(a.mean() - .16) < .001
    assert not np.array_equal(a, lg.generate(gid + "-other", family, .16))


def test_invalid_generator_and_density():
    with pytest.raises(ValueError):
        lg.generate("development", "invalid", .16)
    with pytest.raises(ValueError):
        lg.generate("development", "random_field", 0)


def test_fresh_disjoint_ids():
    ids = {lg.identity(s, i) for s in lg.PLAN["split_counts"] for i in range(48)}
    assert len(ids) == 96
    old = {r.geometry_id for s in lg.prior.PLAN["geometries"] for r in lg.prior.records(s)}
    assert not ids & old


def trials():
    result = []
    for seed in range(3):
        stats = {}
        for task in lg.base.TASKS:
            good = [90, 10, 10] if task in lg.base.TASKS[:2] else [5, 100]
            bad = [20, 80, 80] if task in lg.base.TASKS[:2] else [20, 100]
            stats[task] = {c: [good if c == "trained" else bad for _ in range(48)] for c in ("trained", *lg.prior.CONTROLS)}
        result.append({"seed": seed, "integrity_ok": True, "mask_isolation_max_abs": 0.,
                       "rank": {"near_dead_fraction": 0.}, "statistics": stats})
    return result


def test_family_failure_cannot_hide_in_aggregate(monkeypatch):
    monkeypatch.setitem(lg.prior.PLAN, "bootstrap_replicates", 5)
    values = trials()
    assert lg.assess(values)["decision"] == "independent_generators_confirmed"
    for i in range(48):
        if (i % 12)//3 == 0:
            values[0]["statistics"]["boundary_f1"]["trained"][i] = [20, 80, 80]
    report = lg.assess(values)
    assert not report["components"]["boundary_f1"]["family_absolute_pass"]
    assert report["decision"] == "not_confirmed"


def test_seed_integrity_required(monkeypatch):
    monkeypatch.setitem(lg.prior.PLAN, "bootstrap_replicates", 5)
    values = trials()
    values[1]["integrity_ok"] = False
    assert not lg.assess(values)["integrity_pass"]
    with pytest.raises(ValueError):
        lg.assess(values[:2])


def test_checkpoint_contract_uses_exact_prior_report():
    root = Path(__file__).resolve().parents[2]
    contract = lg.artifact_contract(root / "docs/perception-encoder-local-geometry/lg2-report.json")
    assert len(contract["files"]) == 6
    assert set(contract["states"]) == {"0", "1", "2"}


def test_registration_tamper_rejected(tmp_path):
    p = tmp_path / "registration.json"
    lg.base.write_json(p, {"payload": {"plan": lg.PLAN}, "identity_sha256": "wrong"})
    with pytest.raises(ValueError, match="mismatch"):
        lg.read_frozen(p, tmp_path)
