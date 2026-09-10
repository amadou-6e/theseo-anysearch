import numpy as np
import json
from pathlib import Path
import pytest
from theseo_anysearch.garden.pilots import family_transfer as f


@pytest.mark.parametrize("family",f.FAMILIES)
def test_new_generators_deterministic_and_nontrivial(family):
    a=f.generate("development-fixture",family,.16)
    assert a.shape==(17,17,17)
    assert np.array_equal(a,f.generate("development-fixture",family,.16))
    assert .15<a.mean()<.17
    assert not np.array_equal(a,f.generate("another-development-fixture",family,.16))


def test_missing_seeds_inconclusive():
    assert f.assess([])["outcome"]=="inconclusive"


def test_perfect_scores_pass_every_family():
    trials=[{"seed":s,"arm":arm,"task":task,"statistics":
             [[100,0,0] if task in f.d.base.TASKS[:2] else [0,256]]*48}
            for s in range(3) for arm in ("old","new") for task in f.d.base.TASKS]
    assert f.assess(trials)["outcome"]=="transfer_supported"


def test_report_replay_and_family_separation():
    root=Path(__file__).resolve().parents[2]/"docs/perception-encoder-local-geometry"
    report=json.loads((root/"transfer-report.json").read_text())
    sha=report.pop("report_payload_sha256")
    assert f.d.base.payload_sha256(report)==sha
    assert f.assess(report["trials"])==report["assessment"]
    env=json.loads((root/"transfer-preregistration.json").read_text())
    assert env==report["registration"]
    assert f.d.base.payload_sha256(env["payload"])==env["identity_sha256"]
    ids=env["payload"]["identity"]
    assert set(ids["assessment"]["families"]).isdisjoint(ids["train"]["families"])
    assert set(ids["assessment"]["families"]).isdisjoint(ids["selection"]["families"])
