import numpy as np
import torch
import pytest
import json
from pathlib import Path
from theseo_anysearch.garden.pilots import context_scale as s


def test_nested_crops_and_mapped_queries():
    parent=torch.arange(49**3).reshape(1,49,49,49)
    small=s.crop(parent,17); large=s.crop(parent,33)
    assert torch.equal(s.crop(large,17),small)
    indices=torch.arange(17**3).reshape(1,-1)
    assert torch.equal(large.flatten(1).gather(1,s.mapped_indices(indices,33)),small.flatten(1))
    assert torch.equal(s.mapped_indices(indices,17),indices)


@pytest.mark.parametrize("family",s.PLAN["families"])
def test_scene_density_and_nesting(family):
    a=s.scene("development-fixture",family,.16)
    assert a.shape==(49,49,49)
    assert .159<a.mean()<.161
    assert np.array_equal(s.crop(a,17),s.crop(s.crop(a,33),17))


def test_paired_noninferiority():
    trials=[{"arm":arm,"task":task,"seed":seed,"statistics":[[100,0,0] if task in s.d.base.TASKS[:2] else [2,256]]*48}
            for arm in ("a","b") for task in s.d.base.TASKS for seed in (0,1,2)]
    result=s.compare(trials,"a","b")
    assert all(x["noninferior"] for x in result.values())
    assert all(x["gain_ci95"]==[0.,0.] for x in result.values())


def test_report_replay():
    root=Path(__file__).resolve().parents[2]/"docs/perception-encoder-local-geometry"
    report=json.loads((root/"scale-report.json").read_text()); sha=report.pop("report_payload_sha256")
    assert s.d.base.payload_sha256(report)==sha
    env=json.loads((root/"scale-preregistration.json").read_text())
    assert env==report["registration"]
    assert s.d.base.payload_sha256(env["payload"])==env["identity_sha256"]
    for name,result in report["comparisons"].items():
        a,b=name.split("_vs_")
        assert s.compare(report["trials"],a,b)==result
