import numpy as np
import pytest
import torch
from theseo_anysearch.garden.pilots import reference_refinement as r


def test_selection_complete_order_and_ties():
    rows=[{"recipe":x["name"],"selection_loss":1.} for x in r.RECIPES]
    assert r.select(rows)=="baseline"
    rows[2]["selection_loss"]=.5
    assert r.select(rows)=="coverage"
    with pytest.raises(ValueError): r.select(rows[:-1])


def test_dynamic_targets_hidden_and_inputs_isolated():
    occ=torch.zeros(2,17,17,17)
    occ[:,8:]=1
    boundary=torch.tensor(np.stack([r.d.base.compute_geometry_targets(x.numpy().astype(bool)).boundary for x in occ]),dtype=torch.float32)
    inputs,queries,targets=r.dynamic_batch({"occupancy":occ,"boundary":boundary},torch.tensor([0,1]),torch.Generator().manual_seed(4))
    assert torch.all(inputs[:,2].flatten(1).gather(1,queries)==1)
    assert torch.all(inputs[:,:2].flatten(2).gather(2,queries[:,None].expand(-1,2,-1))==0)
    assert torch.equal(targets,boundary.flatten(1).gather(1,queries))
    assert queries.shape==(2,256)


def test_fixed_sweep_and_budget():
    assert len(r.RECIPES)==6
    assert r.PLAN["steps"]==2048
    assert r.PLAN["checkpoints"][0]==16
    assert r.RECIPES[0]["count"]==48
    assert r.RECIPES[2]["count"]==192
