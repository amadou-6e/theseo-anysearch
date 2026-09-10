import numpy as np
import json
from pathlib import Path
import pytest
import torch

from theseo_anysearch.garden.pilots import context_65 as s


@pytest.mark.parametrize("family", s.PLAN["families"])
def test_parent_and_nested_queries(family):
    parent = s.scene("development-65-fixture", family, .16)
    assert parent.shape == (81, 81, 81)
    assert .159 < parent.mean() < .161
    assert np.array_equal(parent, s.scene("development-65-fixture", family, .16))
    x = torch.arange(81**3).reshape(1, 81, 81, 81)
    indices = torch.arange(17**3).reshape(1, -1)
    for side in (33, 65):
        crop = s.prior.crop(x, side)
        assert torch.equal(crop.flatten(1).gather(1, s.prior.mapped_indices(indices, side)), s.prior.crop(x, 17).flatten(1))


def test_memory_batch_four():
    batch, attempts = s.select_batch(lambda b: [{"inference": {"peak_allocated_bytes": 100}}])
    assert batch == 4 and len(attempts) == 1


@pytest.mark.parametrize("oom", [False, True])
def test_memory_fallback_common_batch(oom):
    called = []

    def measure(batch):
        called.append(batch)
        if batch == 4 and oom:
            raise torch.cuda.OutOfMemoryError("fixture")
        return [{"inference": {"peak_allocated_bytes": s.PLAN["peak_cap_bytes"] + (1 if batch == 4 else 0)}}]

    batch, attempts = s.select_batch(measure)
    assert batch == 1 and called == [4, 1]
    assert not attempts[0]["accepted"] and attempts[1]["accepted"]


def test_memory_stop_without_quality():
    batch, attempts = s.select_batch(lambda b: [{"inference": {"peak_allocated_bytes": s.PLAN["peak_cap_bytes"] + 1}}])
    assert batch is None and len(attempts) == 2


def test_host_data_identity_and_eligibility(monkeypatch):
    monkeypatch.setitem(s.PLAN, "counts", {"train": 6})
    rows = s.data()
    original = s.identity(rows)
    v = rows["train"]
    assert v["occupancy"].shape == (6, 65, 65, 65)
    assert v["occupancy"].device.type == "cpu"
    assert len(set(v["hashes"])) == 6
    for task, indices in v["indices"].items():
        hidden = s.prior.crop(v["hidden"], 17).flatten(1).gather(1, indices)
        assert bool((~hidden if task == "clearance_nmae" else hidden).all())
        if task in s.d.base.TASKS[2:]:
            occupancy = s.prior.crop(v["occupancy"], 17).flatten(1).gather(1, indices)
            assert not bool(occupancy.any())
        assert v["targets"][task].shape == (6, 256)
    v["targets"]["boundary_f1"][0, 0] += 1
    assert s.identity(rows) != original


def test_completed_report_replay():
    root = Path(__file__).resolve().parents[2] / "docs/perception-encoder-local-geometry"
    report = json.loads((root / "context65-report.json").read_text())
    sha = report.pop("report_payload_sha256")
    assert s.d.base.payload_sha256(report) == sha
    env = json.loads((root / "context65-preregistration.json").read_text())
    assert report["registration"] == env
    assert s.d.base.payload_sha256(env["payload"]) == env["identity_sha256"]
    assert env["payload"]["plan"] == s.PLAN
    assert report["status"] == "completed"
    assert s.prior.compare(report["trials"], "frozen65", "frozen33") == report["comparisons"]
