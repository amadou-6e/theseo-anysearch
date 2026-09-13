import torch
import json
from pathlib import Path

from theseo_anysearch.garden.pilots import context_tiles as s


def test_all_spatial_queries_have_correct_unique_owner_and_halo():
    idx = torch.arange(49**3)
    tiles, local = s.query_mapping(idx)
    assert set(tiles.tolist()) == set(range(27))
    parent = torch.arange(65**3).reshape(65, 65, 65)
    crops, _ = s.tile_inputs(parent, parent)
    actual = crops.flatten(1)[tiles, local]
    expected = parent[8:57, 8:57, 8:57].flatten()
    assert torch.equal(actual, expected)
    xyz = torch.stack((local // 33**2, local // 33 % 33, local % 33), -1)
    assert int(xyz.min()) == 8 and int(xyz.max()) == 24


def test_boundary_ownership_is_deterministic():
    idx = torch.tensor([15, 16, 31, 32, 48])
    tiles, _ = s.query_mapping(idx)
    assert tiles.tolist() == [0, 1, 1, 2, 2]


def test_fixed_no_fit_contract():
    assert s.PLAN["arms"] == ["tiled33_head33", "dense65_head33", "dense65_head65"]
    assert s.PLAN["tile_batch"] == 4
    assert len(s.STARTS) == 27


def test_completed_report_replay_and_fresh_parents():
    root = Path(__file__).resolve().parents[2] / "docs/perception-encoder-local-geometry"
    report = json.loads((root / "tiles-report.json").read_text())
    sha = report.pop("report_payload_sha256")
    assert s.d.base.payload_sha256(report) == sha
    env = json.loads((root / "tiles-preregistration.json").read_text())
    assert report["registration"] == env
    assert s.d.base.payload_sha256(env["payload"]) == env["identity_sha256"]
    assert env["payload"]["plan"] == s.PLAN
    assert report["status"] == "completed"
    for name, comparison in report["comparisons"].items():
        a, b = name.split("_vs_")
        assert s.previous.prior.compare(report["trials"], a, b) == comparison
    old = json.loads((root / "context65-preregistration.json").read_text())["payload"]["identity"]
    old_hashes = {h for split in old.values() for h in split["hashes"]}
    assert not old_hashes.intersection(env["payload"]["identity"]["assessment"]["hashes"])
