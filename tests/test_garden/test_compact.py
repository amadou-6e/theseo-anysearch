import io
import pytest
import torch
from theseo_anysearch.garden.compact import CompactAggregation, CompactEncoder, query_features, ordered_pool
from theseo_anysearch.garden.pilots import local_geometry as base


@pytest.mark.parametrize("mode", ["grid", "strided", "attention"])
@pytest.mark.parametrize("dim", [64, 128, 192])
def test_shapes_gradients_and_roundtrip(mode, dim):
    torch.set_num_threads(2)
    model = CompactAggregation(dim, mode)
    x = torch.randn(2, 8, 33, 33, 33, requires_grad=True)
    y = model(x)
    assert y.shape == (2, dim) and torch.isfinite(y).all()
    y.square().mean().backward()
    assert x.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    other = CompactAggregation(dim, mode)
    other.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(model(x), other(x))


@pytest.mark.parametrize("joint", [False, True])
def test_backbone_freeze_and_hidden_truth_isolation(joint):
    model = CompactEncoder(base.make_encoder(0, torch.device("cpu")), joint=joint).train()
    occ = torch.rand(1, 33, 33, 33) > .7
    hidden = torch.rand(1, 1, 33, 33, 33) > .8
    level = base.VoxelLevel.from_occupancy(occ.float(), unknown_mask=hidden)
    altered = torch.where(hidden[:, 0], ~occ, occ)
    other = base.VoxelLevel.from_occupancy(altered.float(), unknown_mask=hidden)
    y = model(level, hidden)
    assert torch.equal(y, model(other, hidden))
    y.square().mean().backward()
    active = [p for name, p in model.backbone.named_parameters() if not name.startswith("projection")]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in active) == joint
    assert all(p.grad is None for p in model.backbone.projection.parameters())
    assert any(p.grad is not None for p in model.aggregation.parameters())


def test_query_bottleneck_and_coordinates():
    z = torch.randn(2, 64)
    idx = torch.tensor([[0, 4912], [2456, 0]])
    f = query_features(z, idx)
    assert f.shape == (2, 2, 67)
    assert torch.equal(f[:, 0, :64], z)
    assert f[0, 0, -3:].tolist() == [-1., -1., -1.]
    assert f[0, 1, -3:].tolist() == [1., 1., 1.]
    with pytest.raises(ValueError):
        query_features(z, idx + 4913)


def test_smoke_data_is_disjoint_and_query_eligible():
    from theseo_anysearch.garden.pilots import compact_smoke as s
    rows = s.data()
    hashes = [h for v in rows.values() for h in v["hashes"]]
    assert len(hashes) == len(set(hashes)) == 36
    for v in rows.values():
        for task, idx in v["indices"].items():
            hidden = s.prior.crop(v["hidden"], 17).flatten(1).gather(1, idx)
            assert bool((~hidden if task == "clearance_nmae" else hidden).all())
            if task in base.TASKS[2:]:
                occupied = s.prior.crop(v["occupancy"], 17).flatten(1).gather(1, idx)
                assert not bool(occupied.any())
    before = s.identity(rows)
    rows["train"]["targets"]["boundary_f1"][0, 0] += 1
    assert before != s.identity(rows)


def test_ordered_pool_matches_adaptive_bins_and_gradients():
    x = torch.randn(2, 3, 9, 9, 9, dtype=torch.float64, requires_grad=True)
    a = ordered_pool(x, 5)
    b = torch.nn.functional.adaptive_avg_pool3d(x, 5)
    assert torch.allclose(a, b, atol=1e-12, rtol=1e-12)
    ga = torch.autograd.grad(a.square().sum(), x, retain_graph=True)[0]
    gb = torch.autograd.grad(b.square().sum(), x)[0]
    assert torch.allclose(ga, gb, atol=1e-12, rtol=1e-12)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mode", ["grid", "strided", "attention"])
def test_cuda_deterministic_joint_backward(mode, monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        model = CompactAggregation(64, mode).cuda()
        x = torch.randn(1, 8, 33, 33, 33, device="cuda", requires_grad=True)
        model(x).square().mean().backward()
        assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    finally:
        torch.use_deterministic_algorithms(previous)


def test_engineering_report_integrity():
    import json
    from pathlib import Path
    from theseo_anysearch.garden.pilots import compact_smoke as s
    root = Path(__file__).resolve().parents[2] / "docs/perception-encoder-local-geometry"
    report = json.loads((root / "compact-smoke-v2-report.json").read_text())
    sha = report.pop("report_payload_sha256")
    assert s.d.base.payload_sha256(report) == sha
    env = json.loads((root / "compact-smoke-v2-preregistration.json").read_text())
    assert report["registration"] == env
    assert s.d.base.payload_sha256(env["payload"]) == env["identity_sha256"]
    assert env["payload"]["plan"] == s.PLAN
    assert len(report["profiles"]) == 18 and len(report["trials"]) == 6
    assert not report["promotion_eligible"]
    assert report["status"] == "engineering_smoke_completed"
