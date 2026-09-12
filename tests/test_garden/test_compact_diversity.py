import time

import pytest
import torch

from theseo_anysearch.garden.pilots import compact_diversity as experiment
from theseo_anysearch.garden.pilots import compact_diversity_data as corpus


def test_frozen_recipes_and_schedule():
    configs = experiment.configs()
    assert len(configs) == 6
    assert [c["regularizer"] for c in configs] == [0, 0, .1, 1, .1, 1]
    assert experiment.learning_rate(configs[0], 1024) == .003
    assert experiment.learning_rate(configs[1], 1) == pytest.approx(.0003 / 32)
    assert experiment.learning_rate(configs[1], 32) == pytest.approx(.0003)
    assert experiment.learning_rate(configs[1], 1024) == pytest.approx(.00003)
    with pytest.raises(ValueError):
        experiment.learning_rate(configs[1], 0)


def test_regularizer_gradients_and_correlation():
    torch.manual_seed(3)
    z = torch.randn(128, 8, requires_grad=True)
    value = experiment.diversity_penalty(z)
    value.backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    repeated = z.detach()[:, :1].repeat(1, 8)
    assert experiment.diversity_penalty(repeated) > value.detach()
    assert experiment.diversity_penalty(torch.zeros(128, 8)) > 20
    with pytest.raises(ValueError):
        experiment.diversity_penalty(torch.zeros(1, 8))


@pytest.mark.parametrize("family", corpus.old.FAMILIES)
def test_central_density_is_not_dominated_by_filled_shells(family):
    for fraction in (.08, .16, .28):
        occ = corpus.scene(f"test-{family}-{fraction}", family, fraction)
        actual = corpus.old.helpers.prior.crop(occ, 17).mean()
        assert fraction - 1 / 4913 <= actual <= fraction + .065


def test_cache_direct_equivalence_and_hidden_truth_isolation():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    backbone = corpus.d.base.make_encoder(0, torch.device("cpu")).eval()
    occ = torch.rand(1, 33, 33, 33) < .2
    mask = torch.rand_like(occ, dtype=torch.float32) < .2
    rows = {"occupancy": occ, "hidden": mask}
    cached = corpus.features(backbone, rows, time.monotonic() + 30, device="cpu")
    changed = occ.clone(); changed[mask] = ~changed[mask]
    alternate = corpus.features(backbone, {"occupancy": changed, "hidden": mask}, time.monotonic() + 30, device="cpu")
    for side in (5, 9):
        assert torch.equal(cached[side], alternate[side])
        model = corpus.PoolingEncoder(backbone, side).eval()
        level = corpus.d.base.VoxelLevel.from_occupancy(occ.float(), unknown_mask=mask[:, None])
        with torch.no_grad():
            direct = model(level, mask[:, None])
            from_cache = model.aggregation(cached[side])
        assert torch.allclose(direct, from_cache, atol=1e-7)
        assert direct.shape == (1, 64)


def test_bank_shape_support_and_fresh_identity(monkeypatch):
    monkeypatch.setitem(corpus.COUNTS, "train", 24)
    monkeypatch.setitem(corpus.COUNTS, "selection", 24)
    rows = corpus.data(["train", "selection"])
    corpus.support(rows)
    assert rows["train"]["hidden"].shape == (24, 8, 33, 33, 33)
    assert not torch.equal(rows["train"]["hidden"][:, 0], rows["train"]["hidden"][:, 1])
    assert set(rows["train"]["parents"]).isdisjoint(rows["selection"]["parents"])
    assert all(g.startswith("compact-diversity-v1-") for g in rows["train"]["ids"])


def test_family_ranking_uses_worst_family_not_pooled_shortcut():
    def record(weak=False):
        return {"config": {"id": 0}, "steps": 256, "aggregation_parameters": 10,
                "selection": {task: {"score": 1., "families": {f: (bar / 2 if weak and i == 0 else bar)
                 for i, f in enumerate(corpus.old.FAMILIES)}} for task, bar in corpus.old.BARS.items()}}
    assert corpus.rank(record()) > corpus.rank(record(True))


def test_interrupted_recipe_zero_is_charged_and_budget_bounded(tmp_path):
    assert experiment.resume_elapsed({"elapsed_seconds": 10, "inflight": None}) == 10
    assert experiment.resume_elapsed({"elapsed_seconds": 10, "inflight": 0}) == 1810
    path = tmp_path / "ledger.json"
    ledger = experiment.search.reserve_budget(path, experiment.PLAN["run_id"], experiment.PLAN["cap_seconds"])
    assert ledger["runs"][experiment.PLAN["run_id"]]["reserved_seconds"] == 14400
    with pytest.raises(ValueError, match="exhausted"):
        experiment.search.reserve_budget(path, "too-large", 44 * 3600)


def test_optimizer_and_rng_resume_match_uninterrupted(tmp_path):
    torch.manual_seed(5)
    model = corpus.aggregation(5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    generator = torch.Generator().manual_seed(39300)
    def update(module, opt, rng):
        values = torch.randn(128, 1000, generator=rng)
        loss = module(values).square().mean() + .1 * experiment.diversity_penalty(module(values))
        opt.zero_grad(); loss.backward(); opt.step()
    update(model, optimizer, generator)
    path = tmp_path / "state.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": generator.get_state()}, path)
    update(model, optimizer, generator)
    saved = torch.load(path, weights_only=True)
    resumed = corpus.aggregation(5); resumed.load_state_dict(saved["model"])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.0003)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    resumed_rng = torch.Generator(); resumed_rng.set_state(saved["rng"])
    update(resumed, resumed_optimizer, resumed_rng)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), resumed.parameters()))
