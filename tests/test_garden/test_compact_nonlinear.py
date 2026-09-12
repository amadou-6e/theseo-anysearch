import os

import pytest
import torch

from theseo_anysearch.garden import compact_readout as heads
from theseo_anysearch.garden.pilots import compact_nonlinear as experiment


@pytest.mark.parametrize("kind", ["conv", "film"])
def test_head_shape_query_order_and_conditioning_gradients(kind):
    torch.set_num_threads(2); torch.manual_seed(8)
    head = heads.make_readout(kind)
    z = torch.randn(2, 64, requires_grad=True)
    query = torch.tensor([[0, 4912, 20, 20], [10, 100, 200, 300]])
    prediction = head(z, query)
    assert prediction.shape == (2, 3, 4)
    assert torch.allclose(head(z, query.flip(1)), prediction.flip(2), atol=1e-6)
    prediction.square().mean().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0
    with pytest.raises(ValueError):
        head(torch.randn(2, 63), query)
    with pytest.raises(ValueError):
        head(z, torch.full((2, 4), 4913))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("kind", ["conv", "film"])
def test_cuda_deterministic_training_with_duplicate_queries(kind):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    old = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(8)
        head = heads.make_readout(kind).cuda()
        z = torch.randn(2, 64, device="cuda")
        query = torch.tensor([[0, 0, 20, 20], [10, 100, 200, 300]], device="cuda")
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001)
        head(z, query).square().mean().backward(); optimizer.step()
        assert all(torch.isfinite(p).all() for p in head.parameters())
    finally:
        torch.use_deterministic_algorithms(old)


def test_grouped_ridge_matches_explicit_target_repetition():
    torch.manual_seed(9)
    x = torch.randn(24, 5, dtype=torch.float64)
    y = torch.rand(8, 3, 7, dtype=torch.float64)
    grouped = heads.grouped_ridge(x, y, 3)
    explicit = experiment.corpus.old.ridge_fit(x, y.repeat_interleave(3, 0))
    for key in ("mean", "scale", "coefficients"):
        assert torch.allclose(grouped[key], explicit[key], atol=1e-10)
    with pytest.raises(ValueError):
        heads.grouped_ridge(x, y, 2)


def test_normalization_is_probe_only_and_finite_for_constant_channels():
    x = torch.tensor([[1., 2.], [1., 4.]])
    values, statistics = heads.normalized_vectors(x)
    assert torch.equal(values[:, 0], torch.zeros(2))
    heldout = heads.normalize(torch.tensor([[9., 5.]]), statistics)
    assert heldout[0, 1] == 2
    assert torch.equal(statistics["mean"], torch.tensor([1., 3.], dtype=torch.float64))


def test_old_corpus_defaults_preserved_and_new_bank_explicit():
    c = experiment.corpus
    counts = {"probe": 12}
    old = c.data(["probe"], counts=counts)
    explicit = c.data(["probe"], counts=counts, study_id="compact-diversity-v1", bank_splits=("train",))
    assert c.old.identity(old) == c.old.identity(explicit)
    fresh = c.data(["probe"], counts=counts, study_id="compact-readout-v1", bank_splits=("probe",))
    c.support(fresh)
    assert fresh["probe"]["hidden"].shape == (12, 8, 33, 33, 33)
    assert set(old["probe"]["parents"]).isdisjoint(fresh["probe"]["parents"])


def test_control_counts_schedule_and_interruption_charge():
    configs = experiment.configs()
    assert len(configs) == 20 and len({c["id"] for c in configs}) == 20
    assert sum(c["mode"] == "candidate" for c in configs) == 12
    assert sum(c["mode"] == "oracle" for c in configs) == 6
    assert all(c["mode"] == "oracle" for c in configs[:6])
    for c in configs:
        assert experiment.learning_rate(c, 64) == pytest.approx(c["lr"])
        assert experiment.learning_rate(c, c["steps"]) == pytest.approx(.1 * c["lr"])
    assert experiment.resume_elapsed({"elapsed_seconds": 10, "inflight": 0}) == 910


def test_oracle_scores_separate_learnability_from_hidden_cell_metrics():
    target = torch.zeros(4, 3, 4913)
    target[:, 0, :100] = 1; target[:, 1, 100:200] = 1
    target[:, 2, 100:] = .2
    result = experiment.oracle_scores(target, target)
    assert result["overfit_targets_met"]
    assert result["scores"]["free_nmae"] == 0
