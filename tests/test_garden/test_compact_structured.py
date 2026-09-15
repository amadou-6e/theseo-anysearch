import time

import numpy as np
import pytest
import torch

from theseo_anysearch.garden import compact_structured as models
from theseo_anysearch.garden.pilots import compact_structured as study


@pytest.mark.parametrize("side,channels", [(5, 1), (5, 2), (5, 4), (5, 8), (9, 1)])
def test_structured_shapes_gradients_and_query_order(side, channels):
    torch.set_num_threads(2)
    config = {"side": side, "channels": channels}
    model = models.StructuredModel(config)
    volume = torch.randn(2, 8, 33, 33, 33)
    z = model.aggregation(volume)
    assert z.shape == (2, channels * side**3)
    indices = torch.tensor([[0, 4912, 4], [17, 100, 300]])
    result = model.head(z, indices)
    expected = model.head.network(z.reshape(2, channels, side, side, side)).flatten(2)
    torch.testing.assert_close(result, expected.gather(2, indices[:, None].expand(-1, 3, -1)))
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("side,full,spacing", [(5, 9, 4), (9, 17, 2)])
def test_latent_crop_corresponds_to_central_query_extent(side, full, spacing):
    coords = torch.arange(full) * spacing
    volume = coords[None, None, :, None, None].expand(1, 1, full, full, full)
    result = models.central_latent(volume, side)
    assert result.shape == (1, 1, side, side, side)
    assert result[0, 0, 0, 0, 0] == 8
    assert result[0, 0, -1, 0, 0] == 24


def test_channel_normalization_preserves_spatial_parameter_sharing():
    config = {"kind": "structured", "side": 5, "channels": 2}
    vectors = torch.randn(12, 250)
    normalized, stats = models.normalized_vectors(vectors, config)
    x = normalized.reshape(12, 2, 125)
    torch.testing.assert_close(x.mean((0, 2)), torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(x.std((0, 2), unbiased=False), torch.ones(2), atol=1e-6, rtol=0)
    assert torch.unique(stats["mean"][:125]).numel() == 1
    assert torch.unique(stats["scale"][125:]).numel() == 1
    assert not normalized.requires_grad


def test_mask_bank_extension_preserves_default_identity():
    corpus = study.corpus
    default = corpus.data(counts={"train": 12})
    explicit = corpus.data(counts={"train": 12}, bank_sizes={"train": 8})
    assert corpus.old.identity(default) == corpus.old.identity(explicit)
    smaller = corpus.data(counts={"train": 12}, bank_sizes={"train": np.int64(2)})
    torch.testing.assert_close(default["train"]["hidden"][:, :2], smaller["train"]["hidden"])
    for n in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            corpus.data(counts={"train": 12}, bank_sizes={"train": n})


def test_campaign_plan_is_explicit_not_a_global_mutation(tmp_path):
    prior = study.prior
    before = prior.PLAN.copy()
    start = time.monotonic()
    original = prior.Campaign(tmp_path, {}, {}, start, 10)
    new = prior.Campaign(tmp_path, {}, {}, start, 10, plan=study.PLAN)
    assert original.deadline == start + before["cap_seconds"] - 10
    assert new.deadline == start + study.PLAN["cap_seconds"] - 10
    assert prior.PLAN == before
    assert new.plan["steps"] == 8192 and original.plan["steps"] == 4096


def test_frozen_search_counts_and_exposure():
    configs = study.configs()
    assert len(configs) == 12
    assert {c["dimension"] for c in configs} == {125, 250, 500, 1000, 729, 128}
    assert all(sum(x["dimension"] == c["dimension"] for x in configs) == 2 for c in configs)
    assert study.COUNTS["train"] * study.PLAN["bank_sizes"]["train"] == 12288
    assert study.COUNTS["probe"] * study.PLAN["bank_sizes"]["probe"] == 6144
    assert study.PLAN["steps"] * study.PLAN["batch"] == 4 * 4096 * 32


def test_grid_control_is_predecessor_architecture():
    config = study.configs()[10]
    actual = models.make_model(config)
    torch.manual_seed(399)
    expected = models.ReconstructionModel({"kind": "grid", "dimension": 128, "decoder": "conv"})
    for a, b in zip(actual.parameters(), expected.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_common_head_accepts_every_frozen_dimension():
    for dimension in {c["dimension"] for c in study.configs()}:
        model = study.heads.ConvolutionalReadout(dimension)
        assert model(torch.zeros(1, dimension), torch.tensor([[0, 4912]])).shape == (1, 3, 2)


def test_rejects_malformed_spatial_inputs():
    with pytest.raises(ValueError):
        models.StructuredAggregation(7, 1)
    with pytest.raises(ValueError):
        models.central_latent(torch.zeros(1, 8, 17, 17, 17), 5)
    with pytest.raises(ValueError):
        models.StructuredReadout(5, 1)(torch.zeros(1, 128), torch.tensor([[0]]))
