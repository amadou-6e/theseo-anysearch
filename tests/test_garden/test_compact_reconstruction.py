import hashlib

import numpy as np
import pytest
import torch

from theseo_anysearch.garden import compact_reconstruction as models
from theseo_anysearch.garden.compact_readout import ConvolutionalReadout
from theseo_anysearch.garden.pilots import compact_reconstruction as study


@pytest.mark.parametrize("kind", ["grid", "residual"])
@pytest.mark.parametrize("dimension", [64, 128])
def test_aggregation_shape_and_gradients(kind, dimension):
    torch.set_num_threads(2)
    model = models.DetailAggregation(kind, dimension)
    volume = torch.randn(2, 8, 33, 33, 33)
    result = model(volume)
    assert result.shape == (2, dimension)
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.project[0].out_features >= dimension
    with pytest.raises(ValueError):
        model(volume[:, :, :17])


def test_spatial_query_axis_and_central_offset():
    volume = np.arange(2 * 8 * 33**3, dtype=np.float32).reshape(2, 8, 33, 33, 33)
    ids = np.array([1, 0]); indices = np.array([[0, 1, 4912], [289, 17, 100]])
    result = models.spatial_queries(volume, ids, indices)
    assert result.shape == (2, 3, 8)
    for b in range(2):
        for q in range(3):
            k = indices[b, q]
            np.testing.assert_array_equal(result[b, q], volume[ids[b], :, k // 289 + 8, k // 17 % 17 + 8, k % 17 + 8])
    result[:] = 0
    assert volume.sum() > 0
    with pytest.raises(ValueError):
        models.spatial_queries(volume, ids, indices + 4913)


def test_readout_dimensions_and_legacy_default():
    torch.manual_seed(3); default = ConvolutionalReadout()
    torch.manual_seed(3); explicit = ConvolutionalReadout(64)
    for a, b in zip(default.parameters(), explicit.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    query = torch.tensor([[4912, 0, 200]])
    head = ConvolutionalReadout(128)
    assert head(torch.randn(1, 128), query).shape == (1, 3, 3)
    with pytest.raises(ValueError):
        head(torch.randn(1, 64), query)


def test_linear_query_order():
    model = models.LinearReadout(64); z = torch.randn(2, 64)
    indices = torch.tensor([[1, 289, 4912], [0, 7, 8]])
    torch.testing.assert_close(model(z, indices), model.output(z).reshape(2, 3, 4913).gather(2, indices[:, None].expand(-1, 3, -1)))


def test_bank_label_alignment():
    target = torch.arange(2 * 3 * 4913).reshape(2, 3, 4913)
    hidden = torch.zeros(2, 8, 33, 33, 33, dtype=torch.bool)
    hidden[1, 3, 8, 8, 8] = True
    labels = study.fit_labels({"targets": target, "hidden": hidden})
    ids = torch.tensor([0, 11]); indices = torch.tensor([[0, 4912], [0, 1]])
    y, mask = study.batch_labels(labels, ids, indices)
    assert mask.tolist() == [[False, False], [True, False]]
    torch.testing.assert_close(y[1, :, 1], target[1, :, 1])


def test_readonly_cache_integrity(tmp_path):
    shape = (1, 8, 33, 33, 33); path = tmp_path / "cache.f32"
    volume = np.memmap(path, mode="w+", dtype="<f4", shape=shape)
    volume[:] = 1; volume.flush(); volume._mmap.close()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = study.open_cache(path, shape, digest)
    assert not cache.flags.writeable
    assert cache[0, 0, 0, 0, 0] == 1
    cache._mmap.close()
    with pytest.raises(ValueError):
        study.open_cache(path, shape, "0" * 64)
    with pytest.raises(ValueError):
        study.open_cache(path, (2, *shape[1:]), digest)


def test_factorial_and_resume():
    configs = study.configs()
    assert len(configs) == len({(c["kind"], c["dimension"], c["decoder"]) for c in configs}) == 8
    assert study.resume_charge({"elapsed_seconds": 10, "inflight": None}) == 10
    assert study.resume_charge({"elapsed_seconds": 10, "inflight": {"name": "train-0", "cap_seconds": 1800}}) == 1810
    assert study.resume_charge({"elapsed_seconds": 10, "inflight": {"name": "preparation", "cap_seconds": 3600}}) == 3610


def test_fresh_data_identity_and_mask_bank():
    rows = study.corpus.data(study_id="compact-reconstruction-v1", counts={"probe": 12}, bank_splits=("probe",))
    assert rows["probe"]["hidden"].shape == (12, 8, 33, 33, 33)
    assert all(gid.startswith("compact-reconstruction-v1-probe-") for gid in rows["probe"]["ids"])
    study.corpus.support(rows)


def test_spatial_reference_conditions_on_local_features():
    model = models.SpatialReference()
    x = torch.randn(2, 3, 8, requires_grad=True); indices = torch.tensor([[0, 1, 2], [3, 4, 5]])
    result = model(x, indices); result.square().mean().backward()
    assert result.shape == (2, 3, 3)
    assert x.grad.abs().sum() > 0
