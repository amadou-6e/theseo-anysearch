import pytest
import torch
from theseo_anysearch.garden.pilots import compact_learnability as s


def test_oracle_matches_every_target_and_data_are_fresh():
    rows = s.data()
    idx = torch.arange(4913).expand(4, -1)
    assert torch.equal(s.analytic(rows["codes"], s.query_coordinates(idx).double()), rows["targets"].bool())
    assert rows["occupancy"].shape == (4, 33, 33, 33)
    assert len(set(rows["parents"])) == 4
    before = s.identity(rows)
    rows["codes"][0, 0] += .1
    assert before != s.identity(rows)


@pytest.mark.parametrize("bundle,features", [("plain32", 3), ("fourier128", 27)])
def test_decoder_gradient_and_coordinate_features(bundle, features):
    idx = torch.tensor([[0, 2456, 4912]])
    assert s.coordinates(idx, bundle).shape == (1, 3, features)
    code = torch.randn(1, 6, requires_grad=True)
    decoder = s.QueryDecoder(6, bundle)
    y = decoder(code, idx)
    assert y.shape == idx.shape
    y.square().mean().backward()
    assert code.grad.abs().sum() > 0
    assert all(p.grad is not None for p in decoder.parameters())


def test_table_and_oracle_do_not_call_voxel_provider():
    code = torch.randn(4, 6)
    assert torch.equal(s.code_for("oracle", None, None, None, code), code)
    table = torch.nn.Embedding(4, 128)
    assert s.code_for("table", table, None, None, code).shape == (4, 128)
