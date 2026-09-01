"""Contract tests for Tiny perception-encoder backbone candidates."""
from __future__ import annotations

import io

import pytest
import torch

from theseo_anysearch.garden.models.backbones import (
    BackboneUnavailable,
    DenseResidualBackbone,
    SharedPyramidBackbone,
    TriPlanarBackbone,
    build_pilot_backbone,
    pilot_backbone_capabilities,
)
from theseo_anysearch.garden.models.outputs import VoxelLevel


SMALL = {
    "stem_width": 2,
    "blocks_per_stage": (1, 1, 1, 1),
    "embedding_dim": 8,
    "local_channels": 2,
}


def _level(side: int, stride: int = 1, *, partial: bool = False) -> VoxelLevel:
    generator = torch.Generator().manual_seed(side * 10 + stride)
    occupancy = torch.randint(0, 2, (1, side, side, side), generator=generator).float()
    valid = torch.ones_like(occupancy, dtype=torch.bool)
    if partial:
        valid[:, : side // 3] = False
    return VoxelLevel.from_occupancy(occupancy, validity_mask=valid, stride=stride)


@pytest.mark.parametrize("radius", [8, 16, 32])
@pytest.mark.parametrize("backbone_type", [DenseResidualBackbone, TriPlanarBackbone])
def test_single_level_backbones_satisfy_contract_at_pilot_radii(
    radius: int, backbone_type: type[torch.nn.Module]
) -> None:
    model = backbone_type(**SMALL).eval()
    level = _level(2 * radius + 1, partial=True)

    with torch.no_grad():
        output = model(level)

    assert output.global_embedding.shape == (1, 8)
    assert output.scale_embeddings[1].shape == (1, 8)
    assert output.local_feature_volume.shape == (1, 2, 2 * radius + 1, 2 * radius + 1, 2 * radius + 1)
    assert torch.equal(output.local_validity_mask, level.validity_mask)
    assert output.metadata.validity_fractions[0, 0] < 1.0


@pytest.mark.parametrize(
    ("radius", "strides"),
    [(8, (1,)), (16, (1,)), (32, (1, 2))],
)
def test_shared_pyramid_satisfies_scale_contract(radius: int, strides: tuple[int, ...]) -> None:
    model = SharedPyramidBackbone(**SMALL).eval()
    levels = {stride: _level(17, stride=stride, partial=True) for stride in strides}

    with torch.no_grad():
        output = model(levels)

    assert output.global_embedding.shape == (1, 8)
    assert tuple(output.scale_embeddings) == strides
    assert output.local_feature_volume.shape == (1, 2, 17, 17, 17)
    assert output.metadata.active_strides == strides
    assert output.metadata.validity_fractions.shape == (1, len(strides))


@pytest.mark.parametrize("name", ["dense_residual", "triplanar", "shared_pyramid"])
def test_backbone_forward_backward_and_serialization(name: str) -> None:
    model = build_pilot_backbone(name, **SMALL)
    inputs = {1: _level(9)} if name == "shared_pyramid" else _level(9)
    output = model(inputs)
    (output.global_embedding.square().mean() + output.local_feature_volume.square().mean()).backward()

    assert any(parameter.grad is not None for parameter in model.parameters())
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = build_pilot_backbone(name, **SMALL)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    restored.eval()
    model.eval()
    with torch.no_grad():
        expected = model(inputs).global_embedding
        actual = restored(inputs).global_embedding
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("name", ["dense_residual", "triplanar", "shared_pyramid"])
def test_invalid_cells_cannot_change_embeddings(name: str) -> None:
    base = _level(9, partial=True)
    changed_features = base.features.clone()
    changed_features.masked_fill_(~base.validity_mask.expand_as(changed_features), 1000.0)
    changed = VoxelLevel(changed_features, base.validity_mask, stride=1)
    model = build_pilot_backbone(name, **SMALL).eval()
    left = {1: base} if name == "shared_pyramid" else base
    right = {1: changed} if name == "shared_pyramid" else changed

    with torch.no_grad():
        left_output = model(left)
        right_output = model(right)

    torch.testing.assert_close(left_output.global_embedding, right_output.global_embedding)
    torch.testing.assert_close(left_output.local_feature_volume, right_output.local_feature_volume)


def test_voxel_level_preserves_semantic_channels() -> None:
    occupancy = torch.zeros((1, 2, 2, 2))
    occupancy[0, 0, 0, 0] = 1.0
    valid = torch.ones_like(occupancy, dtype=torch.bool)
    valid[0, 0, 0, 1] = False
    unknown = torch.zeros_like(occupancy, dtype=torch.bool)
    unknown[0, 0, 0, 1] = True

    level = VoxelLevel.from_occupancy(
        occupancy, validity_mask=valid, unknown_mask=unknown, stride=2
    )

    assert level.features[0, 0, 0, 0, 0] == 1
    assert level.features[0, 1, 0, 0, 0] == 0
    assert level.features[:, 2].sum() == 0
    assert level.features[0, 3, 0, 0, 1] == 0
    assert level.features[0, 1].sum() == 6


def test_frozen_forward_does_not_mutate_state() -> None:
    model = DenseResidualBackbone(**SMALL).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    with torch.no_grad():
        model(_level(9))

    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_sparse_backend_is_explicitly_unavailable() -> None:
    capabilities = pilot_backbone_capabilities()
    assert capabilities["dense_residual"].available
    assert capabilities["triplanar"].available
    assert capabilities["shared_pyramid"].available
    if not capabilities["sparse_residual"].available:
        assert capabilities["sparse_residual"].reason
        with pytest.raises(BackboneUnavailable, match="MinkowskiEngine"):
            build_pilot_backbone("sparse_residual")


def test_tiny_defaults_expose_192_dimensions() -> None:
    models = (DenseResidualBackbone(), TriPlanarBackbone(), SharedPyramidBackbone())
    assert all(model.embedding_dim == 192 for model in models)
    parameter_counts = [sum(parameter.numel() for parameter in model.parameters()) for model in models]
    assert max(parameter_counts) / min(parameter_counts) <= 1.10
