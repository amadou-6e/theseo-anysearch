"""Tests for compound-scaled Garden voxel ResNet profiles."""
from pathlib import Path

import pytest
import torch

from theseo_anysearch.garden.data_config import load_garden_config
from theseo_anysearch.garden.encoder import build_model
from theseo_anysearch.garden.models.ae import (
    VoxelResNet3D,
    binary_reconstruction_loss,
    build_encoder,
)


@pytest.mark.parametrize(("profile", "latent_dim"), [("b0", 64), ("b1", 96)])
def test_profile_output_shape_for_supported_grid_sizes(profile: str, latent_dim: int):
    encoder = VoxelResNet3D(n=5, profile=profile)
    for n in (5, 7):
        output = encoder(torch.zeros(2, n, n, n))
        assert output.shape == (2, latent_dim)


def test_b1_scales_depth_and_parameter_count_over_b0():
    b0 = VoxelResNet3D(n=5, profile="b0")
    b1 = VoxelResNet3D(n=5, profile="b1")
    b0_params = sum(parameter.numel() for parameter in b0.parameters())
    b1_params = sum(parameter.numel() for parameter in b1.parameters())

    assert len(b0.stages[1]) == 1
    assert len(b1.stages[1]) == 2
    assert b1_params > b0_params


def test_spatial_pooling_preserves_final_two_cubed_map():
    encoder = VoxelResNet3D(n=5, profile="b0", pooling="spatial_2x2x2")
    output = encoder(torch.zeros(2, 5, 5, 5))
    assert encoder.proj.in_features == 64 * 8
    assert output.shape == (2, 64)


def test_positive_weight_only_scales_occupied_voxels():
    reconstruction = torch.tensor([0.5, 0.5])
    target = torch.tensor([0.0, 1.0])
    loss = binary_reconstruction_loss(reconstruction, target, torch.tensor([3.0]))
    assert loss == pytest.approx(2.0 * torch.log(torch.tensor(2.0)).item())


@pytest.mark.parametrize("latent_dim", [16, 24, 32, 48, 64])
def test_profile_supports_projection_dimension_search(latent_dim: int):
    encoder = build_encoder(
        "voxel_resnet_3d",
        n=5,
        channels=[24, 48, 64],
        latent_dim=latent_dim,
        profile="b0",
        pooling="spatial_2x2x2",
    )
    assert encoder(torch.zeros(2, 5, 5, 5)).shape == (2, latent_dim)


def test_profile_config_allows_latent_override(tmp_path: Path):
    preset_path = Path("theseo_anysearch", "garden", "presets", "voxel_resnet_b0.yaml")
    raw = preset_path.read_text().replace("latent_dim: 64", "latent_dim: 32")
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(raw)
    cfg = load_garden_config(changed_path)
    assert build_model(cfg).encoder.latent_dim == 32


def test_profile_config_rejects_channel_drift(tmp_path: Path):
    preset_path = Path("theseo_anysearch", "garden", "presets", "voxel_resnet_b0.yaml")
    raw = preset_path.read_text().replace("conv_channels: [24, 48, 64]", "conv_channels: [16, 32, 48]")
    changed_path = tmp_path / "changed.yaml"
    changed_path.write_text(raw)
    with pytest.raises(ValueError, match="requires conv_channels"):
        load_garden_config(changed_path)


@pytest.mark.parametrize("preset", ["voxel_resnet_b0", "voxel_resnet_b1"])
def test_builtin_profile_preset_builds_reconstruction_model(preset: str):
    preset_path = Path("theseo_anysearch", "garden", "presets", f"{preset}.yaml")
    cfg = load_garden_config(preset_path)
    model = build_model(cfg)
    n = 2 * cfg.encoder.box_radius + 1
    reconstruction, latent = model(torch.zeros(2, n, n, n))

    assert reconstruction.shape == (2, 1, n, n, n)
    assert latent.shape == (2, cfg.encoder.latent_dim)


def test_b0_tuning_preset_keeps_objective_fixed():
    preset_path = Path("theseo_anysearch", "garden", "presets", "voxel_resnet_b0_tune.yaml")
    cfg = load_garden_config(preset_path)
    assert cfg.tune is not None
    assert cfg.tune.max_concurrent == 1
    assert "focal_gamma" not in cfg.tune.search_space["training"]
    assert cfg.training.loss_version == "positive_bce_v2"


def test_spatial_tuning_preset_uses_representation_score():
    preset_path = Path(
        "theseo_anysearch", "garden", "presets", "voxel_resnet_b0_spatial_tune.yaml"
    )
    cfg = load_garden_config(preset_path)
    model = build_model(cfg)
    loss = model.loss(
        torch.zeros(2, 5, 5, 5),
        pos_weight=torch.tensor([2.0]),
        dice_weight=cfg.training.dice_weight,
        face_loss_weight=cfg.training.face_loss_weight,
    )
    assert torch.isfinite(loss)
    assert cfg.encoder.pooling == "spatial_2x2x2"
    assert cfg.tune is not None
    assert cfg.tune.metric == "representation_score"
    assert cfg.tune.mode == "max"
    assert cfg.tune.num_samples == 100
    assert cfg.tune.search_space["encoder"]["latent_dim"] == {
        "choice": [16, 24, 32, 48, 64]
    }
