"""Factory for building and loading garden encoders."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from theseo_anysearch.garden.data_config import GardenConfig
from theseo_anysearch.garden.models.ae import VoxelAE, VoxelVAE, build_encoder


def build_model(cfg: GardenConfig) -> nn.Module:
    """Build the full pre-training model (encoder + decoder) from config."""
    radii = cfg.encoder.box_radii or [cfg.encoder.box_radius]
    n = 2 * max(radii) + 1  # decoder default; encoder is size-agnostic via AdaptiveAvgPool
    channels = cfg.encoder.conv_channels
    latent_dim = cfg.encoder.latent_dim

    if cfg.architecture == "custom":
        if cfg.custom_encoder is None:
            raise ValueError("architecture=custom requires a custom_encoder block")
        return cfg.custom_encoder.instantiate()

    if cfg.approach == "mvmencoder":
        from theseo_anysearch.garden.models.mvm import VoxelMVMEncoder
        mvm = cfg.mvmencoder
        return VoxelMVMEncoder(
            architecture=cfg.architecture,
            n=n,
            channels=channels,
            latent_dim=latent_dim,
            mask_ratio=mvm.mask_ratio,
            decoder_layers=mvm.decoder_layers,
            decoder_dim=mvm.decoder_dim,
            decoder_heads=mvm.decoder_heads,
        )

    if cfg.approach == "vae":
        return VoxelVAE(
            architecture=cfg.architecture,
            n=n,
            channels=channels,
            latent_dim=latent_dim,
            beta=cfg.autoencoder.beta,
        )

    return VoxelAE(
        architecture=cfg.architecture,
        n=n,
        channels=channels,
        latent_dim=latent_dim,
    )


def extract_encoder(model: nn.Module) -> nn.Module:
    """Return just the encoder sub-module from a pre-training model."""
    if hasattr(model, "encoder"):
        return model.encoder
    if hasattr(model, "_base_enc"):
        return model._base_enc
    return model


def load_encoder_from_store(garden_dir: Path, name: str, device: str = "cpu") -> nn.Module:
    """Load a saved encoder state_dict from the garden store."""
    ckpt = garden_dir / name / "encoder.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"No encoder checkpoint at {ckpt}")
    state = torch.load(ckpt, map_location=device, weights_only=True)
    # State dict only — caller must build the matching module first.
    return state
