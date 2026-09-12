"""Spatially arranged single-vector codes with independent layout-aware heads."""
import torch
from torch import nn

from . import compact_readout as heads
from .compact_reconstruction import ReconstructionModel
from .models.backbones import _PreActivationBlock3D


def central_latent(volume, side):
    expected = {5: 9, 9: 17}.get(side)
    if expected is None or volume.ndim != 5 or volume.shape[2:] != (expected,) * 3:
        raise ValueError("invalid central latent geometry")
    first = (expected - side) // 2
    return volume[:, :, first:first + side, first:first + side, first:first + side]


class StructuredAggregation(nn.Module):
    def __init__(self, side, channels):
        super().__init__()
        if (side, channels) not in ((5, 1), (5, 2), (5, 4), (5, 8), (9, 1)):
            raise ValueError("unsupported structured code")
        self.side = side
        self.blocks = nn.Sequential(_PreActivationBlock3D(8, 16, stride=2),
                                    _PreActivationBlock3D(16, 32, stride=2 if side == 5 else 1))
        self.channels = nn.Conv3d(32, channels, 1)

    def forward(self, volume):
        if volume.ndim != 5 or volume.shape[1:] != (8, 33, 33, 33):
            raise ValueError("expected full frozen feature volume")
        return self.channels(central_latent(self.blocks(volume), self.side)).flatten(1)


class StructuredReadout(nn.Module):
    def __init__(self, side, channels):
        super().__init__()
        if (side, channels) not in ((5, 1), (5, 2), (5, 4), (5, 8), (9, 1)):
            raise ValueError("unsupported structured readout")
        self.side, self.channels = side, channels
        layers = [nn.Conv3d(channels, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU()]
        if side == 5:
            layers.extend([nn.ConvTranspose3d(16, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU()])
        layers.extend([nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1), nn.GroupNorm(4, 8), nn.SiLU(),
                       nn.Conv3d(8, 3, 3, padding=1)])
        self.network = nn.Sequential(*layers)

    def forward(self, z, indices):
        heads.validate(z, indices, self.channels * self.side**3)
        volume = self.network(z.reshape(-1, self.channels, self.side, self.side, self.side)).flatten(2)
        return volume.gather(2, indices[:, None].expand(-1, 3, -1))


class StructuredModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.aggregation = StructuredAggregation(config["side"], config["channels"])
        self.head = StructuredReadout(config["side"], config["channels"])

    def forward(self, volume, indices):
        return self.head(self.aggregation(volume), indices)


def make_model(config):
    torch.manual_seed(399)
    if config["kind"] == "grid":
        return ReconstructionModel({"kind": "grid", "dimension": 128, "decoder": "conv"})
    return StructuredModel(config)


def make_head(config):
    torch.manual_seed(399)
    if config["kind"] == "grid":
        return heads.ConvolutionalReadout(128)
    return StructuredReadout(config["side"], config["channels"])


def normalized_vectors(vectors, config):
    if config["kind"] == "grid":
        return heads.normalized_vectors(vectors)
    x = vectors.double().reshape(len(vectors), config["channels"], config["side"]**3)
    mean = x.mean((0, 2)).repeat_interleave(config["side"]**3)
    scale = x.std((0, 2), unbiased=False).clamp_min(1e-5).repeat_interleave(config["side"]**3)
    statistics = {"mean": mean, "scale": scale}
    return heads.normalize(vectors, statistics), statistics
