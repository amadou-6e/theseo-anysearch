"""Nonlinear spatial and conditioned coordinate heads for immutable compact codes."""
import torch
from torch import nn
from torch.nn import functional as F

from .compact import query_coordinates


def validate(z, indices, dimension=64):
    if z.ndim != 2 or z.shape[1] != dimension or indices.ndim != 2 or len(z) != len(indices):
        raise ValueError(f"expected batch-aligned {dimension}-vector codes and query indices")
    return query_coordinates(indices)


class ConvolutionalReadout(nn.Module):
    def __init__(self, dimension=64):
        super().__init__()
        if dimension not in (64, 128, 125, 250, 500, 729, 1000):
            raise ValueError("unsupported convolutional code dimension")
        self.dimension = dimension
        self.expand = nn.Linear(dimension, 16 * 125)
        self.spatial = nn.Sequential(
            nn.ConvTranspose3d(16, 16, 3, stride=2, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.ConvTranspose3d(16, 8, 3, stride=2, padding=1), nn.GroupNorm(4, 8), nn.SiLU(),
            nn.Conv3d(8, 3, 3, padding=1))

    def forward(self, z, indices):
        validate(z, indices, self.dimension)
        volume = self.spatial(F.silu(self.expand(z)).reshape(-1, 16, 5, 5, 5)).flatten(2)
        return volume.gather(2, indices[:, None].expand(-1, 3, -1))


class FiLMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(128, 128)
        self.second = nn.Linear(128, 128)
        self.condition = nn.Linear(64, 256)

    def forward(self, h, z):
        gamma, beta = self.condition(z).chunk(2, -1)
        value = self.first(F.silu(h)) * (1 + gamma[:, None]) + beta[:, None]
        return h + self.second(F.silu(value))


class FiLMReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("frequencies", torch.tensor([1., 2., 4., 8.]) * torch.pi)
        self.input = nn.Linear(27, 128)
        self.blocks = nn.ModuleList([FiLMBlock() for _ in range(3)])
        self.output = nn.Linear(128, 3)

    def forward(self, z, indices):
        xyz = validate(z, indices)
        angles = (xyz[..., None] * self.frequencies).flatten(-2)
        h = F.silu(self.input(torch.cat((xyz, angles.sin(), angles.cos()), -1)))
        for block in self.blocks:
            h = block(h, z)
        return self.output(F.silu(h)).transpose(1, 2)


def make_readout(kind):
    if kind == "conv":
        return ConvolutionalReadout()
    if kind == "film":
        return FiLMReadout()
    raise ValueError("unknown compact readout")


def normalized_vectors(vectors):
    x = vectors.double()
    mean = x.mean(0); scale = x.std(0, unbiased=False).clamp_min(1e-5)
    return ((x - mean) / scale).float(), {"mean": mean, "scale": scale}


def normalize(vectors, statistics):
    return ((vectors.double() - statistics["mean"]) / statistics["scale"]).float()


def grouped_ridge(vectors, targets, bank, alpha=1.):
    if bank < 1 or len(vectors) != len(targets) * bank or alpha <= 0:
        raise ValueError("invalid grouped ridge inputs")
    x = vectors.double(); mean = x.mean(0); scale = x.std(0, unbiased=False).clamp_min(1e-5)
    x = torch.cat(((x - mean) / scale, x.new_ones(len(x), 1)), 1)
    summed_x = x.reshape(len(targets), bank, -1).sum(1)
    penalty = torch.eye(x.shape[1], device=x.device, dtype=x.dtype) * alpha; penalty[-1, -1] = 0
    coefficients = torch.linalg.solve(x.T @ x + penalty, summed_x.T @ targets.flatten(1).double())
    if not torch.isfinite(coefficients).all():
        raise FloatingPointError("nonfinite ridge coefficients")
    return {"mean": mean, "scale": scale, "coefficients": coefficients, "shape": list(targets.shape[1:])}
