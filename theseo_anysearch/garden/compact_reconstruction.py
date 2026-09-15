"""Fine-detail aggregation and controls for the fixed small-field study."""
import numpy as np
import torch
from torch import nn

from .compact import ordered_pool, query_coordinates
from .compact_readout import ConvolutionalReadout, validate
from .models.backbones import _PreActivationBlock3D


class DetailAggregation(nn.Module):
    def __init__(self, kind, dimension):
        super().__init__()
        if kind not in ("grid", "residual") or dimension not in (64, 128):
            raise ValueError("unsupported reconstruction aggregation")
        self.kind = kind
        if kind == "residual":
            self.blocks = nn.Sequential(_PreActivationBlock3D(8, 16, stride=2),
                                        _PreActivationBlock3D(16, 32, stride=2),
                                        _PreActivationBlock3D(32, 64, stride=2))
        self.project = nn.Sequential(nn.Linear(8 * 17**3 if kind == "grid" else 64 * 5**3, 256),
                                     nn.SiLU(), nn.Linear(256, dimension))

    def forward(self, volume):
        if volume.ndim != 5 or volume.shape[1:] != (8, 33, 33, 33):
            raise ValueError("expected B x 8 x 33 x 33 x 33 features")
        value = ordered_pool(volume, 17) if self.kind == "grid" else self.blocks(volume)
        return self.project(value.flatten(1))


class LinearReadout(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.output = nn.Linear(dimension, 3 * 4913)

    def forward(self, vectors, indices):
        validate(vectors, indices, self.dimension)
        return self.output(vectors).reshape(-1, 3, 4913).gather(2, indices[:, None].expand(-1, 3, -1))


class ReconstructionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.aggregation = DetailAggregation(config["kind"], config["dimension"])
        self.head = (LinearReadout if config["decoder"] == "linear" else ConvolutionalReadout)(config["dimension"])

    def forward(self, volume, indices):
        return self.head(self.aggregation(volume), indices)


class SpatialReference(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(11, 64), nn.SiLU(), nn.Linear(64, 3))

    def forward(self, features, indices):
        if features.shape != (*indices.shape, 8):
            raise ValueError("expected local features for each query")
        return self.network(torch.cat((features, query_coordinates(indices)), -1)).transpose(1, 2)


def spatial_queries(volume, ids, indices):
    """Gather only central-coordinate features from a read-only CPU/mmap volume."""
    ids, indices = np.asarray(ids), np.asarray(indices)
    if ids.ndim != 1 or indices.ndim != 2 or len(ids) != len(indices):
        raise ValueError("batch-aligned IDs/indices required")
    if np.any((ids < 0) | (ids >= len(volume))) or np.any((indices < 0) | (indices >= 4913)):
        raise ValueError("query outside volume")
    x, y, z = indices // 289 + 8, indices // 17 % 17 + 8, indices % 17 + 8
    return np.array(volume[ids[:, None], :, x, y, z], copy=True)
