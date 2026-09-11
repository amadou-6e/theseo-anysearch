"""Position-aware compact bottlenecks for fixed 33-cubed observations."""
import torch
from torch import nn
from torch.nn import functional as F


class CompactAggregation(nn.Module):
    def __init__(self, dimension=128, mode="grid"):
        super().__init__()
        if dimension not in (64, 128, 192) or mode not in ("grid", "strided", "attention"):
            raise ValueError("unsupported compact configuration")
        self.dimension, self.mode = dimension, mode
        if mode == "grid":
            self.project = nn.Sequential(nn.Linear(8 * 5**3, 128), nn.SiLU(), nn.Linear(128, dimension))
        elif mode == "strided":
            self.convs = nn.Sequential(nn.Conv3d(8, 16, 3, stride=2, padding=1), nn.SiLU(),
                                       nn.Conv3d(16, 32, 3, stride=2, padding=1), nn.SiLU())
            self.project = nn.Linear(32 * 3**3, dimension)
        else:
            self.tokens = nn.Linear(8, 32)
            self.position = nn.Parameter(torch.randn(1, 5**3, 32) * .02)
            self.query = nn.Parameter(torch.randn(1, 4, 32) * .02)
            self.attention = nn.MultiheadAttention(32, 4, batch_first=True)
            self.project = nn.Linear(4 * 32, dimension)

    def forward(self, volume):
        if volume.ndim != 5 or volume.shape[1:] != (8, 33, 33, 33):
            raise ValueError("expected B x 8 x 33 x 33 x 33 spatial features")
        if self.mode == "strided":
            return self.project(F.adaptive_avg_pool3d(self.convs(volume), 3).flatten(1))
        grid = F.adaptive_avg_pool3d(volume, 5)
        if self.mode == "grid":
            return self.project(grid.flatten(1))
        tokens = self.tokens(grid.flatten(2).transpose(1, 2)) + self.position
        pooled, _ = self.attention(self.query.expand(len(volume), -1, -1), tokens, tokens, need_weights=False)
        return self.project(pooled.flatten(1))


class CompactEncoder(nn.Module):
    def __init__(self, backbone, dimension=128, mode="grid", *, joint=False):
        super().__init__()
        self.backbone = backbone
        self.aggregation = CompactAggregation(dimension, mode)
        self.joint = joint
        self.backbone.requires_grad_(joint)
        self.backbone.projection.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if not self.joint:
            self.backbone.eval()
        return self

    def forward(self, level, hidden_mask):
        if self.joint:
            volume = self.backbone(level, hidden_mask).local_feature_volume
        else:
            with torch.no_grad():
                volume = self.backbone(level, hidden_mask).local_feature_volume
        return self.aggregation(volume)


def query_coordinates(indices):
    if indices.dtype != torch.long or bool(((indices < 0) | (indices >= 17**3)).any()):
        raise ValueError("expected central17 flattened query indices")
    xyz = torch.stack((indices // 289, indices // 17 % 17, indices % 17), -1)
    return xyz.float() / 8 - 1


def query_features(vector, indices):
    """The only decoder inputs: compact vector and normalized query location."""
    if vector.ndim != 2 or indices.ndim != 2 or len(vector) != len(indices):
        raise ValueError("batch-aligned vector and query indices required")
    return torch.cat((vector[:, None].expand(-1, indices.shape[1], -1), query_coordinates(indices)), -1)
