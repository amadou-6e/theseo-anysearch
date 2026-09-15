"""Small layout-aware collision heads and support-aware probability metrics."""
import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torch import nn
from torch.nn import functional as F


class CollisionHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.grid = nn.Sequential(nn.Conv3d(channels, 8, 3, padding=1), nn.GroupNorm(2, 8), nn.SiLU(),
                                  nn.Conv3d(8, 8, 3, padding=1), nn.GroupNorm(2, 8), nn.SiLU())
        self.point = nn.Sequential(nn.Linear(11, 32), nn.SiLU())
        self.output = nn.Sequential(nn.Linear(65, 32), nn.SiLU(), nn.Linear(32, 1))
        positions = torch.arange(17) / 2; lo = positions.long(); hi = positions.ceil().long(); weight = positions - lo
        self.register_buffer("interpolation", F.one_hot(lo, 9).float() * (1 - weight[:, None]) + F.one_hot(hi, 9).float() * weight[:, None])

    def forward(self, grids, paths, valid, geometry_indices):
        if grids.ndim != 5 or grids.shape[2:] not in ((9, 9, 9), (33, 33, 33)):
            raise ValueError("expected9 or33 cubic feature grids")
        if paths.ndim != 3 or paths.shape[1:] != (9, 3) or valid.shape != paths.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("expected padded9-node paths and boolean validity")
        if paths.dtype != torch.long or ((paths < 0) | (paths >= 17)).any() or not valid.any(1).all():
            raise ValueError("invalid query coordinates")
        x = self.grid(grids)
        if x.shape[-1] == 9:
            # Separable aligned-corner interpolation avoids nondeterministic CUDA resize backward.
            x = torch.einsum("oi,bcijk->bcojk", self.interpolation, x)
            x = torch.einsum("oj,bcijk->bciok", self.interpolation, x)
            x = torch.einsum("ok,bcijk->bcijo", self.interpolation, x)
        else:
            x = x[:, :, 8:25, 8:25, 8:25]
        indices = paths[..., 0] * 289 + paths[..., 1] * 17 + paths[..., 2]
        features = x.flatten(2)[geometry_indices].gather(2, indices[:, None].expand(-1, 8, -1)).transpose(1, 2)
        features = self.point(torch.cat((features, paths.float() / 8 - 1), -1))
        mean = (features * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
        maximum = features.masked_fill(~valid[..., None], -torch.inf).max(1).values
        return self.output(torch.cat((mean, maximum, valid.sum(1, keepdim=True).float() / 9), 1)).squeeze(1)


def decision_threshold(probability, labels):
    p = np.asarray(probability); y = np.asarray(labels, dtype=bool)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not y.any():
        raise ValueError("valid probabilities and calibration positives required")
    positive = np.sort(p[y])
    return float(positive[len(positive) - math.ceil(.95 * len(positive))])


def probability_metrics(probability, labels, threshold):
    p = np.asarray(probability, dtype=np.float64).reshape(-1); y = np.asarray(labels, dtype=bool).reshape(-1)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid probabilities")
    positives, negatives = int(y.sum()), int((~y).sum())
    if not len(y): return {"count": 0, "positives": 0, "negatives": 0, "auprc": None, "auroc": None,
                           "log_loss": None, "brier": None, "false_safe": None, "false_alarm": None, "safe_fraction": None}
    decision = p >= threshold; clipped = np.clip(p, 1e-6, 1 - 1e-6)
    return {"count": len(y), "positives": positives, "negatives": negatives,
            "auprc": float(average_precision_score(y, p)) if positives and negatives else None,
            "auroc": float(roc_auc_score(y, p)) if positives and negatives else None,
            "log_loss": float(-(y * np.log(clipped) + (~y) * np.log1p(-clipped)).mean()),
            "brier": float(((p - y) ** 2).mean()), "false_safe": float((~decision & y).sum() / positives) if positives else None,
            "false_alarm": float((decision & ~y).sum() / negatives) if negatives else None, "safe_fraction": float((~decision).mean())}
