"""Fixed-field collision readout alternatives on an unchanged compact code."""
import torch
from torch import nn

from .collision_transfer import CollisionHead, decision_threshold


class UpsamplingCollisionHead(CollisionHead):
    def __init__(self, mode):
        if mode not in ("resize_conv", "transpose"):
            raise ValueError("unsupported learned readout")
        super().__init__(1)
        self.mode = mode
        layer = nn.Conv3d(8, 8, 3, padding=1) if mode == "resize_conv" else nn.ConvTranspose3d(8, 8, 3, stride=2, padding=1)
        self.refine = nn.Sequential(layer, nn.SiLU())

    def feature_grid(self, grids):
        if grids.ndim != 5 or grids.shape[1:] != (1, 9, 9, 9):
            raise ValueError("expected compact1x9x9x9 grid")
        if self.mode == "resize_conv":
            return self.refine(super().feature_grid(grids))
        return self.refine(self.grid(grids))


class NativeCollisionHead(CollisionHead):
    def __init__(self):
        super().__init__(3)

    def feature_grid(self, grids):
        if grids.ndim != 5 or grids.shape[1:] != (3, 17, 17, 17):
            raise ValueError("expected frozen native3x17x17x17 predictions")
        return self.grid(grids)


def uncertain(row):
    return row["unknown_path"] & ~row["visible_hit"]


def thresholds(prediction, row):
    mask = uncertain(row)
    if min(int(row["labels"][mask].sum()), int((~row["labels"][mask]).sum())) < 100:
        raise ValueError("insufficient uncertain calibration support")
    return {"pooled": decision_threshold(prediction.numpy(), row["labels"].numpy()),
            "uncertain": decision_threshold(prediction[mask].numpy(), row["labels"][mask].numpy())}
