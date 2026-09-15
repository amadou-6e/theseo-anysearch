"""Target-density-balanced geometry and exact-mask frozen feature caches."""
import hashlib
from numbers import Integral

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch import nn

from ..compact import ordered_pool
from . import compact_search_data as old

d = old.d
COUNTS = {"train": 768, "probe": 768, "selection": 192, "development": 192}
BANK = 8


def seed(value):
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


def scene(gid, family, fraction):
    rng = np.random.default_rng(seed(gid))
    xyz = np.stack(np.meshgrid(*([np.arange(-24, 25) / 8] * 3), indexing="ij"))
    if family == "random_field":
        field = gaussian_filter(rng.normal(size=(49, 49, 49)), 1.5, mode="reflect")
    elif family == "oblique_sheets":
        normal = rng.normal(size=3); normal /= np.linalg.norm(normal)
        field = np.abs(np.sin(rng.uniform(4, 8) * (xyz * normal[:, None, None, None]).sum(0) + rng.uniform(-np.pi, np.pi)))
    elif family in ("sphere_shells", "box_shells"):
        fields = []
        for _ in range(2):
            center = rng.uniform(-1, 1, 3)[:, None, None, None]
            if family == "sphere_shells":
                fields.append(np.abs(np.sqrt(((xyz - center) ** 2).sum(0)) - rng.uniform(.4, 1.5)))
            else:
                fields.append(np.abs((np.abs(xyz - center) - rng.uniform(.3, 1.2, 3)[:, None, None, None]).max(0)))
        field = np.minimum(*fields)
    else:
        raise ValueError("unknown family")
    return field <= np.quantile(old.helpers.prior.crop(field, 17), fraction)


def data(splits=None, *, study_id="compact-diversity-v1", counts=None, bank_splits=("train",), bank_sizes=None):
    counts = COUNTS if counts is None else counts
    bank_sizes = {} if bank_sizes is None else bank_sizes
    if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1 for n in bank_sizes.values()):
        raise ValueError("positive integer mask-bank sizes required")
    result = {}; seen = set()
    for split in (list(counts) if splits is None else splits):
        occupancy, hidden, targets, ids, hashes = [], [], [], [], []
        for i in range(counts[split]):
            gid = f"{study_id}-{split}-{i:04d}"
            occ = scene(gid, old.FAMILIES[(i % 12) // 3], [.08, .16, .28][i % 3])
            digest = hashlib.sha256(occ.tobytes()).hexdigest()
            if digest in seen:
                raise ValueError("duplicate parent")
            seen.add(digest); ids.append(gid); hashes.append(digest)
            truth = d.base.compute_geometry_targets(occ, truncation=8.)
            occupancy.append(old.helpers.prior.crop(occ, 33).copy())
            masks = [np.random.default_rng(seed(f"{gid}-mask-{j}")).random((33, 33, 33)) < .2
                     for j in range(int(bank_sizes.get(split, BANK)) if split in bank_splits else 1)]
            hidden.append(np.stack(masks) if split in bank_splits else masks[0])
            targets.append(np.stack([old.helpers.prior.crop(v, 17).copy()
                                     for v in (occ, truth.boundary, np.maximum(truth.signed_distance / 8, 0))]))
        result[split] = {"occupancy": torch.tensor(np.stack(occupancy), dtype=torch.bool),
                         "hidden": torch.tensor(np.stack(hidden), dtype=torch.bool),
                         "targets": torch.tensor(np.stack(targets), dtype=torch.float32).flatten(2),
                         "ids": ids, "parents": hashes}
    return result


def support(rows):
    from .compact_search import check_support
    check_support({s: {**v, "hidden": v["hidden"][:, 0] if v["hidden"].ndim == 5 else v["hidden"]}
                   for s, v in rows.items()})


def aggregation(side):
    if side not in (5, 9):
        raise ValueError("unsupported pooling side")
    return nn.Sequential(nn.Linear(8 * side ** 3, 128), nn.SiLU(), nn.Linear(128, 64))


class PoolingEncoder(nn.Module):
    def __init__(self, backbone, side):
        super().__init__()
        self.backbone = backbone.requires_grad_(False).eval()
        self.side = side
        self.aggregation = aggregation(side)

    def train(self, mode=True):
        super().train(mode); self.backbone.eval()
        return self

    def forward(self, level, hidden_mask):
        with torch.no_grad():
            volume = self.backbone(level, hidden_mask).local_feature_volume
        return self.aggregation(ordered_pool(volume, self.side).flatten(1))


@torch.no_grad()
def features(backbone, rows, deadline, device="cuda"):
    bank = rows["hidden"].shape[1] if rows["hidden"].ndim == 5 else 1
    hidden = rows["hidden"].reshape(-1, 33, 33, 33)
    result = {5: [], 9: []}
    for i in range(0, len(hidden), 16):
        d.check_deadline(deadline)
        indices = torch.arange(i, min(i + 16, len(hidden))) // bank
        occ = rows["occupancy"][indices].to(device=device, dtype=torch.float32)
        mask = hidden[i:i + 16, None].to(device)
        volume = backbone(d.base.VoxelLevel.from_occupancy(occ, unknown_mask=mask), mask).local_feature_volume
        for side in result:
            result[side].append(ordered_pool(volume, side).flatten(1).cpu())
    return {side: torch.cat(parts) for side, parts in result.items()}


def rank(record):
    metrics = record["selection"]
    macro, families = [], []
    for task, bar in old.BARS.items():
        values = list(metrics[task]["families"].values())
        mean = float(np.mean(values))
        ratio = lambda x: x / bar if task in d.base.TASKS[:2] else bar / max(x, 1e-8)
        macro.append(ratio(mean)); families.extend(ratio(x) for x in values)
    return (sum(x >= 1 for x in macro), min(families), float(np.minimum(macro, 2).mean()),
            -record["aggregation_parameters"], -record["steps"], -record["config"]["id"])
