"""Experimental frozen spatial-code interface; no generic-head transfer claim."""
import hashlib
import json
from numbers import Integral
from pathlib import Path

import torch
from torch import nn

from .compact_structured import StructuredAggregation, StructuredReadout
from .evaluation.probes import encoder_state_sha256
from .masking import DenseMaskAwareEncoder
from .models.outputs import VoxelLevel
from .pilots.io import payload_sha256

CONTRACT = {"format": "compact-spatial-code-v1", "input_side": 33, "target_side": 17,
            "dimension": 729, "latent_shape": [1, 9, 9, 9], "latent_centers": [8, 24, 2],
            "dtype": "float32", "normalization": "frozen_probe_channel_population_float64",
            "head": "native_spatial_readout", "default_seed": 401, "seeds": [401, 402, 403],
            "status": "experimental", "promotion_eligible": False}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrozenCompactEncoder(nn.Module):
    """Return only the normalized1x9x9x9 code, flattened to729 numbers."""
    output_dim = 729
    latent_shape = (1, 9, 9, 9)

    def __init__(self, backbone, aggregation, mean, scale):
        super().__init__()
        mean, scale = mean.detach().double().clone(), scale.detach().double().clone()
        if mean.shape != (729,) or scale.shape != (729,):
            raise ValueError("expected729 normalization statistics")
        if not mean.isfinite().all() or not scale.isfinite().all() or (scale < 1e-5).any():
            raise ValueError("invalid normalization statistics")
        if not torch.equal(mean, mean[:1].expand_as(mean)) or not torch.equal(scale, scale[:1].expand_as(scale)):
            raise ValueError("normalization must preserve shared spatial scaling")
        self.backbone, self.aggregation = backbone, aggregation
        self.register_buffer("mean", mean); self.register_buffer("scale", scale)
        self.requires_grad_(False); self.train(False)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, occupancy, unknown_mask):
        if occupancy.ndim == 5 and occupancy.shape[1] == 1:
            occupancy = occupancy[:, 0]
        if unknown_mask.ndim == 5 and unknown_mask.shape[1] == 1:
            unknown_mask = unknown_mask[:, 0]
        if occupancy.ndim != 4 or occupancy.shape[1:] != (33, 33, 33) or not len(occupancy):
            raise ValueError("expected nonempty Bx33x33x33 occupancy")
        if unknown_mask.shape != occupancy.shape or unknown_mask.dtype != torch.bool:
            raise ValueError("expected matching boolean unknown mask")
        if occupancy.device != self.mean.device or unknown_mask.device != occupancy.device:
            raise ValueError("inputs and encoder must share a device")
        if not occupancy.isfinite().all() or not ((occupancy == 0) | (occupancy == 1)).all():
            raise ValueError("occupancy must be binary and finite")
        self.train(False)
        mask = unknown_mask[:, None]
        level = VoxelLevel.from_occupancy(occupancy.float(), unknown_mask=mask)
        # Preserve the verified cache/extraction batch geometry for larger calls.
        features = torch.cat([self.backbone(VoxelLevel(level.features[i:i+16], level.validity_mask[i:i+16]),
                                           mask[i:i+16]).local_feature_volume for i in range(0, len(occupancy), 16)])
        code = torch.cat([self.aggregation(features[i:i+32]) for i in range(0, len(features), 32)])
        return ((code.double() - self.mean) / self.scale).float()


def empty_encoder():
    return FrozenCompactEncoder(DenseMaskAwareEncoder(stem_width=8, local_channels=8),
                                StructuredAggregation(9, 1), torch.zeros(729), torch.ones(729))


def load_compact_package(directory, *, seed=401, device="cpu"):
    """Load a checked frozen encoder and a separate trainable native head."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    digest = manifest.pop("manifest_payload_sha256")
    if payload_sha256(manifest) != digest or manifest["contract"] != CONTRACT:
        raise ValueError("package manifest mismatch")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed not in CONTRACT["seeds"]:
        raise ValueError("unsupported seed")
    seed = int(seed)
    if set(manifest["artifacts"]) != {str(s) for s in CONTRACT["seeds"]}:
        raise ValueError("package seed set mismatch")
    record = manifest["artifacts"][str(seed)]
    name = record["file"]
    if Path(name).name != name or name in ("", ".", ".."):
        raise ValueError("unsafe artifact name")
    if file_sha(directory / name) != record["sha256"]:
        raise ValueError("package weight hash mismatch")
    saved = torch.load(directory / name, map_location="cpu", weights_only=True)
    encoder = empty_encoder(); encoder.load_state_dict(saved["encoder"])
    # Revalidate loaded statistics rather than trusting constructor defaults.
    checked = FrozenCompactEncoder(encoder.backbone, encoder.aggregation, encoder.mean, encoder.scale)
    head = StructuredReadout(9, 1); head.load_state_dict(saved["head"])
    if encoder_state_sha256(checked) != record["encoder_state_sha256"] or encoder_state_sha256(head) != record["head_state_sha256"]:
        raise ValueError("restored package state mismatch")
    return checked.to(device), head.to(device).requires_grad_(True), {**manifest, "manifest_payload_sha256": digest}
