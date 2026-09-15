"""Capacity-controlled frozen probes and geometry-blocked cross-fitting."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from theseo_anysearch.garden.models.outputs import EncoderOutput


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def encoder_state_sha256(module: nn.Module) -> str:
    """Hash parameters and buffers without dtype-dependent serialization."""

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        header = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        digest.update(
            json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
        )
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class CrossFitFold:
    """Two selection blocks and one disjoint out-of-fold reporting block."""

    fold: int
    selection_geometry_ids: tuple[str, ...]
    report_geometry_ids: tuple[str, ...]


def make_cross_fit_folds(
    geometry_ids: Sequence[str], *, seed: int
) -> tuple[CrossFitFold, CrossFitFold, CrossFitFold]:
    """Partition a 24-geometry development pool into rotating blocks of eight."""

    if len(geometry_ids) != 24 or len(set(geometry_ids)) != 24:
        raise ValueError("cross-fitting requires exactly 24 unique geometry IDs")
    ordered = sorted(
        geometry_ids,
        key=lambda geometry_id: _canonical_hash(
            {"seed": seed, "scope": "probe-cross-fit", "geometry_id": geometry_id}
        ),
    )
    blocks = tuple(tuple(ordered[index : index + 8]) for index in range(0, 24, 8))
    folds: list[CrossFitFold] = []
    for report_index, report_ids in enumerate(blocks):
        selection = tuple(
            geometry_id
            for block_index, block in enumerate(blocks)
            if block_index != report_index
            for geometry_id in block
        )
        folds.append(CrossFitFold(report_index, selection, report_ids))
    return tuple(folds)  # type: ignore[return-value]


class GlobalLinearProbe(nn.Module):
    """A single affine map over the frozen global representation."""

    def __init__(self, embedding_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, encoded: EncoderOutput) -> torch.Tensor:
        return self.linear(encoded.global_embedding)


def _sample_local_features(
    encoded: EncoderOutput, normalized_xyz: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if normalized_xyz.ndim != 3 or normalized_xyz.shape[-1] != 3:
        raise ValueError("coordinates must have shape (B, Q, 3) in normalized xyz order")
    if normalized_xyz.shape[0] != encoded.global_embedding.shape[0]:
        raise ValueError("coordinate and embedding batch dimensions must match")
    if torch.any(normalized_xyz < -1) or torch.any(normalized_xyz > 1):
        raise ValueError("probe coordinates must be normalized to [-1, 1]")
    grid = normalized_xyz[:, :, None, None, :]
    sampled = F.grid_sample(
        encoded.local_feature_volume,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    validity = F.grid_sample(
        encoded.local_validity_mask.to(dtype=sampled.dtype),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )
    return sampled[:, :, :, 0, 0].transpose(1, 2), validity[:, 0, :, 0, 0].bool()


class CoordinateProbe(nn.Module):
    """Small implicit-field probe conditioned on local/global scene features."""

    def __init__(
        self,
        local_channels: int,
        embedding_dim: int,
        output_dim: int,
        *,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = local_channels + embedding_dim + 5
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        encoded: EncoderOutput,
        normalized_xyz: torch.Tensor,
        *,
        voxel_size: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local, valid = _sample_local_features(encoded, normalized_xyz)
        batch, queries, _ = local.shape
        metadata = torch.stack((voxel_size, stride), dim=-1)
        if metadata.shape != (batch, 2):
            raise ValueError("voxel_size and stride must each have shape (B,)")
        conditioning = torch.cat(
            (
                local,
                encoded.global_embedding[:, None, :].expand(-1, queries, -1),
                normalized_xyz,
                metadata[:, None, :].expand(-1, queries, -1),
            ),
            dim=-1,
        )
        return self.network(conditioning), valid


class PairTopologyProbe(nn.Module):
    """Symmetric local-feature probe for reachability and geodesic targets."""

    def __init__(
        self,
        local_channels: int,
        embedding_dim: int,
        output_dim: int = 2,
        *,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        input_dim = local_channels * 3 + embedding_dim + 8
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        encoded: EncoderOutput,
        start_xyz: torch.Tensor,
        goal_xyz: torch.Tensor,
        *,
        voxel_size: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        start_features, start_valid = _sample_local_features(encoded, start_xyz)
        goal_features, goal_valid = _sample_local_features(encoded, goal_xyz)
        batch, pairs, _ = start_features.shape
        metadata = torch.stack((voxel_size, stride), dim=-1)
        if metadata.shape != (batch, 2):
            raise ValueError("voxel_size and stride must each have shape (B,)")
        conditioning = torch.cat(
            (
                start_features + goal_features,
                torch.abs(start_features - goal_features),
                start_features * goal_features,
                encoded.global_embedding[:, None, :].expand(-1, pairs, -1),
                start_xyz + goal_xyz,
                torch.abs(start_xyz - goal_xyz),
                metadata[:, None, :].expand(-1, pairs, -1),
            ),
            dim=-1,
        )
        return self.network(conditioning), start_valid & goal_valid


class TopologyDecoder(nn.Module):
    """Shallow local decoder whose capacity is fixed independently of the encoder."""

    def __init__(self, local_channels: int, output_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv3d(local_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, output_channels, kernel_size=1),
        )

    def forward(self, encoded: EncoderOutput) -> tuple[torch.Tensor, torch.Tensor]:
        return self.network(encoded.local_feature_volume), encoded.local_validity_mask


def extract_frozen(encoder: nn.Module, *encoder_args: object, **encoder_kwargs: object) -> EncoderOutput:
    """Extract detached features and prove that encoder state remained unchanged."""

    before = encoder_state_sha256(encoder)
    was_training = encoder.training
    try:
        encoder.eval()
        with torch.no_grad():
            encoded = encoder(*encoder_args, **encoder_kwargs)
    finally:
        encoder.train(was_training)
    if not isinstance(encoded, EncoderOutput):
        raise TypeError("frozen pilot probes require the EncoderOutput contract")
    encoded.validate()
    after = encoder_state_sha256(encoder)
    if before != after:
        raise RuntimeError("encoder state changed during frozen feature extraction")
    return encoded


def train_probe_step(
    encoder: nn.Module,
    probe: nn.Module,
    encoder_args: Sequence[object],
    probe_forward: Callable[[nn.Module, EncoderOutput], torch.Tensor],
    target: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> float:
    """Run one probe-only update while enforcing the frozen-state boundary."""

    encoder_parameter_ids = {id(parameter) for parameter in encoder.parameters()}
    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if encoder_parameter_ids & optimized_ids:
        raise ValueError("probe optimizer must not contain encoder parameters")
    before = encoder_state_sha256(encoder)
    encoded = extract_frozen(encoder, *encoder_args)
    optimizer.zero_grad(set_to_none=True)
    prediction = probe_forward(probe, encoded)
    loss = loss_fn(prediction, target)
    if not torch.isfinite(loss):
        raise FloatingPointError("probe loss is non-finite")
    loss.backward()
    optimizer.step()
    if encoder_state_sha256(encoder) != before:
        raise RuntimeError("encoder state changed during a frozen probe update")
    return float(loss.detach())
