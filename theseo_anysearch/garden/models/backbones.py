"""Tiny perception-encoder backbones used by the preregistered pilots."""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from theseo_anysearch.garden.models.outputs import EncoderMetadata, EncoderOutput, VoxelLevel


TINY_WIDTH = 16
TRIPLANAR_TINY_WIDTH = 26
TINY_BLOCKS = (1, 1, 2, 1)
TINY_EMBEDDING_DIM = 192


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _PreActivationBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(inputs))
        residual = self.skip(hidden) if not isinstance(self.skip, nn.Identity) else inputs
        hidden = self.conv1(hidden)
        hidden = self.conv2(F.silu(self.norm2(hidden)))
        return hidden + residual


class _FeatureBackbone3D(nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        stem_width: int,
        blocks_per_stage: tuple[int, int, int, int],
        local_channels: int,
    ) -> None:
        super().__init__()
        widths = tuple(stem_width * (2**index) for index in range(4))
        self.stem = nn.Conv3d(input_channels, widths[0], kernel_size=3, padding=1, bias=False)
        stages: list[nn.Module] = []
        in_channels = widths[0]
        for stage_index, (out_channels, block_count) in enumerate(zip(widths, blocks_per_stage)):
            blocks: list[nn.Module] = []
            for block_index in range(block_count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(_PreActivationBlock3D(in_channels, out_channels, stride))
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.laterals = nn.ModuleList(
            nn.Conv3d(width, local_channels, kernel_size=1) for width in widths
        )
        self.output_channels = widths[-1]

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_size = inputs.shape[2:]
        hidden = self.stem(inputs)
        features: list[torch.Tensor] = []
        for stage in self.stages:
            hidden = stage(hidden)
            features.append(hidden)
        local = sum(
            F.interpolate(lateral(feature), size=target_size, mode="trilinear", align_corners=False)
            for lateral, feature in zip(self.laterals, features)
        )
        return features[-1], F.silu(local)


def _validity_fraction(level: VoxelLevel) -> torch.Tensor:
    return level.validity_mask.to(dtype=level.features.dtype).flatten(1).mean(dim=1)


class DenseResidualBackbone(nn.Module):
    """Pre-activation dense 3D ResNet baseline on one exact voxel volume."""

    def __init__(
        self,
        *,
        input_channels: int = 4,
        stem_width: int = TINY_WIDTH,
        blocks_per_stage: tuple[int, int, int, int] = TINY_BLOCKS,
        embedding_dim: int = TINY_EMBEDDING_DIM,
        local_channels: int | None = None,
    ) -> None:
        super().__init__()
        local_channels = local_channels or stem_width
        self.embedding_dim = embedding_dim
        self.features = _FeatureBackbone3D(
            input_channels=input_channels,
            stem_width=stem_width,
            blocks_per_stage=blocks_per_stage,
            local_channels=local_channels,
        )
        self.projection = nn.Linear(self.features.output_channels, embedding_dim)

    def forward(self, level: VoxelLevel) -> EncoderOutput:
        deep, local = self.features(level.masked_features)
        embedding = self.projection(F.adaptive_avg_pool3d(deep, 1).flatten(1))
        output = EncoderOutput(
            global_embedding=embedding,
            scale_embeddings={level.stride: embedding},
            local_feature_volume=local,
            local_validity_mask=level.validity_mask,
            metadata=EncoderMetadata(
                active_strides=(level.stride,),
                validity_fractions=_validity_fraction(level).unsqueeze(1),
            ),
        )
        return output.validate(embedding_dim=self.embedding_dim)


class _PreActivationBlock2D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(self.norm1(inputs))
        residual = self.skip(hidden) if not isinstance(self.skip, nn.Identity) else inputs
        hidden = self.conv1(hidden)
        return self.conv2(F.silu(self.norm2(hidden))) + residual


class TriPlanarBackbone(nn.Module):
    """Shared 2D residual encoder over three orthogonal center planes."""

    def __init__(
        self,
        *,
        input_channels: int = 4,
        stem_width: int = TRIPLANAR_TINY_WIDTH,
        blocks_per_stage: tuple[int, int, int, int] = TINY_BLOCKS,
        embedding_dim: int = TINY_EMBEDDING_DIM,
        local_channels: int | None = None,
    ) -> None:
        super().__init__()
        local_channels = local_channels or stem_width
        widths = tuple(stem_width * (2**index) for index in range(4))
        self.embedding_dim = embedding_dim
        self.stem = nn.Conv2d(input_channels, widths[0], kernel_size=3, padding=1, bias=False)
        stages: list[nn.Module] = []
        in_channels = widths[0]
        for stage_index, (out_channels, block_count) in enumerate(zip(widths, blocks_per_stage)):
            blocks: list[nn.Module] = []
            for block_index in range(block_count):
                stride = 2 if stage_index > 0 and block_index == 0 else 1
                blocks.append(_PreActivationBlock2D(in_channels, out_channels, stride))
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.token_projection = nn.Linear(widths[-1] * 3, embedding_dim)
        self.local_projection = nn.Conv2d(widths[-1], local_channels, kernel_size=1)

    def _encode_plane(self, plane: torch.Tensor, side: int) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.stages(self.stem(plane))
        token = F.adaptive_avg_pool2d(feature, 1).flatten(1)
        local = F.interpolate(
            self.local_projection(feature), size=(side, side), mode="bilinear", align_corners=False
        )
        return token, local

    def forward(self, level: VoxelLevel) -> EncoderOutput:
        features = level.masked_features
        side = features.shape[-1]
        middle = side // 2
        planes = (
            features[:, :, :, :, middle],
            features[:, :, :, middle, :],
            features[:, :, middle, :, :],
        )
        encoded = [self._encode_plane(plane, side) for plane in planes]
        embedding = self.token_projection(torch.cat([item[0] for item in encoded], dim=1))
        xy, xz, yz = (item[1] for item in encoded)
        local = (
            xy.unsqueeze(-1).expand(-1, -1, -1, -1, side)
            + xz.unsqueeze(3).expand(-1, -1, -1, side, -1)
            + yz.unsqueeze(2).expand(-1, -1, side, -1, -1)
        ) / 3.0
        output = EncoderOutput(
            global_embedding=embedding,
            scale_embeddings={level.stride: embedding},
            local_feature_volume=F.silu(local),
            local_validity_mask=level.validity_mask,
            metadata=EncoderMetadata(
                active_strides=(level.stride,),
                validity_fractions=_validity_fraction(level).unsqueeze(1),
            ),
        )
        return output.validate(embedding_dim=self.embedding_dim)


class SharedPyramidBackbone(nn.Module):
    """Shared 3D residual backbone with token fusion across voxel strides."""

    def __init__(
        self,
        *,
        input_channels: int = 4,
        stem_width: int = TINY_WIDTH,
        blocks_per_stage: tuple[int, int, int, int] = TINY_BLOCKS,
        embedding_dim: int = TINY_EMBEDDING_DIM,
        local_channels: int | None = None,
        maximum_scales: int = 6,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.maximum_scales = maximum_scales
        self.features = _FeatureBackbone3D(
            input_channels=input_channels,
            stem_width=stem_width,
            blocks_per_stage=blocks_per_stage,
            local_channels=local_channels or stem_width,
        )
        self.token_projection = nn.Linear(self.features.output_channels, embedding_dim)
        self.stride_embedding = nn.Embedding(maximum_scales, embedding_dim)
        self.scale_gate = nn.Linear(embedding_dim, 1)
        self.fusion_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, levels: Mapping[int, VoxelLevel]) -> EncoderOutput:
        if not levels:
            raise ValueError("the pyramid requires at least one voxel level")
        strides = tuple(sorted(levels))
        if len(strides) > self.maximum_scales:
            raise ValueError(f"at most {self.maximum_scales} pyramid levels are supported")
        if any(levels[stride].stride != stride for stride in strides):
            raise ValueError("pyramid mapping keys must equal each VoxelLevel stride")
        batch_sizes = {levels[stride].features.shape[0] for stride in strides}
        channel_counts = {levels[stride].features.shape[1] for stride in strides}
        if len(batch_sizes) != 1 or len(channel_counts) != 1:
            raise ValueError("all pyramid levels must share batch and channel dimensions")

        tokens: list[torch.Tensor] = []
        scale_embeddings: dict[int, torch.Tensor] = {}
        local: torch.Tensor | None = None
        for index, stride in enumerate(strides):
            deep, level_local = self.features(levels[stride].masked_features)
            token = self.token_projection(F.adaptive_avg_pool3d(deep, 1).flatten(1))
            token = token + self.stride_embedding.weight[index]
            tokens.append(token)
            scale_embeddings[stride] = token
            if index == 0:
                local = level_local
        sequence = torch.stack(tokens, dim=1)
        weights = torch.softmax(self.scale_gate(sequence), dim=1)
        fused = self.fusion_projection((sequence * weights).sum(dim=1))
        finest = levels[strides[0]]
        output = EncoderOutput(
            global_embedding=fused,
            scale_embeddings=scale_embeddings,
            local_feature_volume=local,
            local_validity_mask=finest.validity_mask,
            metadata=EncoderMetadata(
                active_strides=strides,
                validity_fractions=torch.stack(
                    [_validity_fraction(levels[stride]) for stride in strides], dim=1
                ),
            ),
        )
        return output.validate(embedding_dim=self.embedding_dim)


class BackboneUnavailable(RuntimeError):
    """Raised when an optional architecture backend is not installed."""


@dataclass(frozen=True)
class BackboneCapability:
    available: bool
    reason: str | None = None


def sparse_backbone_capability() -> BackboneCapability:
    """Report sparse backend availability without importing optional native code."""

    if importlib.util.find_spec("MinkowskiEngine") is None:
        return BackboneCapability(
            available=False,
            reason="MinkowskiEngine is not installed for this Python/device platform",
        )
    return BackboneCapability(
        available=False,
        reason="MinkowskiEngine was detected but the pilot sparse adapter is not enabled",
    )


def build_pilot_backbone(name: str, **kwargs: object) -> nn.Module:
    """Build a preregistered pilot backbone by stable architecture name."""

    if name == "dense_residual":
        return DenseResidualBackbone(**kwargs)
    if name == "triplanar":
        return TriPlanarBackbone(**kwargs)
    if name == "shared_pyramid":
        return SharedPyramidBackbone(**kwargs)
    if name == "sparse_residual":
        capability = sparse_backbone_capability()
        raise BackboneUnavailable(capability.reason)
    raise ValueError(f"unknown pilot backbone: {name!r}")


def pilot_backbone_capabilities() -> dict[str, BackboneCapability]:
    """Return availability used by P3 manifests and CLI reporting."""

    return {
        "dense_residual": BackboneCapability(True),
        "triplanar": BackboneCapability(True),
        "shared_pyramid": BackboneCapability(True),
        "sparse_residual": sparse_backbone_capability(),
    }
