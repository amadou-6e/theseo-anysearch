"""Mask sampling and visible-cell-only dense convolution for pilot recipes."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from theseo_anysearch.garden.models.outputs import EncoderMetadata, EncoderOutput, VoxelLevel


@dataclass(frozen=True)
class MaskSample:
    hidden_mask: torch.Tensor
    requested_ratio: float
    actual_ratio: float
    center_strata: dict[str, int]


def _candidate_center_masks(
    occupancy: torch.Tensor, valid: torch.Tensor, unknown: torch.Tensor
) -> dict[str, torch.Tensor]:
    occupied = occupancy & valid & ~unknown
    free = valid & ~occupied & ~unknown
    kernel = torch.ones(1, 1, 3, 3, 3, device=occupancy.device)
    free_neighbors = F.conv3d(free.float(), kernel, padding=1) > 0
    unknown_neighbors = F.conv3d(unknown.float(), kernel, padding=1) > 0
    boundary = occupied & free_neighbors
    frontier = (free & unknown_neighbors) | (
        unknown & F.conv3d(free.float(), kernel, padding=1).bool()
    )
    return {
        "boundary_frontier": boundary | frontier,
        "ordinary_free": free,
        "unknown": unknown & valid,
        "occupied": occupied,
    }


def _patch_values(values: torch.Tensor, patch_side: int) -> torch.Tensor:
    """Return flattened per-patch sums for one (1, D, H, W) boolean tensor."""

    spatial = values.shape[1:]
    padded = tuple(math.ceil(side / patch_side) * patch_side for side in spatial)
    padding = (0, padded[2] - spatial[2], 0, padded[1] - spatial[1], 0, padded[0] - spatial[0])
    array = F.pad(values.float(), padding)
    return (
        array.reshape(
            padded[0] // patch_side,
            patch_side,
            padded[1] // patch_side,
            patch_side,
            padded[2] // patch_side,
            patch_side,
        )
        .sum(dim=(1, 3, 5))
        .flatten()
    )


def sample_patch_mask(
    occupancy: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    unknown_mask: torch.Tensor | None = None,
    ratio: float = 0.60,
    patch_side: int = 2,
    seed: int = 0,
) -> MaskSample:
    """Sample deterministic patches with a 50/30/20 center-stratum schedule."""

    if occupancy.ndim == 4:
        occupancy = occupancy[:, None]
    if occupancy.ndim != 5 or occupancy.shape[1] != 1:
        raise ValueError("mask sampling expects occupancy with shape (B, 1, D, H, W)")
    if not 0 < ratio < 1 or patch_side not in {2, 4, 8}:
        raise ValueError("mask ratio must be in (0, 1) and patch side one of 2, 4, 8")
    valid = (
        torch.ones_like(occupancy, dtype=torch.bool)
        if valid_mask is None
        else valid_mask.bool()
    )
    unknown = (
        torch.zeros_like(occupancy, dtype=torch.bool)
        if unknown_mask is None
        else unknown_mask.bool()
    )
    if valid.shape != occupancy.shape or unknown.shape != occupancy.shape:
        raise ValueError("occupancy, validity, and unknown masks must share a shape")
    occupied = (occupancy > 0.5) & valid & ~unknown
    if not torch.all(occupied.flatten(1).any(dim=1)):
        raise ValueError("every masked observation must contain an occupied target cell")

    # Draw scalar candidate indices on the CPU so a fixed seed selects the same
    # patches on CPU and CUDA. The selected index is converted to a Python int
    # before indexing the device-local candidate tensor.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.zeros_like(valid)
    counts = {"boundary_frontier": 0, "ordinary_free": 0, "unknown": 0}
    schedule = ("boundary_frontier",) * 5 + ("ordinary_free",) * 3 + ("unknown",) * 2
    spatial_shape = occupancy.shape[2:]
    for batch_index in range(len(occupancy)):
        candidates = _candidate_center_masks(
            occupancy[batch_index : batch_index + 1] > 0.5,
            valid[batch_index : batch_index + 1],
            unknown[batch_index : batch_index + 1],
        )
        target_count = max(1, math.ceil(ratio * int(valid[batch_index].sum())))
        grid_shape = tuple(math.ceil(side / patch_side) for side in spatial_shape)
        patch_labels = torch.full((math.prod(grid_shape),), 3, dtype=torch.long)
        priorities = ("ordinary_free", "unknown", "boundary_frontier")
        for label, stratum in enumerate(priorities):
            selected_patches = _patch_values(
                candidates[stratum][0], patch_side
            ).cpu() > 0
            patch_labels[selected_patches] = label
        queues: dict[str, list[int]] = {}
        for label, stratum in enumerate(priorities):
            available = torch.nonzero(patch_labels == label).flatten()
            order = torch.randperm(len(available), generator=generator)
            queues[stratum] = available[order].tolist()
        fallback = torch.nonzero(patch_labels == 3).flatten()
        fallback_order = torch.randperm(len(fallback), generator=generator)
        queues["boundary_frontier"].extend(fallback[fallback_order].tolist())

        valid_counts = _patch_values(valid[batch_index], patch_side).cpu()
        occupied_patches = _patch_values(occupied[batch_index], patch_side).cpu() > 0
        selected_indices: list[int] = []
        selected_set: set[int] = set()
        covered = 0
        step = 0
        while covered < target_count:
            requested = schedule[step % len(schedule)]
            available = queues[requested]
            if not available:
                requested = next((name for name, queue in queues.items() if queue), "")
                if not requested:
                    raise RuntimeError("could not reach requested mask ratio")
                available = queues[requested]
            patch_index = available.pop()
            if patch_index not in selected_set:
                selected_indices.append(patch_index)
                selected_set.add(patch_index)
                covered += int(valid_counts[patch_index])
                counts[requested] += 1
            step += 1
        if not any(bool(occupied_patches[index]) for index in selected_indices):
            forced = torch.nonzero(occupied_patches).flatten().tolist()
            if not forced:
                raise RuntimeError("mask sampler found no occupied patch")
            selected_indices.append(forced[0])
            counts["boundary_frontier"] += 1

        patch_grid = torch.zeros(grid_shape, dtype=torch.bool)
        patch_grid.flatten()[selected_indices] = True
        expanded = patch_grid.repeat_interleave(patch_side, 0)
        expanded = expanded.repeat_interleave(patch_side, 1)
        expanded = expanded.repeat_interleave(patch_side, 2)
        expanded = expanded[
            : spatial_shape[0], : spatial_shape[1], : spatial_shape[2]
        ].to(device=occupancy.device)
        hidden[batch_index, 0] = expanded & valid[batch_index, 0]
        if not torch.any(hidden[batch_index] & occupied[batch_index]):
            raise RuntimeError("mask sampler failed to hide a nonempty patch")
    actual = float(hidden.sum() / valid.sum())
    return MaskSample(hidden, ratio, actual, counts)


class MaskAwareNorm3d(nn.Module):
    """Per-channel normalization using supported cells only."""

    def __init__(self, channels: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        weights = support.to(dtype=inputs.dtype).expand_as(inputs)
        count = weights.sum(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1)
        mean = (inputs * weights).sum(dim=(0, 2, 3, 4), keepdim=True) / count
        variance = (
            (inputs - mean).square() * weights
        ).sum(dim=(0, 2, 3, 4), keepdim=True) / count
        normalized = (inputs - mean) * torch.rsqrt(variance + self.epsilon)
        affine = (
            normalized * self.weight[None, :, None, None, None]
            + self.bias[None, :, None, None, None]
        )
        return affine * weights


class PartialConv3d(nn.Module):
    """Dense fallback that excludes hidden values and reports no sparse savings."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.convolution = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=False,
        )
        self.register_buffer(
            "support_kernel", torch.ones(1, 1, kernel_size, kernel_size, kernel_size)
        )
        self.stride = stride
        self.padding = kernel_size // 2
        self.kernel_volume = kernel_size**3

    def forward(
        self, inputs: torch.Tensor, support: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        support = support.bool()
        output = self.convolution(inputs * support.to(dtype=inputs.dtype))
        counts = F.conv3d(
            support.to(dtype=inputs.dtype),
            self.support_kernel.to(dtype=inputs.dtype),
            stride=self.stride,
            padding=self.padding,
        )
        output_support = counts > 0
        output = output * (self.kernel_volume / counts.clamp_min(1))
        return output * output_support.to(dtype=output.dtype), output_support


class _MaskAwareResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int) -> None:
        super().__init__()
        self.norm1 = MaskAwareNorm3d(in_channels)
        self.conv1 = PartialConv3d(in_channels, out_channels, stride=stride)
        self.norm2 = MaskAwareNorm3d(out_channels)
        self.conv2 = PartialConv3d(out_channels, out_channels)
        self.skip = (
            PartialConv3d(in_channels, out_channels, kernel_size=1, stride=stride)
            if in_channels != out_channels or stride != 1
            else None
        )

    def forward(
        self, inputs: torch.Tensor, support: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, hidden_support = self.conv1(
            F.silu(self.norm1(inputs, support)), support
        )
        hidden, hidden_support = self.conv2(
            F.silu(self.norm2(hidden, hidden_support)), hidden_support
        )
        residual, residual_support = (
            self.skip(inputs, support) if self.skip is not None else (inputs, support)
        )
        output_support = hidden_support | residual_support
        output = (hidden + residual) * output_support.to(dtype=hidden.dtype)
        return output, output_support


class DenseMaskAwareEncoder(nn.Module):
    """Partial-convolution residual encoder used when sparse training is unavailable."""

    executes_sparse_operations = False

    def __init__(
        self,
        *,
        input_channels: int = 4,
        stem_width: int = 16,
        embedding_dim: int = 192,
        local_channels: int | None = None,
    ) -> None:
        super().__init__()
        local_channels = local_channels or stem_width
        widths = (stem_width, stem_width * 2, stem_width * 4, stem_width * 8)
        self.embedding_dim = embedding_dim
        self.stem = PartialConv3d(input_channels, widths[0])
        self.blocks = nn.ModuleList(
            _MaskAwareResidualBlock(
                widths[max(0, index - 1)], width, stride=1 if index == 0 else 2
            )
            for index, width in enumerate(widths)
        )
        self.laterals = nn.ModuleList(
            nn.Conv3d(width, local_channels, kernel_size=1, bias=False)
            for width in widths
        )
        self.projection = nn.Linear(widths[-1], embedding_dim)

    def forward(self, level: VoxelLevel, hidden_mask: torch.Tensor) -> EncoderOutput:
        if hidden_mask.shape != level.validity_mask.shape:
            raise ValueError("hidden mask must match the voxel-level validity mask")
        support = level.validity_mask & ~hidden_mask.bool()
        hidden, support = self.stem(level.features, support)
        features: list[torch.Tensor] = []
        supports: list[torch.Tensor] = []
        for block in self.blocks:
            hidden, support = block(hidden, support)
            features.append(hidden)
            supports.append(support)
        target_size = level.features.shape[2:]
        local = torch.zeros(
            len(level.features),
            self.laterals[0].out_channels,
            *target_size,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        local_support = torch.zeros_like(level.validity_mask)
        for lateral, feature, feature_support in zip(
            self.laterals, features, supports
        ):
            upsampled_support = F.interpolate(
                feature_support.float(), size=target_size, mode="nearest"
            ).bool()
            local = local + F.interpolate(
                lateral(feature),
                size=target_size,
                mode="trilinear",
                align_corners=False,
            ) * upsampled_support
            local_support |= upsampled_support
        denominator = support.flatten(2).sum(dim=2).clamp_min(1)
        pooled = (hidden * support).flatten(2).sum(dim=2) / denominator
        embedding = self.projection(pooled)
        validity_fraction = local_support.float().flatten(1).mean(dim=1, keepdim=True)
        return EncoderOutput(
            global_embedding=embedding,
            scale_embeddings={level.stride: embedding},
            local_feature_volume=F.silu(local) * local_support,
            local_validity_mask=local_support,
            metadata=EncoderMetadata((level.stride,), validity_fraction),
        ).validate(embedding_dim=self.embedding_dim)


def mask_isolation_max_abs(
    encoder: DenseMaskAwareEncoder,
    level: VoxelLevel,
    hidden_mask: torch.Tensor,
    *,
    adversarial_value: float = 1000.0,
) -> float:
    """Change hidden inputs adversarially and compare all exposed encoder features."""

    changed_features = torch.where(
        hidden_mask.expand_as(level.features),
        torch.full_like(level.features, adversarial_value),
        level.features,
    )
    changed_level = VoxelLevel(changed_features, level.validity_mask, level.stride)
    encoder.eval()
    with torch.no_grad():
        first = encoder(level, hidden_mask)
        second = encoder(changed_level, hidden_mask)
    differences = [
        (first.global_embedding - second.global_embedding).abs().max(),
        (first.local_feature_volume - second.local_feature_volume).abs().max(),
    ]
    return float(torch.stack(differences).max())


def hidden_jacobian_max_abs(
    encoder: DenseMaskAwareEncoder, level: VoxelLevel, hidden_mask: torch.Tensor
) -> float:
    """Differentiate exposed outputs with respect to hidden input values."""

    features = level.features.detach().clone().requires_grad_(True)
    output = encoder(VoxelLevel(features, level.validity_mask, level.stride), hidden_mask)
    scalar = output.global_embedding.sum() + output.local_feature_volume.sum()
    gradient = torch.autograd.grad(scalar, features)[0]
    hidden_gradient = gradient.masked_select(hidden_mask.expand_as(gradient))
    return (
        0.0
        if hidden_gradient.numel() == 0
        else float(hidden_gradient.abs().max())
    )


def mask_shortcut_advantage(
    prediction: torch.Tensor, target: torch.Tensor, strata: torch.Tensor
) -> float:
    """Accuracy above a frequency baseline conditioned on mask-sampling stratum."""

    prediction = prediction.reshape(-1)
    target = target.reshape(-1)
    strata = strata.reshape(-1)
    if not (len(prediction) == len(target) == len(strata)) or len(target) == 0:
        raise ValueError("mask shortcut inputs must be non-empty and aligned")
    accuracy = (prediction == target).float().mean()
    baseline_correct = 0
    for stratum in torch.unique(strata):
        counts = torch.bincount(target[strata == stratum].long())
        baseline_correct += int(counts.max())
    return float(accuracy - baseline_correct / len(target))
