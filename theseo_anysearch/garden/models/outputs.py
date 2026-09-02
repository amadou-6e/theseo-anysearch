"""Input and output contracts shared by perception-encoder candidates."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class VoxelLevel:
    """One centered semantic voxel level at a declared physical stride."""

    features: torch.Tensor
    validity_mask: torch.Tensor
    stride: int = 1

    def __post_init__(self) -> None:
        if self.features.ndim != 5:
            raise ValueError("voxel features must have shape (B, C, D, H, W)")
        if self.validity_mask.ndim != 5 or self.validity_mask.shape[1] != 1:
            raise ValueError("validity_mask must have shape (B, 1, D, H, W)")
        if self.features.shape[0] != self.validity_mask.shape[0]:
            raise ValueError("features and validity_mask batch dimensions must match")
        if self.features.shape[2:] != self.validity_mask.shape[2:]:
            raise ValueError("features and validity_mask spatial dimensions must match")
        if self.stride < 1 or self.stride & (self.stride - 1):
            raise ValueError("voxel stride must be a positive power of two")
        if self.features.shape[2] != self.features.shape[3] or self.features.shape[2] != self.features.shape[4]:
            raise ValueError("pilot voxel levels must be cubic")

    @classmethod
    def from_occupancy(
        cls,
        occupancy: torch.Tensor,
        *,
        validity_mask: torch.Tensor | None = None,
        unknown_mask: torch.Tensor | None = None,
        stride: int = 1,
    ) -> "VoxelLevel":
        """Expand binary occupancy into occupied/free/unknown/valid channels."""

        if occupancy.ndim == 5:
            if occupancy.shape[1] != 1:
                raise ValueError("5D occupancy must have one channel")
            occupancy = occupancy[:, 0]
        if occupancy.ndim != 4:
            raise ValueError("occupancy must have shape (B, D, H, W)")
        occupied = occupancy > 0.5
        valid = (
            torch.ones_like(occupied, dtype=torch.bool)
            if validity_mask is None
            else validity_mask.squeeze(1).to(dtype=torch.bool)
            if validity_mask.ndim == 5
            else validity_mask.to(dtype=torch.bool)
        )
        unknown = (
            torch.zeros_like(occupied, dtype=torch.bool)
            if unknown_mask is None
            else unknown_mask.squeeze(1).to(dtype=torch.bool)
            if unknown_mask.ndim == 5
            else unknown_mask.to(dtype=torch.bool)
        )
        if valid.shape != occupied.shape or unknown.shape != occupied.shape:
            raise ValueError("occupancy, validity, and unknown masks must share a shape")
        occupied = occupied & valid & ~unknown
        known_free = valid & ~occupied & ~unknown
        features = torch.stack((occupied, known_free, unknown & valid, valid), dim=1).to(
            dtype=occupancy.dtype
        )
        return cls(features=features, validity_mask=valid.unsqueeze(1), stride=stride)

    @property
    def masked_features(self) -> torch.Tensor:
        """Return semantic channels with invalid cells forced to zero."""

        return self.features * self.validity_mask.to(dtype=self.features.dtype)


@dataclass(frozen=True)
class EncoderMetadata:
    """Per-example scale coverage metadata."""

    active_strides: tuple[int, ...]
    validity_fractions: torch.Tensor


@dataclass(frozen=True)
class EncoderOutput:
    """Complete reusable perception-encoder output contract."""

    global_embedding: torch.Tensor
    scale_embeddings: dict[int, torch.Tensor]
    local_feature_volume: torch.Tensor
    local_validity_mask: torch.Tensor
    metadata: EncoderMetadata

    def validate(self, *, embedding_dim: int | None = None) -> "EncoderOutput":
        """Raise on a malformed output and return self for convenient chaining."""

        if self.global_embedding.ndim != 2:
            raise ValueError("global_embedding must have shape (B, E)")
        batch = self.global_embedding.shape[0]
        if embedding_dim is not None and self.global_embedding.shape[1] != embedding_dim:
            raise ValueError(f"expected embedding dimension {embedding_dim}")
        if not self.scale_embeddings:
            raise ValueError("at least one scale embedding is required")
        if tuple(sorted(self.scale_embeddings)) != self.metadata.active_strides:
            raise ValueError("scale embeddings and active-stride metadata disagree")
        for stride, embedding in self.scale_embeddings.items():
            if stride < 1 or stride & (stride - 1):
                raise ValueError("scale strides must be positive powers of two")
            if embedding.shape != self.global_embedding.shape:
                raise ValueError("every scale embedding must match the global shape")
        if self.local_feature_volume.ndim != 5:
            raise ValueError("local_feature_volume must have shape (B, C, D, H, W)")
        if self.local_validity_mask.shape != (
            batch,
            1,
            *self.local_feature_volume.shape[2:],
        ):
            raise ValueError("local validity mask must match the local feature volume")
        if self.local_validity_mask.dtype != torch.bool:
            raise ValueError("local validity mask must be boolean")
        if self.metadata.validity_fractions.shape != (batch, len(self.scale_embeddings)):
            raise ValueError("validity fractions must have shape (B, number_of_scales)")
        tensors = [
            self.global_embedding,
            self.local_feature_volume,
            self.metadata.validity_fractions,
            *self.scale_embeddings.values(),
        ]
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise ValueError("encoder output contains non-finite values")
        return self
