"""Masked Voxel Modeling encoder for voxel pre-training.

The encoder (3D CNN) receives a grid where 75% of cells are replaced with a
learnable mask token. A lightweight transformer decoder predicts all N³ cell
values. The decoder is discarded after pre-training.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from theseo_anysearch.garden.models.ae import build_encoder


class VoxelMVMEncoder(nn.Module):
    """Masked Voxel Modeling: encoder + transformer decoder."""

    def __init__(
        self,
        architecture: str,
        n: int,
        channels: list[int],
        latent_dim: int,
        mask_ratio: float = 0.75,
        decoder_layers: int = 2,
        decoder_dim: int = 64,
        decoder_heads: int = 4,
    ) -> None:
        super().__init__()
        self.n = n
        self.n3 = n ** 3
        self.mask_ratio = mask_ratio
        self.architecture = architecture

        # Learnable mask token value (scalar substituted at masked positions)
        self.mask_token = nn.Parameter(torch.tensor(0.5))

        self.encoder = build_encoder(architecture, n, channels, latent_dim)

        # Positional embedding: one vector per cell in N³
        self.pos_embed = nn.Embedding(self.n3, decoder_dim)

        # Project latent to decoder dim
        self.latent_proj = nn.Linear(latent_dim, decoder_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.head = nn.Linear(decoder_dim, 1)

    def _apply_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace mask_ratio of cells with mask_token. Returns (masked_grid, mask_bool)."""
        B, N, _, _ = x.shape
        flat = x.reshape(B, -1)  # (B, N³)
        num_masked = int(self.n3 * self.mask_ratio)
        noise = torch.rand(B, self.n3, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, self.n3, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_shuffle[:, :num_masked], True)
        flat_masked = flat.clone()
        flat_masked[mask] = self.mask_token
        return flat_masked.reshape(B, N, N, N), mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (predictions (B, N³), mask (B, N³))."""
        masked, mask = self._apply_mask(x)
        z = self.encoder(masked)                              # (B, latent_dim)

        # Decoder: attend from all N³ positions to the latent vector
        B = z.size(0)
        pos_idx = torch.arange(self.n3, device=z.device).unsqueeze(0).expand(B, -1)
        queries = self.pos_embed(pos_idx)                     # (B, N³, decoder_dim)
        memory = self.latent_proj(z).unsqueeze(1)             # (B, 1, decoder_dim)
        out = self.transformer(queries, memory)               # (B, N³, decoder_dim)
        preds = torch.sigmoid(self.head(out)).squeeze(-1)     # (B, N³)
        return preds, mask

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        """BCE loss on masked cells only."""
        preds, mask = self(x)
        target = x.reshape(x.size(0), -1)                    # (B, N³)
        return F.binary_cross_entropy(preds[mask], target[mask])
