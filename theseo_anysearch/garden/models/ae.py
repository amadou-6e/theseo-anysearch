"""Autoencoder and VAE encoder models for voxel pre-training.

Three encoder architectures share the same decoder interface:
  - voxel_box_3dcnn   : Conv3d over full (N,N,N) volume
  - voxel_box_2dcnn   : Conv2d with z-as-channels
  - voxel_triplanar_2dcnn : Conv2d on 3 center slices stacked as channels

After pre-training, discard the decoder and use only the encoder.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class _ResBlock3D(nn.Module):
    """Two Conv3D+BN+ReLU with a residual skip.

    stride=2 halves each spatial dimension for downsampling.
    The skip path uses a 1×1×1 conv+BN when channels or stride differ.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv3d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.body(x) + self.skip(x))


class _ConvBlock2D(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )


class VoxelEncoder3D(nn.Module):
    """3D ResNet encoder: (B, 1, N, N, N) → (B, latent_dim).

    First block: stride=1 (feature extraction at full resolution).
    Remaining blocks: stride=2 (spatial downsampling: 5→3→2→...).
    AdaptiveAvgPool collapses to (B, ch[-1]) regardless of depth.
    """

    def __init__(self, n: int, channels: list[int], latent_dim: int) -> None:
        super().__init__()
        ch = [1] + channels
        strides = [1] + [2] * (len(channels) - 1)
        self.blocks = nn.Sequential(
            *[_ResBlock3D(ch[i], ch[i + 1], strides[i]) for i in range(len(channels))]
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Linear(channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.unsqueeze(1)
        h = self.pool(self.blocks(x)).flatten(1)
        return self.proj(h)


class VoxelEncoder2D(nn.Module):
    """2D CNN with z-as-channels: input (B, N, N, N) → (B, latent_dim)."""

    def __init__(self, n: int, channels: list[int], latent_dim: int) -> None:
        super().__init__()
        # First conv accepts N channels (z levels)
        ch = [n] + channels
        self.convs = nn.Sequential(*[_ConvBlock2D(ch[i], ch[i + 1]) for i in range(len(channels))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, N, N) — treat dim 1 (x-axis) as channel, apply over y-z plane
        # Storage order: x-outer, y-mid, z-inner → permute to (B, z, x, y)
        if x.dim() == 4:
            x = x.permute(0, 3, 1, 2)  # (B, z, x, y)
        h = self.pool(self.convs(x)).flatten(1)
        return self.proj(h)


class VoxelEncoderTriplanar(nn.Module):
    """Tri-planar 2D CNN: 3 center slices → (B, 3, N, N) → (B, latent_dim)."""

    def __init__(self, n: int, channels: list[int], latent_dim: int) -> None:
        super().__init__()
        ch = [3] + channels
        self.convs = nn.Sequential(*[_ConvBlock2D(ch[i], ch[i + 1]) for i in range(len(channels))])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels[-1], latent_dim)
        self._mid = n // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, N, N)
        m = self._mid
        xy = x[:, :, :, m]          # (B, N, N) — z=mid slice
        xz = x[:, :, m, :]          # (B, N, N) — y=mid slice
        yz = x[:, m, :, :]          # (B, N, N) — x=mid slice
        inp = torch.stack([xy, xz, yz], dim=1)   # (B, 3, N, N)
        h = self.pool(self.convs(inp)).flatten(1)
        return self.proj(h)


def build_encoder(architecture: str, n: int, channels: list[int], latent_dim: int) -> nn.Module:
    """Build an encoder module for the requested garden architecture.

    Parameters
    ----------
    architecture : str
        Encoder architecture family name.
    n : int
        Spatial side length of the voxel observation.
    channels : list[int]
        Convolution channel widths.
    latent_dim : int
        Latent embedding size.

    Returns
    -------
    nn.Module
        Encoder module matching the requested architecture.
    """
    if architecture == "voxel_box_3dcnn":
        return VoxelEncoder3D(n, channels, latent_dim)
    if architecture == "voxel_box_2dcnn":
        return VoxelEncoder2D(n, channels, latent_dim)
    if architecture == "voxel_triplanar_2dcnn":
        return VoxelEncoderTriplanar(n, channels, latent_dim)
    raise ValueError(f"Unknown architecture: {architecture!r}")


# ---------------------------------------------------------------------------
# Decoders (discarded after pre-training)
# ---------------------------------------------------------------------------

class _UpsampleBlock3D(nn.Sequential):
    """Trilinear upsample ×2 + Conv3D + BN + ReLU. No ConvTranspose checkerboard."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )


class _DeconvBlock2D(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, activation: bool = True) -> None:
        layers: list[nn.Module] = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=1)]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class VoxelDecoder3D(nn.Module):
    """Progressive upsample decoder: latent → (B, 1, N, N, N).

    Each _UpsampleBlock3D doubles spatial size (1→2→4→...) via trilinear
    interpolation + conv, then a final upsample to exactly N×N×N.

    The target N is inferred at forward time from the optional ``n`` argument,
    so one decoder can reconstruct grids of any resolution.
    """

    def __init__(self, n: int, channels: list[int], latent_dim: int) -> None:
        super().__init__()
        self._n = n
        rev = list(reversed(channels))
        self.proj = nn.Linear(latent_dim, rev[0])
        self.up_blocks = nn.Sequential(
            *[_UpsampleBlock3D(rev[i], rev[i + 1]) for i in range(len(rev) - 1)]
        )
        # final[0] is a no-param Upsample (kept for state-dict compat); final[1] is the conv.
        self.final = nn.Sequential(
            nn.Upsample(size=(n, n, n), mode="trilinear", align_corners=False),
            nn.Conv3d(rev[-1], 1, 3, padding=1),
        )

    def forward(self, z: torch.Tensor, n: int | None = None) -> torch.Tensor:
        h = self.proj(z).reshape(z.size(0), -1, 1, 1, 1)
        h = self.up_blocks(h)
        target_n = n if n is not None else self._n
        h = F.interpolate(h, size=(target_n, target_n, target_n), mode="trilinear", align_corners=False)
        return torch.sigmoid(self.final[1](h))  # (B, 1, N, N, N)


class VoxelDecoder2D(nn.Module):
    """Decodes to 3 planes: (B, 3, N, N) for triplanar, or (B, N, N, N) for z-as-channels."""

    def __init__(self, n: int, channels: list[int], latent_dim: int, out_channels: int) -> None:
        super().__init__()
        self._n = n
        rev = list(reversed(channels))
        self.proj = nn.Linear(latent_dim, rev[0])
        blocks: list[nn.Module] = []
        for i in range(len(rev) - 1):
            blocks.append(_DeconvBlock2D(rev[i], rev[i + 1]))
        blocks.append(_DeconvBlock2D(rev[-1], out_channels, activation=False))
        self.deconvs = nn.Sequential(*blocks)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z).reshape(z.size(0), -1, 1, 1)
        h = F.interpolate(h, size=(self._n, self._n))
        return torch.sigmoid(self.deconvs(h))


# ---------------------------------------------------------------------------
# Autoencoder / VAE
# ---------------------------------------------------------------------------

class VoxelAE(nn.Module):
    """Voxel autoencoder. Supports 3D, 2D, and triplanar architectures."""

    def __init__(
        self,
        architecture: str,
        n: int,
        channels: list[int],
        latent_dim: int,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.encoder = build_encoder(architecture, n, channels, latent_dim)
        if architecture == "voxel_box_3dcnn":
            self.decoder = VoxelDecoder3D(n, channels, latent_dim)
        elif architecture == "voxel_triplanar_2dcnn":
            self.decoder = VoxelDecoder2D(n, channels, latent_dim, out_channels=3)
        else:  # voxel_box_2dcnn
            self.decoder = VoxelDecoder2D(n, channels, latent_dim, out_channels=n)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (reconstruction, z). Decoder n is inferred from x for multi-resolution support."""
        z = self.encoder(x)
        n = x.shape[-1]
        recon = self.decoder(z, n) if isinstance(self.decoder, VoxelDecoder3D) else self.decoder(z)
        return recon, z

    def loss(
        self,
        x: torch.Tensor,
        pos_weight: torch.Tensor | None = None,
        focal_gamma: float = 0.0,
    ) -> torch.Tensor:
        recon, _ = self(x)
        if self.architecture == "voxel_box_3dcnn":
            target = x.unsqueeze(1) if x.dim() == 4 else x
        elif self.architecture == "voxel_triplanar_2dcnn":
            m = x.shape[-1] // 2
            target = torch.stack([x[:, :, :, m], x[:, :, m, :], x[:, m, :, :]], dim=1)
        else:
            target = x.permute(0, 3, 1, 2)
        if focal_gamma > 0.0:
            # Focal loss: weight each voxel by (1 - p_t)^gamma to focus on hard examples
            bce = F.binary_cross_entropy(recon, target, weight=pos_weight, reduction="none")
            p_t = recon * target + (1 - recon) * (1 - target)
            focal_weight = (1.0 - p_t) ** focal_gamma
            return (focal_weight * bce).mean()
        return F.binary_cross_entropy(recon, target, weight=pos_weight)


class VoxelVAE(nn.Module):
    """β-VAE: encoder outputs mu + log_var, KL is added to BCE loss."""

    def __init__(
        self,
        architecture: str,
        n: int,
        channels: list[int],
        latent_dim: int,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.beta = beta
        # Encoder produces 2 × latent_dim (mu + log_var)
        self._base_enc = build_encoder(architecture, n, channels, latent_dim * 2)
        self.latent_dim = latent_dim
        if architecture == "voxel_box_3dcnn":
            self.decoder = VoxelDecoder3D(n, channels, latent_dim)
        elif architecture == "voxel_triplanar_2dcnn":
            self.decoder = VoxelDecoder2D(n, channels, latent_dim, out_channels=3)
        else:
            self.decoder = VoxelDecoder2D(n, channels, latent_dim, out_channels=n)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self._base_enc(x)
        mu, log_var = out.chunk(2, dim=-1)
        return mu, log_var

    def reparameterise(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        return mu + std * torch.randn_like(std)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encode(x)
        z = self.reparameterise(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        recon, mu, log_var = self(x)
        if self.architecture == "voxel_box_3dcnn":
            target = x.unsqueeze(1) if x.dim() == 4 else x
        elif self.architecture == "voxel_triplanar_2dcnn":
            m = x.shape[-1] // 2
            target = torch.stack([x[:, :, :, m], x[:, :, m, :], x[:, m, :, :]], dim=1)
        else:
            target = x.permute(0, 3, 1, 2)
        bce = F.binary_cross_entropy(recon, target)
        kl = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return bce + self.beta * kl
