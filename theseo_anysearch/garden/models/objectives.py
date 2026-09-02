"""Disposable pretraining heads for the P1 T0-T3 recipe bundles."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from theseo_anysearch.garden.models.outputs import EncoderOutput


@dataclass(frozen=True)
class ObjectiveResult:
    loss: torch.Tensor
    components: dict[str, torch.Tensor]
    supervised_values: int


class OccupancyObjective(nn.Module):
    """T0/T1 three-class occupancy reconstruction from local features."""

    def __init__(self, local_channels: int, *, focal_gamma: float = 0.0) -> None:
        super().__init__()
        self.decoder = nn.Conv3d(local_channels, 3, kernel_size=1)
        self.focal_gamma = focal_gamma

    def forward(
        self,
        encoded: EncoderOutput,
        target: torch.Tensor,
        *,
        supervision_mask: torch.Tensor | None = None,
    ) -> ObjectiveResult:
        logits = self.decoder(encoded.local_feature_volume)
        if target.ndim == 5 and target.shape[1] == 1:
            target = target[:, 0]
        if target.shape != logits.shape[:1] + logits.shape[2:]:
            raise ValueError("occupancy target must match decoded spatial dimensions")
        valid = encoded.local_validity_mask[:, 0]
        selected = (
            valid
            if supervision_mask is None
            else valid & supervision_mask.squeeze(1).bool()
        )
        if not selected.any():
            raise ValueError("occupancy objective has no supervised values")
        counts = torch.bincount(target[selected].long(), minlength=3).float().clamp_min(1)
        weights = counts.sum() / (3 * counts)
        per_value = F.cross_entropy(
            logits, target.long(), weight=weights, reduction="none"
        )
        if self.focal_gamma:
            probability = torch.softmax(logits, dim=1).gather(
                1, target.long().unsqueeze(1)
            ).squeeze(1)
            per_value = per_value * (1 - probability).pow(self.focal_gamma)
        loss = per_value[selected].mean()
        return ObjectiveResult(loss, {"occupancy": loss.detach()}, int(selected.sum()))


class ESDFObjective(nn.Module):
    """T2 truncated ESDF regression with optional boundary-balanced weights."""

    def __init__(self, local_channels: int, *, truncation: float) -> None:
        super().__init__()
        if truncation <= 0:
            raise ValueError("ESDF truncation must be positive")
        self.decoder = nn.Conv3d(local_channels, 1, kernel_size=1)
        self.truncation = truncation

    def forward(
        self,
        encoded: EncoderOutput,
        target: torch.Tensor,
        *,
        boundary_mask: torch.Tensor | None = None,
    ) -> ObjectiveResult:
        prediction = self.decoder(encoded.local_feature_volume)[:, 0]
        if target.ndim == 5:
            target = target[:, 0]
        valid = encoded.local_validity_mask[:, 0]
        normalized_target = (
            target.clamp(-self.truncation, self.truncation) / self.truncation
        )
        per_value = F.smooth_l1_loss(
            prediction, normalized_target, reduction="none"
        )
        weights = torch.ones_like(per_value)
        if boundary_mask is not None:
            weights = torch.where(boundary_mask.squeeze(1).bool(), 2.0, 1.0)
        loss = (per_value[valid] * weights[valid]).sum() / weights[valid].sum()
        return ObjectiveResult(loss, {"esdf": loss.detach()}, int(valid.sum()))


class LatentTargetObjective(nn.Module):
    """T3 predicts stop-gradient contextual EMA-teacher local features."""

    def __init__(self, local_channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        hidden_channels = hidden_channels or local_channels * 2
        self.predictor = nn.Sequential(
            nn.Conv3d(local_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv3d(hidden_channels, local_channels, kernel_size=1),
        )

    def forward(
        self,
        online: EncoderOutput,
        teacher: EncoderOutput,
        *,
        supervision_mask: torch.Tensor,
    ) -> ObjectiveResult:
        prediction = self.predictor(online.local_feature_volume)
        target = teacher.local_feature_volume.detach()
        selected = supervision_mask.bool() & teacher.local_validity_mask
        if not selected.any():
            raise ValueError("latent-target objective has no supervised values")
        target_weights = teacher.local_validity_mask.to(dtype=target.dtype)
        count = target_weights.sum(dim=(0, 2, 3, 4), keepdim=True).clamp_min(1)
        mean = (target * target_weights).sum(dim=(0, 2, 3, 4), keepdim=True) / count
        variance = (
            (target - mean).square() * target_weights
        ).sum(dim=(0, 2, 3, 4), keepdim=True) / count
        target = (target - mean) * torch.rsqrt(variance + 1e-5)
        selected_channels = selected.expand_as(prediction)
        loss = F.smooth_l1_loss(
            prediction[selected_channels], target[selected_channels]
        )
        return ObjectiveResult(
            loss, {"latent_target": loss.detach()}, int(selected.sum())
        )


class EMATeacher(nn.Module):
    """Stop-gradient encoder copy updated only by explicit exponential averaging."""

    def __init__(self, encoder: nn.Module, *, decay: float = 0.996) -> None:
        super().__init__()
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be in (0, 1)")
        self.encoder = copy.deepcopy(encoder).eval()
        self.encoder.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, online_encoder: nn.Module) -> None:
        online_state = dict(online_encoder.named_parameters())
        for name, teacher_parameter in self.encoder.named_parameters():
            teacher_parameter.lerp_(online_state[name].detach(), 1 - self.decay)
        online_buffers = dict(online_encoder.named_buffers())
        for name, teacher_buffer in self.encoder.named_buffers():
            teacher_buffer.copy_(online_buffers[name])

    @torch.no_grad()
    def forward(self, *args: object, **kwargs: object) -> EncoderOutput:
        output = self.encoder(*args, **kwargs)
        if not isinstance(output, EncoderOutput):
            raise TypeError("EMA teacher requires an EncoderOutput-compatible encoder")
        return output


def build_pilot_objective(
    bundle: str,
    *,
    local_channels: int,
    truncation: float | None = None,
    focal_gamma: float = 0.0,
) -> nn.Module:
    if bundle in {"T0", "T1"}:
        return OccupancyObjective(local_channels, focal_gamma=focal_gamma)
    if bundle == "T2":
        if truncation is None:
            raise ValueError("T2 requires an ESDF truncation")
        return ESDFObjective(local_channels, truncation=truncation)
    if bundle == "T3":
        return LatentTargetObjective(local_channels)
    raise ValueError(f"unknown pilot objective bundle: {bundle}")
