"""Garden pre-training loop with early stopping and loss curve tracking."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from theseo_anysearch.garden.data_config import GardenConfig, GardenTrainingConfig
from theseo_anysearch.garden.dataset import GardenDataset, MultiRadiusDataset, RadiusBatchSampler

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class UpdateTrainingConfig:
    """Optimizer contract counted in updates rather than dataset epochs."""

    total_updates: int
    peak_learning_rate: float
    weight_decay: float = 0.05
    warmup_fraction: float = 0.05
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1

    def __post_init__(self) -> None:
        if self.total_updates < 1 or self.peak_learning_rate <= 0:
            raise ValueError("update count and peak learning rate must be positive")
        if self.weight_decay < 0 or not 0 <= self.warmup_fraction < 1:
            raise ValueError("invalid weight decay or warmup fraction")
        if self.gradient_clip_norm <= 0 or self.accumulation_steps < 1:
            raise ValueError("gradient clipping and accumulation must be positive")


@dataclass(frozen=True)
class TrainingResourceReport:
    updates: int
    observations: int
    encoded_views: int
    valid_voxels: int
    wall_seconds: float
    trainable_parameters: int
    checkpoint_bytes: int
    dense_masked_voxels_skipped: int = 0


class UpdateTrainer:
    """AdamW warmup/cosine runtime with honest observation and voxel accounting."""

    def __init__(self, module: nn.Module, config: UpdateTrainingConfig) -> None:
        self.module = module
        self.config = config
        parameters = [
            parameter for parameter in module.parameters() if parameter.requires_grad
        ]
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=config.peak_learning_rate,
            weight_decay=config.weight_decay,
        )
        warmup = max(1, round(config.total_updates * config.warmup_fraction))

        def multiplier(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, config.total_updates - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, multiplier
        )
        self._micro_steps = 0
        self._updates = 0
        self._observations = 0
        self._views = 0
        self._valid_voxels = 0
        self._started = time.perf_counter()
        self.optimizer.zero_grad(set_to_none=True)

    @property
    def updates(self) -> int:
        return self._updates

    def step(
        self,
        loss: torch.Tensor,
        *,
        observations: int,
        encoded_views: int,
        valid_voxels: int,
    ) -> bool:
        """Accumulate one micro-batch and report whether an optimizer update occurred."""

        if self._updates >= self.config.total_updates:
            raise RuntimeError("preregistered update budget is exhausted")
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss is non-finite")
        if min(observations, encoded_views, valid_voxels) < 0:
            raise ValueError("resource counts cannot be negative")
        (loss / self.config.accumulation_steps).backward()
        self._micro_steps += 1
        self._observations += observations
        self._views += encoded_views
        self._valid_voxels += valid_voxels
        if self._micro_steps % self.config.accumulation_steps:
            return False
        torch.nn.utils.clip_grad_norm_(
            self.module.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()
        self._updates += 1
        return True

    def report(self) -> TrainingResourceReport:
        state = self.module.state_dict()
        checkpoint_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in state.values()
        )
        return TrainingResourceReport(
            updates=self._updates,
            observations=self._observations,
            encoded_views=self._views,
            valid_voxels=self._valid_voxels,
            wall_seconds=time.perf_counter() - self._started,
            trainable_parameters=sum(
                parameter.numel()
                for parameter in self.module.parameters()
                if parameter.requires_grad
            ),
            checkpoint_bytes=checkpoint_bytes,
            dense_masked_voxels_skipped=0,
        )


class TrainResult(BaseModel):
    """Summary of one completed garden training run.

    Parameters
    ----------
    epochs_trained : int
        Number of epochs completed.
    loss_curve : list[dict]
        Recorded train and validation loss values per epoch.
    final_val_loss : float
        Best validation loss observed during training.
    stopped_early : bool
        Whether training ended via early stopping.
    """
    epochs_trained: int
    loss_curve: list[dict] = Field(default_factory=list)
    final_val_loss: float = float("inf")
    stopped_early: bool = False


def _compute_pos_weight(train_ds: GardenDataset, n_samples: int = 512, device: str = "cpu") -> torch.Tensor:
    """Estimate fill rate from a sample of the dataset and return pos_weight tensor."""
    indices = list(range(min(n_samples, len(train_ds))))
    grids = torch.stack([train_ds[i] for i in indices])
    fill_rate = grids.mean().item()
    fill_rate = max(fill_rate, 1e-3)
    weight = (1.0 - fill_rate) / fill_rate
    return torch.tensor([weight], device=device)


def _compute_val_loss(
    model: nn.Module,
    loader: DataLoader,
    approach: str,
    device: str,
    pos_weight: torch.Tensor | None = None,
    focal_gamma: float = 0.0,
) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            if approach in ("vae", "mvmencoder"):
                loss = model.loss(x)
            else:
                loss = model.loss(x, pos_weight=pos_weight, focal_gamma=focal_gamma)
            total += loss.item() * len(x)
            count += len(x)
    model.train()
    return total / max(count, 1)


def train(
    model: nn.Module,
    train_ds: GardenDataset,
    val_ds: GardenDataset,
    cfg: GardenTrainingConfig,
    approach: str,
    device: str = "cpu",
    epoch_cb: Callable[[int, float, float], None] | None = None,
    checkpoint_path: Path | None = None,
) -> TrainResult:
    """Run the pre-training loop. epoch_cb(epoch, train_loss, val_loss) called each epoch."""
    model = model.to(device)
    model.train()

    if isinstance(train_ds, MultiRadiusDataset):
        train_loader = DataLoader(
            train_ds,
            batch_sampler=RadiusBatchSampler(train_ds, batch_size=cfg.batch_size, shuffle=True),
            num_workers=0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_sampler=RadiusBatchSampler(val_ds, batch_size=cfg.batch_size, shuffle=False),  # type: ignore[arg-type]
            num_workers=0,
        )
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Compute pos_weight for weighted BCE / focal loss
    pos_weight: torch.Tensor | None = None
    focal_gamma = cfg.focal_gamma
    if approach not in ("vae", "mvmencoder"):
        if cfg.pos_weight_factor > 0.0:
            pos_weight = torch.tensor([cfg.pos_weight_factor], device=device)
        else:
            pos_weight = _compute_pos_weight(train_ds, device=device)
        log.info("pos_weight=%.2f  focal_gamma=%.1f", pos_weight.item(), focal_gamma)

    best_val = float("inf")
    patience_counter = 0
    loss_curve: list[dict] = []

    # Resume from checkpoint if provided
    start_epoch = 1
    if checkpoint_path and checkpoint_path.exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
        loss_curve = ckpt.get("loss_curve", [])
        log.info("Resumed from epoch %d", start_epoch - 1)

    for epoch in range(start_epoch, cfg.epochs + 1):
        # Train
        train_total, train_count = 0.0, 0
        for batch in train_loader:
            x = batch.to(device)
            optimizer.zero_grad()
            if approach in ("vae", "mvmencoder"):
                loss = model.loss(x)
            else:
                loss = model.loss(x, pos_weight=pos_weight, focal_gamma=focal_gamma)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(x)
            train_count += len(x)

        if scheduler:
            scheduler.step()

        train_loss = train_total / max(train_count, 1)
        val_loss = _compute_val_loss(model, val_loader, approach, device, pos_weight=pos_weight, focal_gamma=focal_gamma)
        loss_curve.append({"epoch": epoch, "train": train_loss, "val": val_loss})

        if epoch_cb:
            epoch_cb(epoch, train_loss, val_loss)

        # Early stopping
        if val_loss < best_val:
            best_val = val_loss
            patience_counter = 0
            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val,
                    "loss_curve": loss_curve,
                }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                log.info("Early stopping at epoch %d (patience %d)", epoch, cfg.early_stop_patience)
                return TrainResult(
                    epochs_trained=epoch,
                    loss_curve=loss_curve,
                    final_val_loss=best_val,
                    stopped_early=True,
                )

    return TrainResult(
        epochs_trained=cfg.epochs,
        loss_curve=loss_curve,
        final_val_loss=best_val,
    )
