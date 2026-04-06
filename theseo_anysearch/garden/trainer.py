"""Garden pre-training loop with early stopping and loss curve tracking."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from theseo_anysearch.garden.data_config import GardenConfig, GardenTrainingConfig
from theseo_anysearch.garden.dataset import GardenDataset, MultiRadiusDataset, RadiusBatchSampler

log = logging.getLogger(__name__)


@dataclass
class TrainResult:
    epochs_trained: int
    loss_curve: list[dict] = field(default_factory=list)
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
