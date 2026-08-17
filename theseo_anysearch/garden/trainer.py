"""Garden pre-training loop with early stopping and loss curve tracking."""
from __future__ import annotations

import logging
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader

from theseo_anysearch.garden.data_config import GardenConfig, GardenTrainingConfig
from theseo_anysearch.garden.dataset import GardenDataset, MultiRadiusDataset, RadiusBatchSampler

log = logging.getLogger(__name__)


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


def _metric_improved(value: float, best: float, mode: str) -> bool:
    """Return whether a checkpoint-selection metric improved."""
    return value < best if mode == "min" else value > best


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
    dice_weight: float = 0.0,
    face_loss_weight: float = 0.0,
) -> tuple[float, dict[str, float]]:
    model.eval()
    total, count = 0.0, 0
    intersection = union = true_positive = false_negative = face_correct = face_count = 0
    with torch.no_grad():
        for batch in loader:
            x = batch.to(device)
            if approach in ("vae", "mvmencoder"):
                loss = model.loss(x)
            else:
                loss = model.loss(
                    x,
                    pos_weight=pos_weight,
                    focal_gamma=focal_gamma,
                    dice_weight=dice_weight,
                    face_loss_weight=face_loss_weight,
                )
                reconstruction, _ = model(x)
                target = x.unsqueeze(1) if x.dim() == 4 else x
                predicted = reconstruction >= 0.5
                occupied = target.bool()
                batch_intersection = (predicted & occupied).sum().item()
                intersection += batch_intersection
                union += (predicted | occupied).sum().item()
                true_positive += batch_intersection
                false_negative += (~predicted & occupied).sum().item()
                n = x.shape[-1]
                c = n // 2
                predicted_grid = predicted.squeeze(1)
                target_grid = target.squeeze(1)
                for i, j, k in (
                    (c - 1, c, c), (c + 1, c, c),
                    (c, c - 1, c), (c, c + 1, c),
                    (c, c, c - 1), (c, c, c + 1),
                ):
                    face_correct += (predicted_grid[:, i, j, k] == target_grid[:, i, j, k].bool()).sum().item()
                    face_count += len(x)
            total += loss.item() * len(x)
            count += len(x)
    model.train()
    val_loss = total / max(count, 1)
    metrics = {
        "reconstruction_iou": intersection / max(union, 1),
        "occupied_recall": true_positive / max(true_positive + false_negative, 1),
        "face_accuracy": face_correct / max(face_count, 1),
    }
    metrics["representation_score"] = (
        0.5 * metrics["reconstruction_iou"]
        + 0.3 * metrics["occupied_recall"]
        + 0.2 * metrics["face_accuracy"]
    )
    return val_loss, metrics


def _compute_latent_probe_metrics(
    model: nn.Module,
    train_ds: GardenDataset,
    val_ds: GardenDataset,
    device: str,
    max_samples: int = 2048,
) -> dict[str, float]:
    """Fit a fixed ridge-linear voxel probe and score the frozen latent geometry."""
    train_grids = torch.from_numpy(train_ds._grids[:max_samples]).to(device)  # noqa: SLF001
    val_grids = torch.from_numpy(val_ds._grids[:max_samples]).to(device)  # noqa: SLF001
    n = val_grids.shape[-1]
    rng = np.random.default_rng(44)
    coords = np.indices((n, n, n)).transpose(1, 2, 3, 0)
    ood_array = np.zeros((min(512, max_samples), n, n, n), dtype=np.float32)
    for sample in ood_array:
        center = rng.uniform(1, n - 2, size=3)
        radius = rng.uniform(0.8, max(1.0, n / 2.0))
        sample[np.linalg.norm(coords - center, axis=-1) <= radius] = 1.0
        axis = int(rng.integers(0, 3))
        index = int(rng.integers(0, n))
        slices = [slice(None), slice(None), slice(None)]
        slices[axis] = index
        sample[tuple(slices)] = 1.0 - sample[tuple(slices)]
    ood_grids = torch.from_numpy(ood_array).to(device)

    def encode(grids: torch.Tensor) -> torch.Tensor:
        chunks = []
        for start in range(0, len(grids), 512):
            chunks.append(model.encoder(grids[start:start + 512]))
        return torch.cat(chunks)

    model.eval()
    with torch.no_grad():
        train_latent = encode(train_grids)
        val_latent = encode(val_grids)
        ood_latent = encode(ood_grids)
        train_x = torch.cat([train_latent, torch.ones(len(train_latent), 1, device=device)], dim=1)
        val_x = torch.cat([val_latent, torch.ones(len(val_latent), 1, device=device)], dim=1)
        ood_x = torch.cat([ood_latent, torch.ones(len(ood_latent), 1, device=device)], dim=1)
        train_y = train_grids.flatten(1)
        ridge = 1e-3 * torch.eye(train_x.shape[1], device=device)
        weights = torch.linalg.solve(train_x.T @ train_x + ridge, train_x.T @ train_y)
        predicted = (val_x @ weights) >= 0.5
        target = val_grids.flatten(1).bool()
        intersection = (predicted & target).sum().item()
        union = (predicted | target).sum().item()
        false_negative = (~predicted & target).sum().item()
        ood_predicted = (ood_x @ weights) >= 0.5
        ood_target = ood_grids.flatten(1).bool()
        ood_intersection = (ood_predicted & ood_target).sum().item()
        ood_union = (ood_predicted | ood_target).sum().item()
        ood_false_negative = (~ood_predicted & ood_target).sum().item()

        c = n // 2
        predicted_grid = predicted.reshape(-1, n, n, n)
        face_correct = face_count = 0
        for i, j, k in (
            (c - 1, c, c), (c + 1, c, c),
            (c, c - 1, c), (c, c + 1, c),
            (c, c, c - 1), (c, c, c + 1),
        ):
            face_correct += (predicted_grid[:, i, j, k] == val_grids[:, i, j, k].bool()).sum().item()
            face_count += len(val_grids)
    model.train()
    probe_iou = intersection / max(union, 1)
    probe_recall = intersection / max(intersection + false_negative, 1)
    probe_ood_iou = ood_intersection / max(ood_union, 1)
    probe_ood_recall = ood_intersection / max(ood_intersection + ood_false_negative, 1)
    probe_face_accuracy = face_correct / max(face_count, 1)
    return {
        "probe_iou": probe_iou,
        "probe_recall": probe_recall,
        "probe_ood_iou": probe_ood_iou,
        "probe_ood_recall": probe_ood_recall,
        "probe_face_accuracy": probe_face_accuracy,
        "representation_score": (
            0.25 * probe_iou
            + 0.40 * probe_ood_iou
            + 0.20 * probe_ood_recall
            + 0.15 * probe_face_accuracy
        ),
    }


def train(
    model: nn.Module,
    train_ds: GardenDataset,
    val_ds: GardenDataset,
    cfg: GardenTrainingConfig,
    approach: str,
    device: str = "cpu",
    epoch_cb: Callable[[int, float, float, dict[str, float]], None] | None = None,
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

    optimizer_types: dict[str, type[torch.optim.Optimizer]] = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "rmsprop": torch.optim.RMSprop,
    }
    optimizer = optimizer_types[cfg.optimizer](
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    if cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # Compute pos_weight for weighted BCE / focal loss
    pos_weight: torch.Tensor | None = None
    focal_gamma = cfg.focal_gamma
    dice_weight = cfg.dice_weight
    face_loss_weight = cfg.face_loss_weight
    if approach not in ("vae", "mvmencoder"):
        if cfg.pos_weight_factor > 0.0:
            pos_weight = torch.tensor([cfg.pos_weight_factor], device=device)
        else:
            pos_weight = _compute_pos_weight(train_ds, device=device)
        log.info("pos_weight=%.2f  focal_gamma=%.1f", pos_weight.item(), focal_gamma)

    best_val = float("inf")
    best_metric = float("inf") if cfg.checkpoint_mode == "min" else float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
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
        best_metric = ckpt.get("best_metric", best_val)
        loss_curve = ckpt.get("loss_curve", [])
        best_state = deepcopy(model.state_dict())
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
                loss = model.loss(
                    x,
                    pos_weight=pos_weight,
                    focal_gamma=focal_gamma,
                    dice_weight=dice_weight,
                    face_loss_weight=face_loss_weight,
                )
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(x)
            train_count += len(x)

        if scheduler:
            scheduler.step()

        train_loss = train_total / max(train_count, 1)
        val_loss, val_metrics = _compute_val_loss(
            model,
            val_loader,
            approach,
            device,
            pos_weight=pos_weight,
            focal_gamma=focal_gamma,
            dice_weight=dice_weight,
            face_loss_weight=face_loss_weight,
        )
        if approach == "autoencoder" and isinstance(train_ds, GardenDataset) and isinstance(val_ds, GardenDataset):
            val_metrics.update(_compute_latent_probe_metrics(model, train_ds, val_ds, device))
        loss_curve.append({"epoch": epoch, "train": train_loss, "val": val_loss, **val_metrics})

        # Early stopping
        should_stop = False
        selection_value = val_loss if cfg.checkpoint_metric == "val_loss" else val_metrics[cfg.checkpoint_metric]
        if _metric_improved(selection_value, best_metric, cfg.checkpoint_mode):
            best_metric = selection_value
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            patience_counter = 0
            if checkpoint_path:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val": best_val,
                    "best_metric": best_metric,
                    "loss_curve": loss_curve,
                }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                should_stop = True

        # Report only after a newly improved checkpoint has been persisted.
        if epoch_cb:
            epoch_cb(epoch, train_loss, val_loss, val_metrics)

        if should_stop:
            log.info("Early stopping at epoch %d (patience %d)", epoch, cfg.early_stop_patience)
            if best_state is not None:
                model.load_state_dict(best_state)
            return TrainResult(
                epochs_trained=epoch,
                loss_curve=loss_curve,
                final_val_loss=best_val,
                stopped_early=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    return TrainResult(
        epochs_trained=cfg.epochs,
        loss_curve=loss_curve,
        final_val_loss=best_val,
    )
