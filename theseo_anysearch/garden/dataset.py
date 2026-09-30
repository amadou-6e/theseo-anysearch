"""GardenDataset: torch Dataset wrapping collected voxel grids with augmentation."""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from theseo_anysearch.garden.augment import augment, augment_pair
from theseo_anysearch.garden.data_config import AugmentationConfig, SplitConfig


class GardenDataset(Dataset):
    """Binary (N,N,N) voxel grids with online augmentation."""

    def __init__(
        self,
        grids: np.ndarray,
        aug_cfg: AugmentationConfig,
        seed: int = 42,
    ) -> None:
        self._grids = grids.astype(np.float32)
        self._aug_cfg = aug_cfg
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._grids)

    def __getitem__(self, idx: int) -> torch.Tensor:
        grid = self._grids[idx].copy()
        grid = augment(grid, self._aug_cfg, self._rng)
        return torch.from_numpy(grid)


def make_train_val_datasets(
    grids: np.ndarray,
    aug_cfg: AugmentationConfig,
    split_cfg: SplitConfig,
) -> tuple[GardenDataset, GardenDataset]:
    """Split voxel grids into train and validation datasets.

    Parameters
    ----------
    grids : np.ndarray
        Collected voxel grids with shape ``(N, n, n, n)``.
    aug_cfg : AugmentationConfig
        Online augmentation settings applied to the training split.
    split_cfg : SplitConfig
        Random split configuration.

    Returns
    -------
    tuple[GardenDataset, GardenDataset]
        Training and validation datasets.
    """
    rng = np.random.default_rng(split_cfg.seed)
    idx = rng.permutation(len(grids))
    n_val = max(1, int(len(grids) * split_cfg.val_fraction))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_ds = GardenDataset(grids[train_idx], aug_cfg, seed=split_cfg.seed)
    val_ds = GardenDataset(grids[val_idx], AugmentationConfig(), seed=split_cfg.seed + 1)
    return train_ds, val_ds


class MultiRadiusDataset(Dataset):
    """Wraps per-radius GardenDatasets into one Dataset.

    Each item is a (n, n, n) tensor; RadiusBatchSampler ensures all items in a
    batch share the same radius so tensors can be stacked by the default collate.
    """

    def __init__(self, datasets: dict[int, GardenDataset]) -> None:
        self._datasets = datasets
        # Build global index: list of (radius, local_idx)
        self._index: list[tuple[int, int]] = []
        for r, ds in datasets.items():
            for i in range(len(ds)):
                self._index.append((r, i))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        r, i = self._index[idx]
        return self._datasets[r][i]

    @property
    def radius_groups(self) -> dict[int, list[int]]:
        """Map radius → list of global indices."""
        groups: dict[int, list[int]] = {}
        for gi, (r, _) in enumerate(self._index):
            groups.setdefault(r, []).append(gi)
        return groups


class RadiusBatchSampler(Sampler[list[int]]):
    """Yields batches where every index shares the same radius."""

    def __init__(self, dataset: MultiRadiusDataset, batch_size: int, shuffle: bool = True) -> None:
        self._groups = dataset.radius_groups
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __iter__(self):
        all_batches: list[list[int]] = []
        for indices in self._groups.values():
            idx = list(indices)
            if self._shuffle:
                random.shuffle(idx)
            for i in range(0, len(idx) - len(idx) % self._batch_size, self._batch_size):
                all_batches.append(idx[i:i + self._batch_size])
        if self._shuffle:
            random.shuffle(all_batches)
        yield from all_batches

    def __len__(self) -> int:
        return sum(len(g) // self._batch_size for g in self._groups.values())


def make_multi_radius_train_val(
    grids_per_radius: dict[int, np.ndarray],
    aug_cfg: AugmentationConfig,
    split_cfg: SplitConfig,
) -> tuple[MultiRadiusDataset, MultiRadiusDataset]:
    """Build train/val MultiRadiusDatasets from per-radius grids."""
    train_dsets: dict[int, GardenDataset] = {}
    val_dsets: dict[int, GardenDataset] = {}
    for r, grids in grids_per_radius.items():
        train_ds, val_ds = make_train_val_datasets(grids, aug_cfg, split_cfg)
        train_dsets[r] = train_ds
        val_dsets[r] = val_ds
    return MultiRadiusDataset(train_dsets), MultiRadiusDataset(val_dsets)
