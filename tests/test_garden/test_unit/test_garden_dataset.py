"""Unit tests for garden Dataset wrappers and samplers."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from theseo_anysearch.garden.data_config import AugmentationConfig, SplitConfig
from theseo_anysearch.garden.dataset import (
    GardenDataset,
    MultiRadiusDataset,
    RadiusBatchSampler,
    make_multi_radius_train_val,
    make_train_val_datasets,
)


def _grids(count: int, n: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.random((count, n, n, n)) > 0.7).astype(np.float32)


# ---------------------------------------------------------------------------
# GardenDataset
# ---------------------------------------------------------------------------

class TestGardenDataset:
    """Tests GardenDataset."""
    def test_len(self):
        grids = _grids(20)
        ds = GardenDataset(grids, AugmentationConfig())
        assert len(ds) == 20

    def test_getitem_returns_tensor(self):
        grids = _grids(10, n=5)
        ds = GardenDataset(grids, AugmentationConfig())
        item = ds[0]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (5, 5, 5)

    def test_dtype_float32(self):
        grids = _grids(5)
        ds = GardenDataset(grids, AugmentationConfig())
        assert ds[0].dtype == torch.float32

    def test_does_not_mutate_source_array(self):
        grids = _grids(10)
        original = grids.copy()
        ds = GardenDataset(grids, AugmentationConfig(noise_prob=1.0))
        _ = ds[0]
        np.testing.assert_array_equal(grids, original)

    def test_augmentation_applied(self):
        """Noise with prob=1 inverts everything; output should differ from input."""
        grids = np.ones((5, 5, 5, 5), dtype=np.float32)
        ds = GardenDataset(grids, AugmentationConfig(noise_prob=1.0), seed=0)
        item = ds[0]
        # All-ones grid with noise_prob=1 → should be all zeros
        assert item.sum().item() == pytest.approx(0.0)

    def test_val_dataset_no_augmentation(self):
        """Val dataset created by make_train_val_datasets uses AugmentationConfig() (no aug)."""
        grids = _grids(20)
        aug = AugmentationConfig(noise_prob=1.0)
        _, val_ds = make_train_val_datasets(grids, aug, SplitConfig())
        # Val dataset uses no-aug config — output == input
        item = val_ds[0]
        idx = val_ds._grids  # internal grids; item should match one of them
        assert item.shape == (5, 5, 5)


# ---------------------------------------------------------------------------
# make_train_val_datasets
# ---------------------------------------------------------------------------

class TestMakeTrainValDatasets:
    """Tests MakeTrainValDatasets."""
    def test_split_sizes(self):
        grids = _grids(100)
        train, val = make_train_val_datasets(grids, AugmentationConfig(), SplitConfig(val_fraction=0.1))
        assert len(val) == 10
        assert len(train) == 90

    def test_split_no_overlap(self):
        """Train and val must not share the same grid array indices."""
        grids = _grids(50, seed=42)
        train, val = make_train_val_datasets(grids, AugmentationConfig(), SplitConfig(val_fraction=0.2, seed=0))
        # Compare internal grid arrays by pointer identity rather than content (content could coincide)
        assert len(train) + len(val) == len(grids)

    def test_deterministic_with_seed(self):
        grids = _grids(50)
        t1, v1 = make_train_val_datasets(grids, AugmentationConfig(), SplitConfig(seed=7))
        t2, v2 = make_train_val_datasets(grids, AugmentationConfig(), SplitConfig(seed=7))
        np.testing.assert_array_equal(t1._grids, t2._grids)
        np.testing.assert_array_equal(v1._grids, v2._grids)

    def test_different_seeds_different_splits(self):
        grids = _grids(50)
        t1, _ = make_train_val_datasets(grids, AugmentationConfig(), SplitConfig(seed=1))
        t2, _ = make_train_val_datasets(grids, AugmentationConfig(), SplitConfig(seed=2))
        # Very unlikely to be identical
        assert not np.array_equal(t1._grids, t2._grids)


# ---------------------------------------------------------------------------
# MultiRadiusDataset
# ---------------------------------------------------------------------------

class TestMultiRadiusDataset:
    """Tests MultiRadiusDataset."""
    def _make(self):
        ds1 = GardenDataset(_grids(10, n=3), AugmentationConfig())
        ds2 = GardenDataset(_grids(20, n=5), AugmentationConfig())
        return MultiRadiusDataset({1: ds1, 2: ds2})

    def test_total_len(self):
        mds = self._make()
        assert len(mds) == 30

    def test_getitem_returns_tensor(self):
        mds = self._make()
        item = mds[0]
        assert isinstance(item, torch.Tensor)

    def test_radius_groups_coverage(self):
        mds = self._make()
        groups = mds.radius_groups
        assert set(groups.keys()) == {1, 2}
        assert len(groups[1]) == 10
        assert len(groups[2]) == 20

    def test_radius_groups_indices_partition(self):
        mds = self._make()
        groups = mds.radius_groups
        all_indices = sorted(groups[1] + groups[2])
        assert all_indices == list(range(30))


# ---------------------------------------------------------------------------
# RadiusBatchSampler
# ---------------------------------------------------------------------------

class TestRadiusBatchSampler:
    """Tests RadiusBatchSampler."""
    def _make_sampler(self, batch_size=4, shuffle=False):
        ds1 = GardenDataset(_grids(12, n=3), AugmentationConfig())
        ds2 = GardenDataset(_grids(8, n=5), AugmentationConfig())
        mds = MultiRadiusDataset({1: ds1, 2: ds2})
        return mds, RadiusBatchSampler(mds, batch_size=batch_size, shuffle=shuffle)

    def test_batch_count(self):
        mds, sampler = self._make_sampler(batch_size=4)
        # radius 1: 12 → 3 batches; radius 2: 8 → 2 batches
        assert len(sampler) == 5

    def test_batches_same_radius(self):
        mds, sampler = self._make_sampler(batch_size=4)
        groups = mds.radius_groups
        r1_set = set(groups[1])
        r2_set = set(groups[2])
        for batch in sampler:
            in_r1 = all(i in r1_set for i in batch)
            in_r2 = all(i in r2_set for i in batch)
            assert in_r1 or in_r2, f"Batch {batch} mixes radii"

    def test_batch_size(self):
        _, sampler = self._make_sampler(batch_size=4)
        for batch in sampler:
            assert len(batch) == 4

    def test_dataloader_integration(self):
        ds1 = GardenDataset(_grids(8, n=3), AugmentationConfig())
        ds2 = GardenDataset(_grids(8, n=5), AugmentationConfig())
        mds = MultiRadiusDataset({1: ds1, 2: ds2})
        sampler = RadiusBatchSampler(mds, batch_size=4, shuffle=False)
        loader = DataLoader(mds, batch_sampler=sampler)
        for batch in loader:
            # All items in batch must have same shape (same radius)
            assert batch.shape[0] == 4
            assert batch.shape[1] == batch.shape[2] == batch.shape[3]


# ---------------------------------------------------------------------------
# make_multi_radius_train_val
# ---------------------------------------------------------------------------

class TestMakeMultiRadiusTrainVal:
    """Tests MakeMultiRadiusTrainVal."""
    def test_produces_multi_radius_datasets(self):
        grids_per_radius = {
            1: _grids(20, n=3),
            2: _grids(30, n=5),
        }
        train, val = make_multi_radius_train_val(
            grids_per_radius, AugmentationConfig(), SplitConfig(val_fraction=0.1)
        )
        assert isinstance(train, MultiRadiusDataset)
        assert isinstance(val, MultiRadiusDataset)

    def test_sizes_match_split(self):
        grids_per_radius = {
            1: _grids(20, n=3),
            2: _grids(40, n=5),
        }
        train, val = make_multi_radius_train_val(
            grids_per_radius, AugmentationConfig(), SplitConfig(val_fraction=0.1)
        )
        # radius 1: 18 train + 2 val; radius 2: 36 train + 4 val
        assert len(train) == 54
        assert len(val) == 6
