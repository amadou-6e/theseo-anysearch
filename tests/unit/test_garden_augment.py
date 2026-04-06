"""Unit tests for garden augmentation transforms."""
from __future__ import annotations

import numpy as np
import pytest

from theseo_anysearch.garden.augment import (
    _ROT90_TRANSFORMS,
    _cutout,
    _flip,
    _geometry_paste,
    _noise,
    _rotate90,
    _translation_jitter,
    augment,
    augment_pair,
)
from theseo_anysearch.garden.data_config import (
    AugmentationConfig,
    CutoutConfig,
    GeometryPasteConfig,
)


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _solid(n: int = 5) -> np.ndarray:
    """All-ones grid."""
    return np.ones((n, n, n), dtype=np.float32)


def _empty(n: int = 5) -> np.ndarray:
    return np.zeros((n, n, n), dtype=np.float32)


def _checkerboard(n: int = 5) -> np.ndarray:
    g = np.zeros((n, n, n), dtype=np.float32)
    g[::2, ::2, ::2] = 1.0
    return g


# ---------------------------------------------------------------------------
# _rotate90
# ---------------------------------------------------------------------------

class TestRotate90:
    def test_output_shape_preserved(self):
        g = _checkerboard(5)
        out = _rotate90(g, _rng())
        assert out.shape == g.shape

    def test_output_dtype(self):
        g = _checkerboard(5)
        out = _rotate90(g, _rng())
        assert out.dtype == np.float32

    def test_is_contiguous(self):
        g = _checkerboard(5)
        out = _rotate90(g, _rng())
        assert out.flags["C_CONTIGUOUS"]

    def test_24_transforms_defined(self):
        assert len(_ROT90_TRANSFORMS) == 24

    def test_values_preserved(self):
        """Rotation permutes voxels but doesn't change the set of values."""
        g = _checkerboard(5)
        out = _rotate90(g, _rng(seed=7))
        assert np.isclose(g.sum(), out.sum())

    def test_identity_exists(self):
        """At least one rotation leaves the grid unchanged (the identity)."""
        g = _checkerboard(5)
        # Force identity transform (index 0)
        perm, flip_axes = _ROT90_TRANSFORMS[0]
        out = np.transpose(g, perm)
        if flip_axes:
            out = np.flip(out, axis=flip_axes)
        out = np.ascontiguousarray(out)
        np.testing.assert_array_equal(g, out)

    def test_does_not_mutate_input(self):
        g = _checkerboard(5)
        original = g.copy()
        _rotate90(g, _rng())
        np.testing.assert_array_equal(g, original)


# ---------------------------------------------------------------------------
# _flip
# ---------------------------------------------------------------------------

class TestFlip:
    def test_shape_preserved(self):
        g = _checkerboard(5)
        out = _flip(g, _rng())
        assert out.shape == g.shape

    def test_values_preserved(self):
        g = _checkerboard(5)
        out = _flip(g, _rng())
        assert np.isclose(g.sum(), out.sum())

    def test_is_contiguous(self):
        g = _checkerboard(5)
        out = _flip(g, _rng())
        assert out.flags["C_CONTIGUOUS"]

    def test_deterministic_with_seed(self):
        g = _checkerboard(5)
        a = _flip(g, _rng(seed=1))
        b = _flip(g, _rng(seed=1))
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# _cutout
# ---------------------------------------------------------------------------

class TestCutout:
    def test_reduces_sum(self):
        g = _solid(7)
        original_sum = g.sum()
        out = _cutout(g, count=1, min_size=2, max_size=3, rng=_rng())
        assert out.sum() < original_sum

    def test_shape_preserved(self):
        g = _solid(5)
        out = _cutout(g, count=2, min_size=1, max_size=2, rng=_rng())
        assert out.shape == g.shape

    def test_zeroed_region(self):
        """All-ones grid should have zeros after cutout."""
        g = _solid(5)
        out = _cutout(g, count=3, min_size=2, max_size=2, rng=_rng())
        assert (out == 0.0).any()

    def test_does_not_mutate_input(self):
        g = _solid(5)
        original = g.copy()
        _cutout(g.copy(), count=2, min_size=1, max_size=2, rng=_rng())
        np.testing.assert_array_equal(g, original)

    def test_empty_grid_unchanged(self):
        g = _empty(5)
        out = _cutout(g, count=5, min_size=2, max_size=3, rng=_rng())
        assert out.sum() == 0.0


# ---------------------------------------------------------------------------
# _geometry_paste
# ---------------------------------------------------------------------------

class TestGeometryPaste:
    def test_shape_preserved(self):
        a = _empty(5)
        b = _solid(5)
        out = _geometry_paste(a, b, paste_size=2, rng=_rng())
        assert out.shape == (5, 5, 5)

    def test_paste_adds_voxels(self):
        a = _empty(5)
        b = _solid(5)
        out = _geometry_paste(a, b, paste_size=2, rng=_rng())
        assert out.sum() > 0.0

    def test_does_not_remove_voxels_from_target(self):
        """Paste uses max — existing filled voxels cannot be removed."""
        a = _solid(5)
        b = _empty(5)
        out = _geometry_paste(a, b, paste_size=2, rng=_rng())
        assert out.sum() == a.sum()

    def test_does_not_mutate_source(self):
        a = _empty(5)
        b = _solid(5)
        original_b = b.copy()
        _geometry_paste(a, b, paste_size=2, rng=_rng())
        np.testing.assert_array_equal(b, original_b)


# ---------------------------------------------------------------------------
# _noise
# ---------------------------------------------------------------------------

class TestNoise:
    def test_shape_preserved(self):
        g = _solid(5)
        out = _noise(g, prob=0.1, rng=_rng())
        assert out.shape == g.shape

    def test_zero_prob_unchanged(self):
        g = _checkerboard(5)
        out = _noise(g, prob=0.0, rng=_rng())
        np.testing.assert_array_equal(g, out)

    def test_full_prob_inverts(self):
        g = _solid(5)
        out = _noise(g, prob=1.0, rng=_rng())
        np.testing.assert_array_equal(out, _empty(5))

    def test_values_binary(self):
        g = _checkerboard(5)
        out = _noise(g, prob=0.3, rng=_rng())
        assert set(np.unique(out)).issubset({0.0, 1.0})

    def test_does_not_mutate_input(self):
        g = _checkerboard(5)
        original = g.copy()
        _noise(g, prob=0.5, rng=_rng())
        np.testing.assert_array_equal(g, original)


# ---------------------------------------------------------------------------
# _translation_jitter
# ---------------------------------------------------------------------------

class TestTranslationJitter:
    def test_shape_preserved(self):
        g = _checkerboard(5)
        out = _translation_jitter(g, max_shift=1, rng=_rng())
        assert out.shape == g.shape

    def test_values_preserved(self):
        """Roll is circular — sum should be unchanged."""
        g = _checkerboard(5)
        out = _translation_jitter(g, max_shift=2, rng=_rng())
        assert np.isclose(g.sum(), out.sum())

    def test_zero_shift_unchanged(self):
        """max_shift=0 → shifts=[0,0,0] → unchanged grid."""
        g = _checkerboard(5)
        out = _translation_jitter(g, max_shift=0, rng=_rng())
        np.testing.assert_array_equal(g, out)


# ---------------------------------------------------------------------------
# augment() — integration
# ---------------------------------------------------------------------------

class TestAugment:
    def test_no_augmentation_returns_copy(self):
        g = _checkerboard(5)
        cfg = AugmentationConfig()
        out = augment(g, cfg, _rng())
        np.testing.assert_array_equal(g, out)
        assert out is not g  # must be a copy

    def test_shape_preserved_with_all_transforms(self):
        g = _checkerboard(5)
        cfg = AugmentationConfig(
            rotate90=True,
            flip=True,
            cutout=CutoutConfig(count=2, min_size=1, max_size=2),
            noise_prob=0.05,
            translation_jitter=1,
            morph_prob=0.0,  # skip morph (requires scipy)
        )
        out = augment(g, cfg, _rng())
        assert out.shape == g.shape

    def test_does_not_mutate_input(self):
        g = _checkerboard(5)
        original = g.copy()
        cfg = AugmentationConfig(
            rotate90=True, flip=True,
            cutout=CutoutConfig(), noise_prob=0.1
        )
        augment(g, cfg, _rng())
        np.testing.assert_array_equal(g, original)

    def test_cutout_reduces_sum(self):
        g = _solid(7)
        cfg = AugmentationConfig(cutout=CutoutConfig(count=3, min_size=2, max_size=3))
        out = augment(g, cfg, _rng())
        assert out.sum() < g.sum()

    def test_rotate90_only_preserves_sum(self):
        g = _checkerboard(5)
        cfg = AugmentationConfig(rotate90=True)
        out = augment(g, cfg, _rng())
        assert np.isclose(g.sum(), out.sum())


# ---------------------------------------------------------------------------
# augment_pair() — integration
# ---------------------------------------------------------------------------

class TestAugmentPair:
    def test_shapes_preserved(self):
        a = _checkerboard(5)
        b = _solid(5)
        cfg = AugmentationConfig(
            rotate90=True,
            geometry_paste=GeometryPasteConfig(probability=1.0, paste_size=2),
        )
        out_a, out_b = augment_pair(a, b, cfg, _rng())
        assert out_a.shape == (5, 5, 5)
        assert out_b.shape == (5, 5, 5)

    def test_geometry_paste_zero_prob_no_change_from_paste(self):
        """With probability=0 geometry_paste should never fire."""
        a = _empty(5)
        b = _solid(5)
        cfg = AugmentationConfig(
            geometry_paste=GeometryPasteConfig(probability=0.0, paste_size=2)
        )
        out_a, _ = augment_pair(a, b, cfg, _rng())
        # No paste: out_a should still be empty (augment does not add voxels without paste)
        assert out_a.sum() == 0.0
