"""Post-voxelization augmentations on (N, N, N) binary float32 grids."""
from __future__ import annotations

import numpy as np

from theseo_anysearch.garden.data_config import AugmentationConfig


def augment(grid: np.ndarray, cfg: AugmentationConfig, rng: np.random.Generator) -> np.ndarray:
    """Apply all enabled augmentations to a single (N, N, N) grid. Returns a new array."""
    grid = grid.copy()
    if cfg.rotate90:
        grid = _rotate90(grid, rng)
    if cfg.flip:
        grid = _flip(grid, rng)
    if cfg.cutout is not None:
        grid = _cutout(grid, cfg.cutout.count, cfg.cutout.min_size, cfg.cutout.max_size, rng)
    if cfg.noise_prob > 0.0:
        grid = _noise(grid, cfg.noise_prob, rng)
    if cfg.translation_jitter > 0:
        grid = _translation_jitter(grid, cfg.translation_jitter, rng)
    if cfg.morph_prob > 0.0:
        grid = _morph(grid, cfg.morph_prob, rng)
    return grid


def augment_pair(
    grid_a: np.ndarray,
    grid_b: np.ndarray,
    cfg: AugmentationConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Augment a pair of grids, applying geometry_paste between them."""
    grid_a = augment(grid_a, cfg, rng)
    grid_b = augment(grid_b, cfg, rng)
    if cfg.geometry_paste is not None and rng.random() < cfg.geometry_paste.probability:
        grid_a = _geometry_paste(grid_a, grid_b, cfg.geometry_paste.paste_size, rng)
    return grid_a, grid_b


# ---------------------------------------------------------------------------
# Individual transforms
# ---------------------------------------------------------------------------

# All 24 axis-aligned rotation matrices (k90° around x/y/z axes, compositions)
_ROT90_TRANSFORMS: list[tuple[tuple[int, ...], tuple[bool, ...]]] = [
    # (axes permutation, flip flags) — encodes each of the 24 SO(3) cube symmetries
    # Generated as (perm, flips) such that np.flip(np.transpose(g, perm), flip_axes) gives the rotated grid
    ((0, 1, 2), ()),          # identity
    ((0, 2, 1), (1,)),        # 90° around x
    ((0, 1, 2), (1, 2)),      # 180° around x
    ((0, 2, 1), (2,)),        # 270° around x
    ((2, 1, 0), (0,)),        # 90° around y
    ((1, 0, 2), (0, 1)),      # 180° around y
    ((2, 1, 0), (2,)),        # 270° around y
    ((1, 2, 0), ()),          # 90° around z
    ((0, 1, 2), (0, 2)),      # 180° around z
    ((2, 0, 1), ()),          # 270° around z
    ((1, 0, 2), (1,)),        # face diagonal 1
    ((1, 0, 2), (0,)),        # face diagonal 2
    ((0, 2, 1), (0,)),        # face diagonal 3
    ((0, 2, 1), (0, 1, 2)),   # face diagonal 4
    ((2, 1, 0), (0, 1)),      # face diagonal 5
    ((2, 1, 0), (1, 2)),      # face diagonal 6
    ((1, 2, 0), (0, 1)),      # body diagonal 1
    ((1, 2, 0), (1, 2)),      # body diagonal 2
    ((1, 2, 0), (0, 2)),      # body diagonal 3
    ((2, 0, 1), (0, 1)),      # body diagonal 4
    ((2, 0, 1), (1, 2)),      # body diagonal 5
    ((2, 0, 1), (0, 2)),      # body diagonal 6
    ((1, 0, 2), (0, 1, 2)),   # body diagonal 7
    ((2, 1, 0), (0, 1, 2)),   # body diagonal 8
]


def _rotate90(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a random axis-aligned 90° rotation (one of 24 cube symmetries)."""
    idx = int(rng.integers(0, len(_ROT90_TRANSFORMS)))
    perm, flip_axes = _ROT90_TRANSFORMS[idx]
    g = np.transpose(grid, perm)
    if flip_axes:
        g = np.flip(g, axis=flip_axes)
    return np.ascontiguousarray(g)


def _flip(grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomly flip along each of the 3 axes independently."""
    for ax in (0, 1, 2):
        if rng.random() < 0.5:
            grid = np.flip(grid, axis=ax)
    return np.ascontiguousarray(grid)


def _cutout(grid: np.ndarray, count: int, min_size: int, max_size: int, rng: np.random.Generator) -> np.ndarray:
    n = grid.shape[0]
    for _ in range(count):
        size = int(rng.integers(min_size, max_size + 1))
        ox = int(rng.integers(0, max(1, n - size + 1)))
        oy = int(rng.integers(0, max(1, n - size + 1)))
        oz = int(rng.integers(0, max(1, n - size + 1)))
        grid[ox:ox + size, oy:oy + size, oz:oz + size] = 0.0
    return grid


def _geometry_paste(
    target: np.ndarray,
    source: np.ndarray,
    paste_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = target.shape[0]
    ps = min(paste_size, n)
    # Source sub-cube origin
    sx = int(rng.integers(0, max(1, n - ps + 1)))
    sy = int(rng.integers(0, max(1, n - ps + 1)))
    sz = int(rng.integers(0, max(1, n - ps + 1)))
    # Target paste origin
    tx = int(rng.integers(0, max(1, n - ps + 1)))
    ty = int(rng.integers(0, max(1, n - ps + 1)))
    tz = int(rng.integers(0, max(1, n - ps + 1)))
    patch = source[sx:sx + ps, sy:sy + ps, sz:sz + ps]
    target = target.copy()
    target[tx:tx + ps, ty:ty + ps, tz:tz + ps] = np.maximum(
        target[tx:tx + ps, ty:ty + ps, tz:tz + ps], patch
    )
    return target


def _noise(grid: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(grid.shape) < prob
    grid = grid.copy()
    grid[mask] = 1.0 - grid[mask]
    return grid


def _translation_jitter(grid: np.ndarray, max_shift: int, rng: np.random.Generator) -> np.ndarray:
    shifts = rng.integers(-max_shift, max_shift + 1, size=3)
    return np.roll(grid, shifts.tolist(), axis=(0, 1, 2))


def _morph(grid: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    if rng.random() >= prob:
        return grid
    try:
        from scipy.ndimage import binary_erosion, binary_dilation
        binary = grid > 0.5
        if rng.random() < 0.5:
            result = binary_erosion(binary).astype(np.float32)
        else:
            result = binary_dilation(binary).astype(np.float32)
        return result
    except ImportError:
        return grid
