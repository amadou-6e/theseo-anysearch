"""Geometry pool — pre-computed voxel grids sampled at episode reset.

Pool layout on disk::

    <pool_dir>/
    ├── pool_meta.json
    ├── <source_stem>/
    │   ├── 0000.npy   # uint8 (grid_size, grid_size, grid_size), 1=filled
    │   ├── 0001.npy
    │   └── ...
    └── ...

Each .npy is a uint8 numpy array of shape (G, G, G) where G = grid_size.
A value of 1 means the voxel is filled (obstacle), 0 means free.
Voxel coordinates are 0-indexed in the array but 1-indexed in the env
(i.e. array[x-1, y-1, z-1] == 1 means voxel (x, y, z) is filled).
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


class GeometryPool:
    """Loads and samples pre-computed voxel grids from a pool directory.

    Usage::

        pool = GeometryPool(pool_dir="runtime/geometry_pools/highres", seed=42)
        grid = pool.sample()         # np.ndarray uint8 (G, G, G)
        cells = pool.grid_to_cells(grid)  # list of (x, y, z) 1-indexed tuples
    """

    def __init__(self, pool_dir: str | Path, seed: int = 42) -> None:
        self._pool_dir = Path(pool_dir)
        self._rng = np.random.default_rng(seed)
        self._lock = threading.Lock()

        meta_path = self._pool_dir / "pool_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"pool_meta.json not found in {self._pool_dir}. "
                "Run: anysearch extract <config.yaml>"
            )
        with meta_path.open() as f:
            self._meta: dict = json.load(f)

        self._grid_size: int = self._meta.get("grid_size", 32)
        self._files: list[Path] = sorted(self._pool_dir.rglob("*.npy"))

        if not self._files:
            raise ValueError(
                f"Geometry pool at {self._pool_dir} has no .npy files. "
                "Run: anysearch extract <config.yaml>"
            )

        log.info(
            "GeometryPool: %d variants in %s (grid_size=%d)",
            len(self._files),
            self._pool_dir,
            self._grid_size,
        )

    @property
    def size(self) -> int:
        """Total number of .npy files in the pool."""
        return len(self._files)

    @property
    def grid_size(self) -> int:
        return self._grid_size

    @property
    def meta(self) -> dict:
        return self._meta

    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Return a randomly sampled voxel grid (uint8, shape (G,G,G)).

        Thread-safe: the RNG is locked before use.
        """
        selected_rng = self._rng if rng is None else rng
        with self._lock:
            idx = int(selected_rng.integers(0, len(self._files)))
        path = self._files[idx]
        grid = np.load(path)
        if grid.dtype != np.uint8:
            grid = grid.astype(np.uint8)
        return grid

    @staticmethod
    def grid_to_cells(grid: np.ndarray) -> list[tuple[int, int, int]]:
        """Convert a (G,G,G) uint8 grid to a list of 1-indexed (x,y,z) tuples.

        The .npy arrays use 0-based indexing; the Rust env uses 1-based coords.
        """
        xs, ys, zs = np.nonzero(grid)
        # +1 to convert 0-based array index to 1-based env coord
        return [(int(x) + 1, int(y) + 1, int(z) + 1) for x, y, z in zip(xs, ys, zs)]

    def validate(self, min_per_source: int = 10, *, min_total: int | None = None) -> tuple[bool, str]:
        """Check the pool has enough entries per source.

        Returns (ok, message). Called by `anysearch run` before training.
        """
        sources = self._meta.get("sources", {})
        threshold = min_total if min_total is not None else min_per_source
        if not sources:
            # No per-source tracking — just check total
            if len(self._files) < threshold:
                return False, (
                    f"Geometry pool '{self._pool_dir}' has only {len(self._files)} variants "
                    f"(need >= {threshold}). Run: anysearch extract <config.yaml>"
                )
            return True, f"{len(self._files)} variants OK"

        errors: list[str] = []
        for src, count in sources.items():
            if count < min_per_source:
                errors.append(f"  {src}: {count} / {min_per_source}")
        if errors:
            msg = (
                f"Geometry pool '{self._pool_dir}' is too small:\n"
                + "\n".join(errors)
                + "\nRun: anysearch extract <config.yaml>"
            )
            return False, msg
        return True, f"{len(self._files)} variants across {len(sources)} sources OK"


def paste_boxes(
    grid: np.ndarray,
    config: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply random box paste-in augmentation to a voxel grid.

    Places N random solid boxes into the grid (obstacles).  Each reset samples
    fresh box positions so the same base geometry yields different obstacle
    layouts.

    Args:
        grid:   uint8 (G, G, G) array — 1 = filled, 0 = free.  Modified in-place
                and returned.
        config: dict with keys:
                  num_boxes:    [min, max] or int — number of boxes to paste
                  box_min_size: [x, y, z]         — minimum box side lengths
                  box_max_size: [x, y, z]         — maximum box side lengths
                  prob:         float (default 1.0) — probability of applying
        rng:    numpy Generator (caller's RNG, kept in sync with episode seed)

    Returns:
        The augmented grid (same object, modified in-place).
    """
    prob = float(config.get("prob", 1.0))
    if prob < 1.0 and rng.random() >= prob:
        return grid

    G = grid.shape[0]

    num_boxes_cfg = config.get("num_boxes", [2, 8])
    if isinstance(num_boxes_cfg, (list, tuple)):
        n_boxes = int(rng.integers(num_boxes_cfg[0], num_boxes_cfg[1] + 1))
    else:
        n_boxes = int(num_boxes_cfg)

    min_sz = config.get("box_min_size", [2, 2, 2])
    max_sz = config.get("box_max_size", [6, 6, 6])

    for _ in range(n_boxes):
        sx = int(rng.integers(min_sz[0], max_sz[0] + 1))
        sy = int(rng.integers(min_sz[1], max_sz[1] + 1))
        sz = int(rng.integers(min_sz[2], max_sz[2] + 1))
        # Random corner, leaving room for the box
        ox = int(rng.integers(0, max(G - sx, 1)))
        oy = int(rng.integers(0, max(G - sy, 1)))
        oz = int(rng.integers(0, max(G - sz, 1)))
        grid[ox:ox + sx, oy:oy + sy, oz:oz + sz] = 1

    return grid
