"""3D voxel map loader for ``.3dmap.zip`` benchmark files.

Supports the ``voxel W H D`` (obstacle list) and ``rev_voxel W H D``
(free-space list) headers used by the shortestpathlab and MovingAI benchmarks.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

import numpy as np


def _parse_3dmap(content: str) -> tuple[np.ndarray, int, int, int]:
    """Parse ASCII .3dmap content.  Returns (obstacle_coords_nx3, W, H, D)."""
    lines = content.strip().splitlines()
    header = lines[0].split()
    mode = header[0]          # "voxel" or "rev_voxel"
    W, H, D = int(header[1]), int(header[2]), int(header[3])

    coords: list[tuple[int, int, int]] = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 3:
            try:
                coords.append((int(parts[0]), int(parts[1]), int(parts[2])))
            except ValueError:
                continue

    arr = np.array(coords, dtype=np.int32) if coords else np.empty((0, 3), dtype=np.int32)

    if mode == "rev_voxel":
        # Free-space list → invert to obstacle list
        free_set = {(r[0], r[1], r[2]) for r in arr}
        obs = [
            (x, y, z)
            for x in range(W)
            for y in range(H)
            for z in range(D)
            if (x, y, z) not in free_set
        ]
        arr = np.array(obs, dtype=np.int32) if obs else np.empty((0, 3), dtype=np.int32)

    return arr, W, H, D


def load_3dmap_sparse(path: str | Path) -> tuple[np.ndarray, int, int, int]:
    """Load a ``.3dmap.zip`` file.

    Returns ``(coords, W, H, D)`` where *coords* is an ``(N, 3)`` int32 array
    of obstacle voxel coordinates and *W, H, D* are the map dimensions.
    """
    path = str(path)
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        content = zf.read(name).decode(errors="replace")
    return _parse_3dmap(content)


def crop_to_grid(
    coords: np.ndarray,
    origin: tuple[int, int, int],
    grid_size: int,
) -> np.ndarray:
    """Crop a ``grid_size³`` window at *origin* from sparse obstacle coords.

    Returns a ``(grid_size, grid_size, grid_size)`` uint8 array (1 = obstacle).
    """
    ox, oy, oz = origin
    g = grid_size
    if coords.shape[0] == 0:
        return np.zeros((g, g, g), dtype=np.uint8)

    ix = coords[:, 0] - ox
    iy = coords[:, 1] - oy
    iz = coords[:, 2] - oz
    mask = (ix >= 0) & (ix < g) & (iy >= 0) & (iy < g) & (iz >= 0) & (iz < g)
    sub = np.stack([ix[mask], iy[mask], iz[mask]], axis=1)

    grid = np.zeros((g, g, g), dtype=np.uint8)
    if sub.shape[0] > 0:
        grid[sub[:, 0], sub[:, 1], sub[:, 2]] = 1
    return grid


def suggest_crop_origin(
    path: str | Path,
    grid_size: int,
    n_trials: int = 100,
    seed: int = 42,
) -> tuple[int, int, int]:
    """Find a crop window with non-trivial obstacle density.

    Samples *n_trials* random origins and picks the one whose fill percentage
    is closest to 30 % (biased toward interesting routing environments).
    """
    coords, W, H, D = load_3dmap_sparse(path)
    rng = np.random.default_rng(seed)
    total = grid_size ** 3

    max_x = max(0, W - grid_size)
    max_y = max(0, H - grid_size)
    max_z = max(0, D - grid_size)

    best_origin: tuple[int, int, int] = (0, 0, 0)
    best_score = -1.0

    for _ in range(n_trials):
        ox = int(rng.integers(0, max_x + 1))
        oy = int(rng.integers(0, max_y + 1))
        oz = int(rng.integers(0, max_z + 1))

        if coords.shape[0] > 0:
            ix = coords[:, 0] - ox
            iy = coords[:, 1] - oy
            iz = coords[:, 2] - oz
            mask = (
                (ix >= 0) & (ix < grid_size) &
                (iy >= 0) & (iy < grid_size) &
                (iz >= 0) & (iz < grid_size)
            )
            count = int(mask.sum())
        else:
            count = 0

        pct = count / total
        # Prefer 5–60 % fill; penalise near-empty or near-solid
        score = pct if 0.05 <= pct <= 0.60 else pct * 0.1
        if score > best_score:
            best_score = score
            best_origin = (ox, oy, oz)

    return best_origin
