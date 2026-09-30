"""``anysearch extract`` — offline geometry pool builder.

Voxelizes STL files (with optional SO(3) rotation and scale sampling), applies
quality filters, and saves ready-to-use ``.npy`` grids to a pool directory.

Worker function is module-level so it can be pickled by ProcessPoolExecutor.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Uniform SO(3) sample via Shoemake's quaternion method."""
    u1, u2, u3 = rng.uniform(0.0, 1.0, 3)
    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    R = np.array([
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ], dtype=np.float64)
    return R


def _rotate_ascii_stl(path: str, R: np.ndarray) -> str:
    """Read an ASCII STL, apply rotation matrix R to all vertices, return new content."""
    lines_out: list[str] = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("vertex"):
                parts = stripped.split()
                if len(parts) >= 4:
                    v = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    rv = R @ v
                    indent = line[: len(line) - len(line.lstrip())]
                    lines_out.append(f"{indent}vertex {rv[0]:.8f} {rv[1]:.8f} {rv[2]:.8f}\n")
                    continue
            lines_out.append(line)
    return "".join(lines_out)


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

def _bfs_largest_free_component(
    filled_cells: list[tuple[int, int, int]],
    grid_size: int,
) -> int:
    """BFS on free cells; returns size of the largest connected component."""
    total = grid_size ** 3
    filled_set = set(filled_cells)
    # Collect all free cells
    free_cells = [
        (x, y, z)
        for x in range(1, grid_size + 1)
        for y in range(1, grid_size + 1)
        for z in range(1, grid_size + 1)
        if (x, y, z) not in filled_set
    ]
    if not free_cells:
        return 0

    unvisited = set(free_cells)
    start = free_cells[0]
    unvisited.discard(start)
    q: collections.deque[tuple[int, int, int]] = collections.deque([start])
    visited = 1
    while q:
        cx, cy, cz = q.popleft()
        for dx, dy, dz in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
            nb = (cx + dx, cy + dy, cz + dz)
            if nb in unvisited:
                unvisited.discard(nb)
                visited += 1
                q.append(nb)
    return visited


# ---------------------------------------------------------------------------
# Per-sample worker (module-level for pickling)
# ---------------------------------------------------------------------------

def _extract_worker(args: dict) -> dict:
    """Generate one voxelized geometry sample.

    Returns a dict with keys: ``saved`` (Path or None), ``reason`` (str).
    """
    stl_path: str = args["stl_path"]
    out_path: str = args["out_path"]
    grid_size: int = args["grid_size"]
    scale: float = args["scale"]
    padding: int = args["padding"]
    rotate: bool = args["rotate"]
    seed: int = args["seed"]
    min_fill_pct: float = args["min_fill_pct"]
    min_free_pct: float = args["min_free_pct"]
    connectivity_check: bool = args["connectivity_check"]

    from theseo_anysearch.environments.pettingzoo.multi_voxel_env import _load_stl_geometry

    rng = np.random.default_rng(seed)
    work_stl = stl_path
    rotation_matrix: list | None = None

    tmp_stl: str | None = None
    try:
        if rotate:
            R = _random_rotation_matrix(rng)
            rotation_matrix = R.tolist()
            rotated_content = _rotate_ascii_stl(stl_path, R)
            fd, tmp_stl = tempfile.mkstemp(suffix=".stl")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(rotated_content)
            work_stl = tmp_stl

        cells = _load_stl_geometry(work_stl, scale, grid_size, padding=padding)

        total = grid_size ** 3
        filled = len(cells)
        free = total - filled
        fill_pct = 100.0 * filled / total
        free_pct = 100.0 * free / total

        if fill_pct < min_fill_pct:
            return {"saved": None, "reason": f"fill {fill_pct:.1f}% < {min_fill_pct}%"}
        if free_pct < min_free_pct:
            return {"saved": None, "reason": f"free {free_pct:.1f}% < {min_free_pct}%"}

        if connectivity_check and free > 0:
            largest = _bfs_largest_free_component(cells, grid_size)
            conn_pct = 100.0 * largest / free
            if conn_pct < 80.0:
                return {
                    "saved": None,
                    "reason": f"connectivity {conn_pct:.1f}% < 80%",
                }

        # Build (G, G, G) uint8 grid — 0-indexed array, 1-indexed env coords
        grid = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
        for x, y, z in cells:
            grid[x - 1, y - 1, z - 1] = 1

        # Atomic write: temp file → rename
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fd2, tmp_out = tempfile.mkstemp(suffix=".npy", dir=out.parent)
        os.close(fd2)
        try:
            np.save(tmp_out, grid)
            os.replace(tmp_out, out)
        except Exception:
            try:
                os.unlink(tmp_out)
            except OSError:
                pass
            raise

        # Write sidecar with the exact scale and rotation used for this sample
        sidecar = {"scale": scale, "rotation": rotation_matrix}
        out.with_suffix(".meta.json").write_text(json.dumps(sidecar))

        return {"saved": out_path, "reason": f"fill={fill_pct:.1f}%  free={free_pct:.1f}%"}

    finally:
        if tmp_stl and os.path.exists(tmp_stl):
            try:
                os.unlink(tmp_stl)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _src_stem(path: Path) -> str:
    """Stable directory name for a source file.

    Strips .3dmap from .3dmap.zip map files so the pool sub-directory is
    e.g. ``plant01`` rather than ``plant01.3dmap``.
    """
    stem = path.stem  # removes last extension (.zip or .stl)
    if stem.endswith(".3dmap"):
        stem = stem[: -len(".3dmap")]
    return stem


# ---------------------------------------------------------------------------
# Per-map worker (module-level for pickling)
# ---------------------------------------------------------------------------

def _extract_map_worker(args: dict) -> dict:
    """Crop one window from a .3dmap.zip file and save as .npy.

    Returns a dict with keys: ``saved`` (Path or None), ``reason`` (str).
    """
    map_path: str = args["map_path"]
    out_path: str = args["out_path"]
    grid_size: int = args["grid_size"]
    origin: tuple[int, int, int] = tuple(args["origin"])  # type: ignore[assignment]
    min_fill_pct: float = args["min_fill_pct"]
    min_free_pct: float = args["min_free_pct"]
    pad: int = args.get("padding", 2)

    from theseo_anysearch.environments.map3d import load_3dmap_sparse, crop_to_grid

    coords, W, H, D = load_3dmap_sparse(map_path)
    grid = crop_to_grid(coords, origin, grid_size)

    # Clear outer padding layers so agents always have navigable space at edges
    if pad > 0:
        grid[:pad, :, :] = 0
        grid[-pad:, :, :] = 0
        grid[:, :pad, :] = 0
        grid[:, -pad:, :] = 0
        grid[:, :, :pad] = 0
        grid[:, :, -pad:] = 0

    total = grid_size ** 3
    filled = int(grid.sum())
    free = total - filled
    fill_pct = 100.0 * filled / total
    free_pct = 100.0 * free / total

    if fill_pct < min_fill_pct:
        return {"saved": None, "reason": f"fill {fill_pct:.1f}% < {min_fill_pct}%"}
    if free_pct < min_free_pct:
        return {"saved": None, "reason": f"free {free_pct:.1f}% < {min_free_pct}%"}

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_out = tempfile.mkstemp(suffix=".npy", dir=out.parent)
    os.close(fd)
    try:
        np.save(tmp_out, grid)
        os.replace(tmp_out, out)
    except Exception:
        try:
            os.unlink(tmp_out)
        except OSError:
            pass
        raise

    sidecar = {"scale": None, "rotation": None, "origin": list(origin), "source_type": "map"}
    Path(out_path).with_suffix(".meta.json").write_text(json.dumps(sidecar))

    return {"saved": out_path, "reason": f"fill={fill_pct:.1f}%  free={free_pct:.1f}%"}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_extract(
    stl_files: list[Path],
    pool_dir: Path,
    target_per_source: int,
    grid_size: int,
    scale_range: tuple[float, float],
    padding: int,
    rotate: bool,
    workers: int,
    resume: bool,
    min_fill_pct: float,
    min_free_pct: float,
    connectivity_check: bool,
    seed: int = 0,
    map_files: list[Path] | None = None,
    map_grid_size: int | None = None,
) -> None:
    """Orchestrate parallel extraction and write pool_meta.json."""
    import concurrent.futures
    import sys
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

    console = Console(file=sys.stdout)
    pool_dir.mkdir(parents=True, exist_ok=True)

    # Build work items
    work_items: list[dict] = []
    existing_by_source: dict[str, int] = {}

    for src in stl_files:
        stem = _src_stem(src)
        src_dir = pool_dir / stem
        existing = sorted(src_dir.glob("*.npy")) if src_dir.exists() else []
        existing_count = len(existing)
        existing_by_source[stem] = existing_count

        next_idx = existing_count if resume else 0
        if resume:
            remaining = max(0, target_per_source - existing_count)
        else:
            # Re-generate everything
            remaining = target_per_source
            next_idx = existing_count  # Start after existing if any

        if remaining == 0:
            console.print(f"  [dim]{stem}: {existing_count}/{target_per_source} already done, skipping[/dim]")
            continue

        scale_lo, scale_hi = scale_range
        for i in range(remaining):
            rng_i = np.random.default_rng(seed + abs(hash(stem)) + i + next_idx)
            sample_scale = float(rng_i.uniform(scale_lo, scale_hi))
            file_idx = next_idx + i
            out_path = src_dir / f"{file_idx:04d}.npy"
            work_items.append({
                "stl_path": str(src),
                "out_path": str(out_path),
                "grid_size": grid_size,
                "scale": sample_scale,
                "padding": padding,
                "rotate": rotate,
                "seed": seed + abs(hash(stem)) + i + next_idx,
                "min_fill_pct": min_fill_pct,
                "min_free_pct": min_free_pct,
                "connectivity_check": connectivity_check,
            })

    # Build map work items
    map_work_items: list[dict] = []
    eff_map_grid = map_grid_size or grid_size
    for src in (map_files or []):
        stem = _src_stem(src)
        src_dir = pool_dir / stem
        existing = sorted(src_dir.glob("*.npy")) if src_dir.exists() else []
        existing_count = len(existing)
        existing_by_source[stem] = existing_count

        next_idx = existing_count if resume else 0
        remaining = max(0, target_per_source - existing_count) if resume else target_per_source

        if remaining == 0:
            console.print(f"  [dim]{stem}: {existing_count}/{target_per_source} already done, skipping[/dim]")
            continue

        # Sample crop origins biased toward obstacle voxels.  Generate 4× the target
        # so quality filters in the worker can discard near-empty windows.
        from theseo_anysearch.environments.map3d import load_3dmap_sparse
        coords, W, H, D = load_3dmap_sparse(str(src))
        rng_map = np.random.default_rng(seed + abs(hash(stem)))
        half = eff_map_grid // 2
        n_candidates = remaining * 4

        if coords.shape[0] > 0:
            pick_idx = rng_map.integers(0, coords.shape[0], size=n_candidates)
            cx_arr = coords[pick_idx, 0]
            cy_arr = coords[pick_idx, 1]
            cz_arr = coords[pick_idx, 2]
        else:
            cx_arr = rng_map.integers(0, max(1, W), size=n_candidates)
            cy_arr = rng_map.integers(0, max(1, H), size=n_candidates)
            cz_arr = rng_map.integers(0, max(1, D), size=n_candidates)

        seen_origins: set[tuple[int, int, int]] = set()
        for ci in range(n_candidates):
            ox = int(np.clip(int(cx_arr[ci]) - half, 0, max(0, W - eff_map_grid)))
            oy = int(np.clip(int(cy_arr[ci]) - half, 0, max(0, H - eff_map_grid)))
            oz = int(np.clip(int(cz_arr[ci]) - half, 0, max(0, D - eff_map_grid)))
            key = (ox, oy, oz)
            if key in seen_origins:
                continue
            seen_origins.add(key)
            file_idx = next_idx + len(seen_origins) - 1
            out_path = src_dir / f"{file_idx:04d}.npy"
            map_work_items.append({
                "map_path": str(src),
                "out_path": str(out_path),
                "grid_size": eff_map_grid,
                "origin": [ox, oy, oz],
                "padding": padding,
                "min_fill_pct": min_fill_pct,
                "min_free_pct": min_free_pct,
            })

    all_sources = (stl_files or []) + (map_files or [])
    if not work_items and not map_work_items:
        console.print("[green]Pool is already complete.[/green]")
        _write_meta(pool_dir, all_sources, target_per_source, grid_size, scale_range, rotate)
        return

    total_items = len(work_items) + len(map_work_items)
    console.print(
        f"Extracting [bold]{total_items}[/bold] samples "
        f"({len(all_sources)} source(s) × up to {target_per_source} each) "
        f"with {workers} worker(s)..."
    )

    saved = 0
    rejected = 0
    source_counts: dict[str, int] = dict(existing_by_source)

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting", total=total_items)

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            stl_futures = {executor.submit(_extract_worker, item): item for item in work_items}
            map_futures = {executor.submit(_extract_map_worker, item): item for item in map_work_items}
            all_futures = {**stl_futures, **map_futures}
            for future in concurrent.futures.as_completed(all_futures):
                item = all_futures[future]
                src_key = "stl_path" if "stl_path" in item else "map_path"
                try:
                    result = future.result()
                except Exception as exc:
                    log.warning("Worker failed for %s: %s", item[src_key], exc)
                    rejected += 1
                else:
                    if result["saved"]:
                        saved += 1
                        stem = _src_stem(Path(item[src_key]))
                        source_counts[stem] = source_counts.get(stem, 0) + 1
                    else:
                        rejected += 1
                progress.advance(task)

    console.print(
        f"[green]Done:[/green] {saved} saved, {rejected} rejected "
        f"(total pool: {saved + sum(existing_by_source.values())} files)"
    )

    _write_meta(pool_dir, all_sources, target_per_source, grid_size, scale_range, rotate)


def _write_meta(
    pool_dir: Path,
    all_sources: list[Path],
    target_per_source: int,
    grid_size: int,
    scale_range: tuple[float, float],
    rotate: bool,
) -> None:
    """Write pool_meta.json with generation config and per-source file counts."""
    sources: dict[str, int] = {}
    for src in all_sources:
        stem = _src_stem(src)
        src_dir = pool_dir / stem
        sources[stem] = len(list(src_dir.glob("*.npy"))) if src_dir.exists() else 0

    total = sum(sources.values())
    # Absolute paths so the pool explorer can find the source from any cwd.
    source_paths = {_src_stem(src): str(src.resolve()) for src in all_sources}
    meta = {
        "grid_size": grid_size,
        "target_per_source": target_per_source,
        "scale_range": list(scale_range),
        "rotate": rotate,
        "total_variants": total,
        "sources": sources,
        "source_paths": source_paths,
    }
    meta_path = pool_dir / "pool_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
