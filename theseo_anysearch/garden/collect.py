"""Data collection for garden pre-training.

Two strategies:
  - RandomRolloutCollector: run MultiVoxelEnv with random actions, collect
    local_grid at every step. Works immediately with existing Rust env.
    (PyVoxelSampler batch sampling is a future optimisation.)
  - TrainingRunCollector: extract cursor positions from saved trajectory JSON
    files then query box_obs at those positions.

Both write cached .npz files to cache_dir/obs/<source_hash>.npz.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from theseo_anysearch.garden.data_config import (
    CacheConfig,
    DataConfig,
    EnvGeometryConfig,
    PositionSampleSourceConfig,
    RandomBoxesSamplerConfig,
    SourceConfig,
    StlListSamplerConfig,
    TrainingRunSourceConfig,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PilotObservationArtifact:
    """Geometry-identified observation rows stored as one immutable artifact."""

    grids: np.ndarray
    geometry_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    identity_sha256: str


def pilot_observation_sha256(
    grids: np.ndarray, geometry_ids: Sequence[str], observation_ids: Sequence[str]
) -> str:
    """Hash observation bytes together with their geometry and row identities."""

    array = np.ascontiguousarray(grids)
    if array.ndim != 4 or len(set(array.shape[1:])) != 1:
        raise ValueError("pilot observations must have shape (N, D, H, W) with cubic rows")
    if len(array) != len(geometry_ids) or len(array) != len(observation_ids):
        raise ValueError("observation rows and identities must have equal lengths")
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("pilot observation IDs must be unique")
    header = {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "geometry_ids": list(geometry_ids),
        "observation_ids": list(observation_ids),
    }
    digest = hashlib.sha256(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def save_pilot_observations(
    path: Path,
    grids: np.ndarray,
    geometry_ids: Sequence[str],
    observation_ids: Sequence[str],
) -> str:
    """Write one content-addressed pilot observation artifact without replacement."""

    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("pilot observation artifacts must use .npz")
    identity = pilot_observation_sha256(grids, geometry_ids, observation_ids)
    if path.exists():
        existing = load_pilot_observations(path)
        if existing.identity_sha256 != identity:
            raise FileExistsError(f"refusing to replace frozen pilot observations: {path}")
        return identity

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            observations=np.asarray(grids),
            geometry_ids=np.asarray(geometry_ids, dtype=str),
            observation_ids=np.asarray(observation_ids, dtype=str),
        )
    temporary.replace(path)
    manifest = {
        "schema_version": 1,
        "identity_sha256": identity,
        "rows": len(grids),
        "shape": list(np.asarray(grids).shape),
        "dtype": np.asarray(grids).dtype.str,
    }
    path.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return identity


def load_pilot_observations(path: Path) -> PilotObservationArtifact:
    """Load an immutable observation artifact and verify its content identity."""

    path = Path(path)
    manifest_path = path.with_suffix(".json")
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"pilot observation artifact is incomplete: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest) != {"schema_version", "identity_sha256", "rows", "shape", "dtype"}:
        raise ValueError("invalid pilot observation manifest")
    with np.load(path, allow_pickle=False) as stored:
        grids = stored["observations"]
        geometry_ids = tuple(str(value) for value in stored["geometry_ids"].tolist())
        observation_ids = tuple(str(value) for value in stored["observation_ids"].tolist())
    identity = pilot_observation_sha256(grids, geometry_ids, observation_ids)
    if manifest["schema_version"] != 1 or manifest["identity_sha256"] != identity:
        raise ValueError("pilot observation artifact failed identity verification")
    if manifest["rows"] != len(grids) or manifest["shape"] != list(grids.shape):
        raise ValueError("pilot observation manifest shape does not match stored rows")
    if manifest["dtype"] != grids.dtype.str:
        raise ValueError("pilot observation manifest dtype does not match stored rows")
    return PilotObservationArtifact(grids, geometry_ids, observation_ids, identity)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _source_hash(source: SourceConfig) -> str:
    raw = source.model_dump_json()
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _obs_cache_path(cache_cfg: CacheConfig, source: SourceConfig) -> Path:
    return Path(cache_cfg.path) / "obs" / f"{_source_hash(source)}.npz"


def _load_cached(path: Path) -> np.ndarray | None:
    if path.exists():
        data = np.load(path)
        return data["observations"]
    return None


def _save_cached(path: Path, observations: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, observations=observations)
    (path.parent / f"{path.stem}_meta.json").write_text(json.dumps(meta, indent=2))


# ---------------------------------------------------------------------------
# STL scale helper
# ---------------------------------------------------------------------------

def _stl_fit_scale(stl_path: str, grid_size: int) -> float:
    """Return a scale that fits the STL's bounding box within grid_size voxels.

    Reads vertex coordinates from an ASCII STL file and computes:
        scale = (grid_size - 2) / max_extent
    leaving a 1-cell margin on each side. Falls back to 1.0 if the file
    cannot be parsed or has no vertices.
    """
    try:
        verts: list[list[float]] = []
        with open(stl_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("vertex"):
                    parts = stripped.split()
                    if len(parts) >= 4:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if not verts:
            return 1.0
        arr = np.array(verts, dtype=np.float64)
        max_extent = float((arr.max(axis=0) - arr.min(axis=0)).max())
        if max_extent <= 0.0:
            return 1.0
        return (grid_size - 2) / max_extent
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Geometry environment builder
# ---------------------------------------------------------------------------

def _build_env_config(geo: EnvGeometryConfig, box_radius: int) -> dict:
    cfg: dict = {
        "agent_count": 1,
        "max_steps": 200,
        "obs_mode": "box",
        "box_radius": box_radius,
        "trail_mode": False,
        "grid_size": geo.grid_size,
        "scale": geo.scale,
        "seed": 0,
    }
    if geo.stl_path is not None:
        cfg["stl_path"] = str(geo.stl_path)
    if geo.geometry_boxes is not None:
        cfg["geometry_boxes"] = geo.geometry_boxes
    return cfg


def _make_env(env_cfg_dict: dict):
    from theseo_anysearch.environments.pettingzoo.multi_voxel_env import MultiVoxelEnv
    return MultiVoxelEnv(env_cfg_dict)


def _random_boxes_geometry(sampler: RandomBoxesSamplerConfig, idx: int, rng: np.random.Generator) -> dict:
    """Generate a random geometry_boxes config for one procedural scene."""
    n_boxes = int(rng.integers(sampler.num_boxes[0], sampler.num_boxes[1] + 1))
    boxes = []
    gs = sampler.grid_size
    for _ in range(n_boxes):
        sx = int(rng.integers(sampler.box_min_size[0], sampler.box_max_size[0] + 1))
        sy = int(rng.integers(sampler.box_min_size[1], sampler.box_max_size[1] + 1))
        sz = int(rng.integers(sampler.box_min_size[2], sampler.box_max_size[2] + 1))
        xmin = int(rng.integers(1, gs - sx))
        ymin = int(rng.integers(1, gs - sy))
        zmin = int(rng.integers(1, gs - sz))
        boxes.append([xmin, ymin, zmin, xmin + sx, ymin + sy, zmin + sz])
    return {
        "agent_count": 1,
        "max_steps": 200,
        "obs_mode": "box",
        "box_radius": 2,
        "trail_mode": False,
        "grid_size": sampler.grid_size,
        "scale": 1.0,
        "seed": idx,
        "geometry_boxes": boxes,
    }


# ---------------------------------------------------------------------------
# Core collection: run random episodes in an env, yield local_grid arrays
# ---------------------------------------------------------------------------

def _collect_from_env(env_cfg: dict, n_samples: int, progress_cb: Callable[[int], None] | None = None) -> np.ndarray:
    """Collect n_samples local_grid observations via random rollouts."""
    env = _make_env(env_cfg)
    agent = "agent_0"
    box_radius = env_cfg.get("box_radius", 2)
    n = 2 * box_radius + 1

    grids: list[np.ndarray] = []
    collected = 0

    while collected < n_samples:
        obs, _ = env.reset()
        done = False
        while not done and collected < n_samples:
            action = {a: env.action_space(a).sample() for a in env.agents}
            obs_next, _, terms, truncs, _ = env.step(action)
            done = all(terms.values()) or all(truncs.values())

            for a in env.possible_agents:
                if a in obs_next and "local_grid" in obs_next[a]:
                    g = np.array(obs_next[a]["local_grid"], dtype=np.float32).reshape(n, n, n)
                    grids.append(g)
                    collected += 1
                    if progress_cb:
                        progress_cb(1)
                    if collected >= n_samples:
                        break
            obs = obs_next

    return np.stack(grids[:n_samples])


# ---------------------------------------------------------------------------
# Position-sample collector
# ---------------------------------------------------------------------------

def _split_by_fill(obs: np.ndarray, min_fill_pct: float) -> tuple[np.ndarray, np.ndarray]:
    """Split an (N, n, n, n) array into (nonempty, empty) by fill threshold."""
    fills = (obs.reshape(len(obs), -1) > 0.5).mean(axis=1) * 100.0
    mask = fills >= min_fill_pct
    return obs[mask], obs[~mask]


class PositionSampleCollector:
    """Collect garden observations by sampling positions from geometry sources.

    Parameters
    ----------
    source : PositionSampleSourceConfig
        Source definition describing how geometry observations are collected.
    cache_cfg : CacheConfig
        Cache settings used for observation reuse.
    box_radius : int
        Radius of the local observation cube to sample.
    min_fill_pct : float
        Minimum occupied percentage required for a sample to count as non-empty.
    empty_sample_ratio : float
        Fraction of the final dataset reserved for empty samples.
    """

    def __init__(
        self,
        source: PositionSampleSourceConfig,
        cache_cfg: CacheConfig,
        box_radius: int = 2,
        min_fill_pct: float = 0.0,
        empty_sample_ratio: float = 0.05,
    ) -> None:
        self.source = source
        self.cache_cfg = cache_cfg
        self.box_radius = box_radius
        self.min_fill_pct = min_fill_pct
        self.empty_sample_ratio = empty_sample_ratio

    def collect(self, progress_cb: Callable[[int], None] | None = None) -> np.ndarray:
        cache_path = _obs_cache_path(self.cache_cfg, self.source)
        if not self.cache_cfg.force_refresh:
            cached = _load_cached(cache_path)
            if cached is not None:
                log.debug("Loaded %d samples from cache: %s", len(cached), cache_path)
                if progress_cb:
                    progress_cb(len(cached))
                return cached

        geo = self.source.geometry
        grids = self._collect_geometry(geo, progress_cb)

        _save_cached(cache_path, grids, {"source": self.source.model_dump(mode="json")})
        return grids

    def _collect_geometry(self, geo, progress_cb) -> np.ndarray:
        if geo.env is not None:
            n_samples = self.source.num_samples or 10000
            env_cfg = _build_env_config(geo.env, self.box_radius)
            return _collect_from_env(env_cfg, n_samples, progress_cb)

        sampler = geo.sampler
        if isinstance(sampler, StlListSamplerConfig):
            return self._collect_stl_list(sampler, progress_cb)
        if isinstance(sampler, RandomBoxesSamplerConfig):
            return self._collect_random_boxes(sampler, progress_cb)
        raise ValueError("geometry must specify env or sampler")

    def _mix(self, nonempty: np.ndarray, empty: np.ndarray, total: int) -> np.ndarray:
        """Return *total* samples: (1-empty_ratio) non-empty + empty_ratio empty."""
        n_empty = min(int(total * self.empty_sample_ratio), len(empty))
        n_full = min(total - n_empty, len(nonempty))
        parts = [nonempty[:n_full]]
        if n_empty > 0:
            parts.append(empty[:n_empty])
        return np.concatenate(parts)

    def _collect_stl_list(self, sampler: StlListSamplerConfig, progress_cb) -> np.ndarray:
        from theseo_core import PyVoxelSampler

        paths = sampler.resolve_paths()
        if not paths:
            raise ValueError("stl_list sampler found no .stl files")
        per_file = sampler.total_samples // len(paths)
        rng = np.random.default_rng(sampler.seed)
        n = 2 * self.box_radius + 1
        all_grids: list[np.ndarray] = []

        for stl_path in paths:
            try:
                grid_size = 32
                scale = _stl_fit_scale(str(Path(stl_path).resolve()), grid_size)
                sampler_obj = PyVoxelSampler(grid_size=grid_size)
                sampler_obj.load_stl(str(Path(stl_path).resolve()), scale)
                free = sampler_obj.free_cells()
                if not free:
                    log.warning("No free cells in %s — skipping", stl_path)
                    continue

                # Sample all available positions (or oversample with replacement)
                seed = int(rng.integers(0, 2**31))
                cell_rng = np.random.default_rng(seed)
                n_draw = max(per_file * 4, len(free))  # oversample to ensure enough non-empty
                chosen = cell_rng.choice(len(free), size=min(n_draw, len(free)), replace=False)
                positions = [free[i] for i in chosen]

                raw = sampler_obj.sample_box_obs(positions, self.box_radius)
                obs = np.array(raw, dtype=np.float32).reshape(len(positions), n, n, n)

                nonempty, empty = _split_by_fill(obs, self.min_fill_pct)
                mixed = self._mix(nonempty, empty, per_file)
                all_grids.append(mixed)
                if progress_cb:
                    progress_cb(len(mixed))
            except Exception as exc:
                log.warning("Failed to collect from %s: %s", stl_path, exc)

        return np.concatenate(all_grids) if all_grids else np.empty((0, n, n, n), dtype=np.float32)

    def _collect_random_boxes(self, sampler: RandomBoxesSamplerConfig, progress_cb) -> np.ndarray:
        rng = np.random.default_rng(sampler.seed)
        per_geo = sampler.total_samples // sampler.num_geometries
        n = 2 * self.box_radius + 1
        all_grids: list[np.ndarray] = []

        for i in range(sampler.num_geometries):
            env_cfg = _random_boxes_geometry(sampler, i, rng)
            env_cfg["box_radius"] = self.box_radius
            obs = _collect_from_env(env_cfg, per_geo * 4, progress_cb)  # oversample
            nonempty, empty = _split_by_fill(obs, self.min_fill_pct)
            mixed = self._mix(nonempty, empty, per_geo)
            all_grids.append(mixed)

        return np.concatenate(all_grids) if all_grids else np.empty((0, n, n, n), dtype=np.float32)


# ---------------------------------------------------------------------------
# Training-run collector
# ---------------------------------------------------------------------------

class TrainingRunCollector:
    """Extract box_obs at saved cursor positions from trajectory JSON files."""

    def __init__(self, source: TrainingRunSourceConfig, cache_cfg: CacheConfig, box_radius: int = 2) -> None:
        self.source = source
        self.cache_cfg = cache_cfg
        self.box_radius = box_radius

    def collect(self, run_dir: Path, progress_cb: Callable[[int], None] | None = None) -> np.ndarray:
        cache_path = _obs_cache_path(self.cache_cfg, self.source)
        if not self.cache_cfg.force_refresh:
            cached = _load_cached(cache_path)
            if cached is not None:
                if progress_cb:
                    progress_cb(len(cached))
                return cached

        grids = self._extract_from_trajectories(run_dir, progress_cb)
        _save_cached(cache_path, grids, {"source": self.source.model_dump(mode="json"), "run_dir": str(run_dir)})
        return grids

    def _extract_from_trajectories(self, run_dir: Path, progress_cb) -> np.ndarray:
        traj_dir = run_dir / "trajectories"
        json_files = sorted(traj_dir.glob("*.json")) if traj_dir.exists() else []
        if not json_files:
            # Walk subdirectories (tune runs have trial subdirs)
            json_files = sorted(run_dir.rglob("trajectories/*.json"))

        n = 2 * self.box_radius + 1
        all_grids: list[np.ndarray] = []

        for jf in json_files:
            if jf.stem.endswith("_meta"):
                continue
            try:
                data = json.loads(jf.read_text())
                ep = data.get("episode", data)
                steps = ep.get("steps", [])
                for step in steps:
                    lg = step.get(self.source.obs_key)
                    if lg is not None:
                        g = np.array(lg, dtype=np.float32).reshape(n, n, n)
                        all_grids.append(g)
                        if progress_cb:
                            progress_cb(1)
            except Exception as exc:
                log.warning("Skipping %s: %s", jf, exc)

        if not all_grids:
            return np.empty((0, n, n, n), dtype=np.float32)
        return np.stack(all_grids)


# ---------------------------------------------------------------------------
# Top-level: collect all sources from a DataConfig
# ---------------------------------------------------------------------------

def collect_dataset(
    data_cfg: DataConfig,
    box_radius: int = 2,
    run_registry: dict[str, Path] | None = None,
    progress_cb: Callable[[str, int], None] | None = None,
) -> np.ndarray:
    """Collect and cache all sources. Returns concatenated (M, n, n, n) array."""
    all_grids: list[np.ndarray] = []

    for i, source in enumerate(data_cfg.sources):
        label = f"source {i + 1}"
        cb = (lambda lbl: lambda n: progress_cb(lbl, n))(label) if progress_cb else None

        if isinstance(source, PositionSampleSourceConfig):
            collector = PositionSampleCollector(
                source, data_cfg.cache, box_radius,
                min_fill_pct=data_cfg.min_fill_pct,
                empty_sample_ratio=data_cfg.empty_sample_ratio,
            )
            grids = collector.collect(cb)

        elif isinstance(source, TrainingRunSourceConfig):
            run_dir: Path | None = None
            if run_registry:
                run_dir = run_registry.get(source.run)
            if run_dir is None:
                # Try loading from the runs registry file
                runs_file = Path(data_cfg.cache.path) / "runs.json"
                if runs_file.exists():
                    registry = json.loads(runs_file.read_text())
                    run_dir = Path(registry[source.run]) if source.run in registry else None
            if run_dir is None:
                log.warning("training_run %r not registered — skipping. Use: anysearch garden add %s", source.run, source.run)
                continue
            collector = TrainingRunCollector(source, data_cfg.cache, box_radius)
            grids = collector.collect(run_dir, cb)

        else:
            log.warning("Unknown source type — skipping")
            continue

        if len(grids) > 0:
            all_grids.append(grids)
            log.info("Source %d: %d samples", i + 1, len(grids))

    if not all_grids:
        raise RuntimeError("No observations collected. Check your data sources.")

    return np.concatenate(all_grids)
