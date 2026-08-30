"""Deterministic pre-flight compiler for finite voxel worlds."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import statistics
import time
import uuid
import zlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.cache_lock import cache_key_lock
from theseo_anysearch.worlds.candidates import (
    CANDIDATE_DATA_FILE,
    CANDIDATE_INDEX_FILE,
    CandidateIndexHandle,
    CandidateRecord,
    write_candidate_index,
)
from theseo_anysearch.worlds.manifest import (
    ChunkCoordinate,
    WorldChunkManifest,
    WorldExtent,
    WorldManifest,
)

PACK_FILE = "world.pack"
INDEX_FILE = "index.json"
MANIFEST_FILE = "manifest.json"
COMPLETE_FILE = "COMPLETE"
COMPILER_SCHEMA_VERSION = 2
_CHUNK_MAGIC = b"AWC1"
_ENCODING_IDS = {"uniform": 1, "sparse_u32": 2, "dense_zlib": 3}


class WorldPackError(RuntimeError):
    """Base class for invalid or unavailable compiled worlds."""


class WorldPackCorruptError(WorldPackError):
    """A published pack failed an integrity check."""


class WorldPackUnavailableError(WorldPackError):
    """A pack is invalid and its source is no longer available."""


class WorldCompilerConfig(BaseModel):
    """Settings that participate in compiled-world identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_shape: tuple[int, int, int] = (32, 32, 32)
    sparse_max_fraction: float = Field(default=0.125, gt=0.0, lt=1.0)
    compression_level: int = Field(default=6, ge=0, le=9)


@dataclass(frozen=True)
class BoxSource:
    """Inclusive zero-based bounds for an axis-aligned occupied box."""

    minimum: tuple[int, int, int]
    maximum_inclusive: tuple[int, int, int]


@dataclass(frozen=True)
class NpySource:
    """A uint8/bool occupancy grid stored as an ``.npy`` file."""

    path: Path


@dataclass(frozen=True)
class StlSource:
    """ASCII STL geometry voxelized by the repository's native sampler."""

    path: Path
    scale: float
    padding: int = 2


WorldSource = BoxSource | NpySource | StlSource


@dataclass(frozen=True)
class CompiledWorld:
    """Validated paths and metadata for one immutable pack."""

    root: Path
    manifest: WorldManifest

    @property
    def pack_path(self) -> Path:
        return self.root.joinpath(PACK_FILE)

    @property
    def index_path(self) -> Path:
        return self.root.joinpath(INDEX_FILE)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_contract(source: WorldSource) -> dict[str, Any]:
    if isinstance(source, BoxSource):
        return {
            "type": "box",
            "minimum": list(source.minimum),
            "maximum_inclusive": list(source.maximum_inclusive),
        }
    path = source.path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    contract = {
        "type": "npy",
        "sha256": _sha256_file(path),
        "suffix": path.suffix.lower(),
    }
    if isinstance(source, StlSource):
        contract.update(
            {"type": "stl", "scale": source.scale, "padding": source.padding}
        )
    return contract


def compiler_identity(
    sources: Sequence[WorldSource],
    extent: WorldExtent,
    config: WorldCompilerConfig,
) -> tuple[str, dict[str, Any]]:
    """Return the content identity without embedding machine-local paths."""

    source_contracts = [_source_contract(source) for source in sources]
    source_contracts.sort(key=_canonical_json)
    contract = {
        "compiler_schema_version": COMPILER_SCHEMA_VERSION,
        "world_schema_version": 1,
        "coordinate_type": "u32",
        "extent": list(extent.as_tuple()),
        "config": config.model_dump(mode="json"),
        "sources": source_contracts,
    }
    return _sha256_bytes(_canonical_json(contract)), contract


def _chunk_extent(
    key: tuple[int, int, int], extent: WorldExtent, shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    return tuple(
        min(shape[axis], extent.as_tuple()[axis] - key[axis] * shape[axis])
        for axis in range(3)
    )  # type: ignore[return-value]


def _chunk_for(
    chunks: dict[tuple[int, int, int], np.ndarray],
    key: tuple[int, int, int],
    extent: WorldExtent,
    shape: tuple[int, int, int],
) -> np.ndarray:
    return chunks.setdefault(
        key, np.zeros(_chunk_extent(key, extent, shape), dtype=np.bool_)
    )


def _add_box(
    chunks: dict[tuple[int, int, int], np.ndarray],
    source: BoxSource,
    extent: WorldExtent,
    shape: tuple[int, int, int],
) -> None:
    world_shape = extent.as_tuple()
    if any(source.minimum[axis] < 0 for axis in range(3)) or any(
        source.maximum_inclusive[axis] >= world_shape[axis]
        or source.maximum_inclusive[axis] < source.minimum[axis]
        for axis in range(3)
    ):
        raise ValueError("box bounds must be ordered and inside the world extent")
    first = tuple(source.minimum[axis] // shape[axis] for axis in range(3))
    last = tuple(source.maximum_inclusive[axis] // shape[axis] for axis in range(3))
    for cx in range(first[0], last[0] + 1):
        for cy in range(first[1], last[1] + 1):
            for cz in range(first[2], last[2] + 1):
                key = (cx, cy, cz)
                chunk = _chunk_for(chunks, key, extent, shape)
                starts = tuple(
                    max(source.minimum[a] - key[a] * shape[a], 0) for a in range(3)
                )
                stops = tuple(
                    min(
                        source.maximum_inclusive[a] - key[a] * shape[a] + 1,
                        chunk.shape[a],
                    )
                    for a in range(3)
                )
                chunk[
                    starts[0] : stops[0],
                    starts[1] : stops[1],
                    starts[2] : stops[2],
                ] = True


def _add_npy(
    chunks: dict[tuple[int, int, int], np.ndarray],
    source: NpySource,
    extent: WorldExtent,
    shape: tuple[int, int, int],
) -> None:
    grid = np.load(source.path, mmap_mode="r", allow_pickle=False)
    if grid.ndim != 3 or tuple(grid.shape) != extent.as_tuple():
        raise ValueError(f"occupancy grid {source.path} shape does not match extent")
    counts = tuple((grid.shape[a] + shape[a] - 1) // shape[a] for a in range(3))
    for cx in range(counts[0]):
        for cy in range(counts[1]):
            for cz in range(counts[2]):
                starts = (cx * shape[0], cy * shape[1], cz * shape[2])
                view = grid[
                    starts[0] : starts[0] + shape[0],
                    starts[1] : starts[1] + shape[1],
                    starts[2] : starts[2] + shape[2],
                ]
                if np.any(view):
                    _chunk_for(chunks, (cx, cy, cz), extent, shape)[:] |= np.asarray(
                        view, dtype=np.bool_
                    )


def _add_stl(
    chunks: dict[tuple[int, int, int], np.ndarray],
    source: StlSource,
    extent: WorldExtent,
    shape: tuple[int, int, int],
) -> None:
    if len(set(extent.as_tuple())) != 1:
        raise ValueError("the current STL voxelizer requires a cubic world extent")
    cells = _load_stl_cells(source.path, source.scale, extent.x, source.padding)
    for environment_coord in cells:
        coordinate = tuple(axis - 1 for axis in environment_coord)
        key = tuple(coordinate[axis] // shape[axis] for axis in range(3))
        local = tuple(coordinate[axis] % shape[axis] for axis in range(3))
        _chunk_for(chunks, key, extent, shape)[local] = True


def _load_stl_cells(
    path: Path, scale: float, grid_size: int, padding: int
) -> list[tuple[int, int, int]]:
    """Voxelize without importing the heavyweight RL environment modules."""

    vertices: list[tuple[float, float, float]] = []
    with path.open(encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            parts = line.strip().split()
            if len(parts) >= 4 and parts[0] == "vertex":
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if vertices:
        values = np.asarray(vertices, dtype=np.float64)
        minima = values.min(axis=0)
        max_extent = float((values.max(axis=0) - minima).max()) or 1.0
    else:
        max_extent = 1.0
    max_span = grid_size - 2 * padding - 1
    if max_span < 1:
        raise ValueError("STL padding leaves no voxelizable world interior")
    import theseo_core

    sampler = theseo_core.PyVoxelSampler(grid_size=grid_size)
    origin = float(padding + 1)
    sampler.load_stl_normalized(
        str(path),
        min(float(scale), float(max_span)) / max_extent,
        origin,
        origin,
        origin,
    )
    free = set(sampler.free_cells())
    lower, upper = padding + 1, grid_size - padding
    return [
        (x, y, z)
        for x in range(lower, upper + 1)
        for y in range(lower, upper + 1)
        for z in range(lower, upper + 1)
        if (x, y, z) not in free
    ]


def _encode_chunk(
    chunk: np.ndarray, config: WorldCompilerConfig
) -> tuple[str, bytes, int]:
    flat = np.ravel(chunk, order="C")
    occupied = int(np.count_nonzero(flat))
    if occupied == flat.size:
        return "uniform", _CHUNK_MAGIC + bytes([_ENCODING_IDS["uniform"]]), flat.size
    indices = np.flatnonzero(flat).astype("<u4", copy=False)
    sparse = _CHUNK_MAGIC + bytes([_ENCODING_IDS["sparse_u32"]]) + indices.tobytes()
    dense_raw = np.packbits(flat, bitorder="little").tobytes()
    dense = (
        _CHUNK_MAGIC
        + bytes([_ENCODING_IDS["dense_zlib"]])
        + zlib.compress(dense_raw, level=config.compression_level)
    )
    if occupied / flat.size <= config.sparse_max_fraction and len(sparse) <= len(dense):
        return "sparse_u32", sparse, len(dense_raw)
    return "dense_zlib", dense, len(dense_raw)


def decode_chunk(payload: bytes, shape: tuple[int, int, int]) -> np.ndarray:
    """Decode and validate one independently addressable chunk payload."""

    if len(payload) < 5 or payload[:4] != _CHUNK_MAGIC:
        raise WorldPackCorruptError("invalid chunk header")
    size = shape[0] * shape[1] * shape[2]
    encoding = payload[4]
    if encoding == _ENCODING_IDS["uniform"]:
        if len(payload) != 5:
            raise WorldPackCorruptError("uniform chunk has trailing data")
        flat = np.ones(size, dtype=np.bool_)
    elif encoding == _ENCODING_IDS["sparse_u32"]:
        raw = payload[5:]
        if len(raw) % 4:
            raise WorldPackCorruptError("invalid sparse chunk length")
        indices = np.frombuffer(raw, dtype="<u4")
        if indices.size and int(indices[-1]) >= size:
            raise WorldPackCorruptError("sparse chunk index exceeds chunk shape")
        flat = np.zeros(size, dtype=np.bool_)
        flat[indices] = True
    elif encoding == _ENCODING_IDS["dense_zlib"]:
        try:
            raw = zlib.decompress(payload[5:])
        except zlib.error as error:
            raise WorldPackCorruptError("invalid dense chunk payload") from error
        expected = (size + 7) // 8
        if len(raw) != expected:
            raise WorldPackCorruptError("dense chunk has incorrect decoded length")
        flat = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[
            :size
        ].astype(np.bool_)
    else:
        raise WorldPackCorruptError("unsupported chunk encoding")
    return flat.reshape(shape, order="C")


def benchmark_encodings(
    chunks: Sequence[np.ndarray],
    config: WorldCompilerConfig | None = None,
    *,
    repeats: int = 5,
) -> list[dict[str, float | int | str]]:
    """Measure median encode/decode throughput and compression per chunk."""

    if repeats < 1:
        raise ValueError("benchmark repeats must be positive")
    resolved = config or WorldCompilerConfig()
    results: list[dict[str, float | int | str]] = []
    for chunk in chunks:
        if chunk.ndim != 3 or chunk.size == 0:
            raise ValueError(
                "benchmark chunks must be non-empty three-dimensional arrays"
            )
        boolean = np.asarray(chunk, dtype=np.bool_)
        encode_ns: list[int] = []
        decode_ns: list[int] = []
        encoding = ""
        payload = b""
        for _ in range(repeats):
            started = time.perf_counter_ns()
            encoding, payload, _ = _encode_chunk(boolean, resolved)
            encode_ns.append(time.perf_counter_ns() - started)
            started = time.perf_counter_ns()
            decode_chunk(payload, tuple(boolean.shape))
            decode_ns.append(time.perf_counter_ns() - started)
        dense_bytes = int(boolean.size)
        results.append(
            {
                "voxels": dense_bytes,
                "occupied_voxels": int(np.count_nonzero(boolean)),
                "encoding": encoding,
                "encoded_bytes": len(payload),
                "compression_ratio": dense_bytes / len(payload),
                "encode_voxels_per_second": dense_bytes
                / (statistics.median(encode_ns) / 1e9),
                "decode_voxels_per_second": dense_bytes
                / (statistics.median(decode_ns) / 1e9),
            }
        )
    return results


def _write_pack(
    root: Path,
    chunks: dict[tuple[int, int, int], np.ndarray],
    extent: WorldExtent,
    config: WorldCompilerConfig,
    identity: str,
    contract: dict[str, Any],
) -> None:
    root.mkdir(parents=True)
    entries: list[WorldChunkManifest] = []
    index: dict[str, dict[str, Any]] = {}
    offset = 0
    pack_path = root.joinpath(PACK_FILE)
    with pack_path.open("wb") as pack:
        for key in sorted(chunks):
            chunk = chunks[key]
            occupied = int(np.count_nonzero(chunk))
            if occupied == 0:
                continue
            encoding, payload, decoded_length = _encode_chunk(chunk, config)
            pack.write(payload)
            checksum = _sha256_bytes(payload)
            entry = WorldChunkManifest(
                coordinate=ChunkCoordinate(x=key[0], y=key[1], z=key[2]),
                relative_path=PACK_FILE,
                sha256=checksum,
                byte_length=len(payload),
                occupied_voxels=occupied,
                encoding=encoding,
                pack_offset=offset,
                decoded_byte_length=decoded_length,
            )
            entries.append(entry)
            index[",".join(str(axis) for axis in key)] = {
                "offset": offset,
                "byte_length": len(payload),
                "sha256": checksum,
                "encoding": encoding,
                "shape": list(chunk.shape),
                "occupied_voxels": occupied,
            }
            offset += len(payload)
        pack.flush()
        os.fsync(pack.fileno())
    pack_sha = _sha256_file(pack_path)
    source_sha = _sha256_bytes(_canonical_json(contract["sources"]))
    manifest = WorldManifest(
        extent=extent,
        chunk_shape=WorldExtent.from_value(config.chunk_shape),
        chunks=tuple(entries),
        identity_sha256=identity,
        source_sha256=source_sha,
        compiler={
            "schema_version": COMPILER_SCHEMA_VERSION,
            "config": config.model_dump(mode="json"),
            "identity_contract": contract,
        },
        pack_sha256=pack_sha,
    )
    root.joinpath(INDEX_FILE).write_bytes(_canonical_json(index))
    root.joinpath(MANIFEST_FILE).write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )


def _candidate_records(
    chunks: dict[tuple[int, int, int], np.ndarray],
    extent: WorldExtent,
    chunk_shape: tuple[int, int, int],
) -> list[CandidateRecord]:
    """Derive sparse surface and adjacent free-space candidates pre-flight."""

    occupied: set[tuple[int, int, int]] = set()
    for chunk_key, chunk in chunks.items():
        origin = tuple(chunk_key[axis] * chunk_shape[axis] for axis in range(3))
        for local in np.argwhere(chunk):
            occupied.add(tuple(origin[axis] + int(local[axis]) for axis in range(3)))
    limits = extent.as_tuple()
    directions = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    free: set[tuple[int, int, int]] = set()
    surface: set[tuple[int, int, int]] = set()
    for coordinate in occupied:
        for direction in directions:
            neighbor = tuple(coordinate[a] + direction[a] for a in range(3))
            if (
                all(0 <= neighbor[a] < limits[a] for a in range(3))
                and neighbor not in occupied
            ):
                free.add(neighbor)
                surface.add(coordinate)
    records: list[CandidateRecord] = []
    for coordinate in sorted(free):
        open_neighbors = 0
        total_neighbors = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if (dx, dy, dz) == (0, 0, 0):
                        continue
                    neighbor = (
                        coordinate[0] + dx,
                        coordinate[1] + dy,
                        coordinate[2] + dz,
                    )
                    if all(0 <= neighbor[a] < limits[a] for a in range(3)):
                        total_neighbors += 1
                        open_neighbors += int(neighbor not in occupied)
        quality = open_neighbors / max(total_neighbors, 1)
        public = tuple(value + 1 for value in coordinate)
        region = tuple(coordinate[a] // chunk_shape[a] for a in range(3))
        records.extend(
            CandidateRecord(position=public, kind=kind, quality=quality, region=region)
            for kind in ("spawn", "goal")
        )
    for coordinate in sorted(surface):
        records.append(
            CandidateRecord(
                position=tuple(value + 1 for value in coordinate),
                kind="surface",
                quality=1.0,
                region=tuple(coordinate[a] // chunk_shape[a] for a in range(3)),
            )
        )
    return records


def validate_compiled_world(root: Path, *, verify_chunks: bool = True) -> CompiledWorld:
    """Open a pack only after its completion marker and checksums validate."""

    required = [
        root.joinpath(name)
        for name in (
            PACK_FILE,
            INDEX_FILE,
            MANIFEST_FILE,
            COMPLETE_FILE,
            CANDIDATE_DATA_FILE,
            CANDIDATE_INDEX_FILE,
        )
    ]
    if not all(path.is_file() for path in required):
        raise WorldPackCorruptError("compiled world is incomplete")
    try:
        manifest = WorldManifest.model_validate_json(
            root.joinpath(MANIFEST_FILE).read_text(encoding="utf-8")
        )
        index = json.loads(root.joinpath(INDEX_FILE).read_text(encoding="utf-8"))
    except (ValueError, OSError) as error:
        raise WorldPackCorruptError("compiled world metadata is invalid") from error
    if (
        root.joinpath(COMPLETE_FILE).read_text(encoding="ascii")
        != manifest.identity_sha256
    ):
        raise WorldPackCorruptError("completion marker does not match world identity")
    identity_contract = manifest.compiler.get("identity_contract")
    if (
        not isinstance(identity_contract, dict)
        or _sha256_bytes(_canonical_json(identity_contract)) != manifest.identity_sha256
    ):
        raise WorldPackCorruptError(
            "manifest identity contract does not match world identity"
        )
    if _sha256_file(root.joinpath(PACK_FILE)) != manifest.pack_sha256:
        raise WorldPackCorruptError("world pack checksum mismatch")
    try:
        CandidateIndexHandle(root, world_identity=manifest.identity_sha256)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise WorldPackCorruptError("candidate index is invalid") from error
    expected_keys = {
        f"{chunk.coordinate.x},{chunk.coordinate.y},{chunk.coordinate.z}"
        for chunk in manifest.chunks
    }
    if set(index) != expected_keys:
        raise WorldPackCorruptError("chunk index keys do not match the manifest")
    with root.joinpath(PACK_FILE).open("rb") as pack:
        for chunk in manifest.chunks:
            key = f"{chunk.coordinate.x},{chunk.coordinate.y},{chunk.coordinate.z}"
            indexed = index.get(key)
            expected_index = {
                "offset": chunk.pack_offset,
                "byte_length": chunk.byte_length,
                "sha256": chunk.sha256,
                "encoding": chunk.encoding,
                "shape": list(
                    _chunk_extent(
                        (
                            chunk.coordinate.x,
                            chunk.coordinate.y,
                            chunk.coordinate.z,
                        ),
                        manifest.extent,
                        manifest.chunk_shape.as_tuple(),
                    )
                ),
                "occupied_voxels": chunk.occupied_voxels,
            }
            if indexed != expected_index:
                raise WorldPackCorruptError(f"chunk index mismatch for {key}")
            if verify_chunks:
                pack.seek(chunk.pack_offset)
                payload = pack.read(chunk.byte_length)
                if (
                    len(payload) != chunk.byte_length
                    or _sha256_bytes(payload) != chunk.sha256
                ):
                    raise WorldPackCorruptError(f"chunk checksum mismatch for {key}")
                decode_chunk(payload, tuple(indexed["shape"]))
    return CompiledWorld(root=root, manifest=manifest)


def load_compiled_world(cache_dir: Path, identity: str) -> CompiledWorld:
    """Load a known pack identity when original source files are unavailable."""

    root = cache_dir.joinpath(identity)
    try:
        return validate_compiled_world(root)
    except WorldPackCorruptError as error:
        message = (
            f"world pack {identity} is unavailable or invalid; "
            "source data is required to rebuild it"
        )
        raise WorldPackUnavailableError(message) from error


def compile_world(
    sources: Iterable[WorldSource],
    extent: WorldExtent,
    cache_dir: Path,
    config: WorldCompilerConfig | None = None,
    *,
    lock_timeout_seconds: float = 300.0,
) -> CompiledWorld:
    """Compile or reuse one content-addressed immutable world pack."""

    resolved_config = config or WorldCompilerConfig()
    source_list = tuple(sources)
    identity, contract = compiler_identity(source_list, extent, resolved_config)
    entry = cache_dir.joinpath(identity)
    with cache_key_lock(cache_dir, identity, lock_timeout_seconds, label="world pack"):
        if entry.exists():
            try:
                return validate_compiled_world(entry)
            except WorldPackCorruptError:
                if not all(_source_available(source) for source in source_list):
                    raise WorldPackUnavailableError(
                        f"world pack {identity} is invalid and source data is unavailable"
                    ) from None
                quarantine = cache_dir.joinpath(
                    f".{identity}.{uuid.uuid4().hex}.invalid"
                )
                os.replace(entry, quarantine)
                shutil.rmtree(quarantine)
        temporary = cache_dir.joinpath(f".{identity}.{uuid.uuid4().hex}.tmp")
        chunks: dict[tuple[int, int, int], np.ndarray] = {}
        try:
            shape = resolved_config.chunk_shape
            WorldExtent.from_value(shape)
            for source in source_list:
                if isinstance(source, BoxSource):
                    _add_box(chunks, source, extent, shape)
                elif isinstance(source, NpySource):
                    _add_npy(chunks, source, extent, shape)
                else:
                    _add_stl(chunks, source, extent, shape)
            _write_pack(temporary, chunks, extent, resolved_config, identity, contract)
            write_candidate_index(
                temporary,
                identity,
                _candidate_records(chunks, extent, resolved_config.chunk_shape),
            )
            temporary.joinpath(COMPLETE_FILE).write_text(identity, encoding="ascii")
            validate_compiled_world(temporary)
            os.replace(temporary, entry)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        return validate_compiled_world(entry)


def _source_available(source: WorldSource) -> bool:
    return isinstance(source, BoxSource) or source.path.is_file()


def compile_pool(
    pool_dir: Path,
    cache_dir: Path,
    config: WorldCompilerConfig | None = None,
    *,
    lock_timeout_seconds: float = 300.0,
) -> Iterator[CompiledWorld]:
    """Compile pool variants independently without materializing coordinate tuples."""

    for path in sorted(pool_dir.rglob("*.npy")):
        grid = np.load(path, mmap_mode="r", allow_pickle=False)
        if grid.ndim != 3:
            raise ValueError(f"pool entry {path} is not a three-dimensional grid")
        yield compile_world(
            [NpySource(path)],
            WorldExtent.from_value(tuple(int(axis) for axis in grid.shape)),
            cache_dir,
            config,
            lock_timeout_seconds=lock_timeout_seconds,
        )
