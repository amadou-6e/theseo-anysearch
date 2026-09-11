from __future__ import annotations

import hashlib
import json
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from theseo_anysearch.worlds.candidates import CandidateIndexHandle
from theseo_anysearch.worlds.compiler import (
    COMPLETE_FILE,
    MANIFEST_FILE,
    PACK_FILE,
    BoxSource,
    NpySource,
    StlSource,
    WorldCompilerConfig,
    WorldPackCorruptError,
    WorldPackUnavailableError,
    benchmark_encodings,
    compile_pool,
    compile_world,
    decode_chunk,
    load_compiled_world,
    validate_compiled_world,
)
from theseo_anysearch.worlds.manifest import WorldExtent
from theseo_anysearch.worlds.overview import decode_overview_mesh


def _read_chunks(root: Path) -> dict[tuple[int, int, int], np.ndarray]:
    index = json.loads(root.joinpath("index.json").read_text(encoding="utf-8"))
    decoded: dict[tuple[int, int, int], np.ndarray] = {}
    with root.joinpath(PACK_FILE).open("rb") as pack:
        for raw_key, entry in index.items():
            pack.seek(entry["offset"])
            payload = pack.read(entry["byte_length"])
            decoded[tuple(int(axis) for axis in raw_key.split(","))] = decode_chunk(
                payload, tuple(entry["shape"])
            )
    return decoded


def test_compiler_emits_sparse_surface_and_free_candidates(tmp_path: Path) -> None:
    compiled = compile_world(
        [BoxSource(minimum=(2, 2, 2), maximum_inclusive=(2, 2, 2))],
        WorldExtent(x=8, y=8, z=8),
        tmp_path,
    )
    candidates = CandidateIndexHandle(
        compiled.root, world_identity=compiled.manifest.identity_sha256
    )
    assert len(candidates.sample(20, "spawn", seed=1, stream=1)) == 6
    assert {
        item.position for item in candidates.sample(20, "surface", seed=1, stream=2)
    } == {(3, 3, 3)}
    assert candidates.sample(1, "portal", seed=1, stream=3) == ()


def test_boxes_compile_without_coordinate_tuple_expansion(tmp_path: Path) -> None:
    compiled = compile_world(
        [BoxSource((1, 2, 3), (6, 5, 4))],
        WorldExtent(x=8, y=8, z=8),
        tmp_path,
        WorldCompilerConfig(chunk_shape=(4, 4, 4)),
    )

    chunks = _read_chunks(compiled.root)
    assert sum(int(chunk.sum()) for chunk in chunks.values()) == 6 * 4 * 2
    assert compiled.manifest.chunks
    assert all(chunk.relative_path == PACK_FILE for chunk in compiled.manifest.chunks)


def test_identical_inputs_have_identical_identity_and_bytes(tmp_path: Path) -> None:
    source = BoxSource((0, 0, 0), (3, 3, 3))
    first = compile_world([source], WorldExtent(x=8, y=8, z=8), tmp_path.joinpath("a"))
    second = compile_world([source], WorldExtent(x=8, y=8, z=8), tmp_path.joinpath("b"))

    assert first.manifest.identity_sha256 == second.manifest.identity_sha256
    assert first.pack_path.read_bytes() == second.pack_path.read_bytes()
    assert first.index_path.read_bytes() == second.index_path.read_bytes()
    assert first.overview_path is not None
    assert second.overview_path is not None
    assert first.overview_path.read_bytes() == second.overview_path.read_bytes()


def test_compiler_emits_validated_bounded_overview_mesh(tmp_path: Path) -> None:
    compiled = compile_world(
        [BoxSource((1, 1, 1), (6, 5, 4))],
        WorldExtent(x=8, y=8, z=8),
        tmp_path,
    )

    assert compiled.manifest.overview is not None
    assert compiled.overview_path is not None
    mesh = decode_overview_mesh(compiled.overview_path.read_bytes())
    assert len(mesh.vertices) == compiled.manifest.overview.vertex_count
    assert mesh.triangle_count == compiled.manifest.overview.triangle_count
    assert mesh.triangle_count <= 10_000


def test_overview_corruption_is_detected(tmp_path: Path) -> None:
    compiled = compile_world(
        [BoxSource((0, 0, 0), (3, 3, 3))],
        WorldExtent(x=8, y=8, z=8),
        tmp_path,
    )
    assert compiled.overview_path is not None
    compiled.overview_path.write_bytes(b"corrupt")

    with pytest.raises(WorldPackCorruptError, match="overview mesh checksum mismatch"):
        validate_compiled_world(compiled.root)


def test_structurally_invalid_overview_is_detected_after_checksum(
    tmp_path: Path,
) -> None:
    compiled = compile_world(
        [BoxSource((0, 0, 0), (3, 3, 3))],
        WorldExtent(x=8, y=8, z=8),
        tmp_path,
    )
    assert compiled.overview_path is not None
    payload = struct.pack("<4sIII3I", b"AOM1", 1, 0, 3, 0, 0, 0)
    compiled.overview_path.write_bytes(payload)
    manifest_path = compiled.root.joinpath(MANIFEST_FILE)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["overview"].update(
        {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_length": len(payload),
            "vertex_count": 0,
            "triangle_count": 1,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(WorldPackCorruptError, match="overview mesh is invalid"):
        validate_compiled_world(compiled.root)


def test_source_order_does_not_change_identity(tmp_path: Path) -> None:
    sources = [BoxSource((0, 0, 0), (0, 0, 0)), BoxSource((7, 7, 7), (7, 7, 7))]
    first = compile_world(sources, WorldExtent(x=8, y=8, z=8), tmp_path.joinpath("a"))
    second = compile_world(
        list(reversed(sources)), WorldExtent(x=8, y=8, z=8), tmp_path.joinpath("b")
    )

    assert first.manifest.identity_sha256 == second.manifest.identity_sha256
    assert first.pack_path.read_bytes() == second.pack_path.read_bytes()


def test_pool_compiles_each_grid_without_grid_to_cells(tmp_path: Path) -> None:
    pool = tmp_path.joinpath("pool")
    pool.mkdir()
    for index in range(2):
        grid = np.zeros((8, 8, 8), dtype=np.uint8)
        grid[index : index + 2, 1:3, 4:6] = 1
        np.save(pool.joinpath(f"{index}.npy"), grid)

    compiled = list(compile_pool(pool, tmp_path.joinpath("cache")))

    assert len(compiled) == 2
    assert [
        sum(chunk.occupied_voxels for chunk in item.manifest.chunks)
        for item in compiled
    ] == [8, 8]


def test_stl_source_uses_existing_voxelizer_and_converts_coordinates(
    tmp_path: Path, monkeypatch
) -> None:
    from theseo_anysearch.worlds import compiler as compiler_module

    source = tmp_path.joinpath("shape.stl")
    source.write_text("solid shape\nendsolid shape\n", encoding="utf-8")
    calls = []

    def fake_voxelizer(path: Path, scale: float, grid_size: int, padding: int):
        calls.append((path, scale, grid_size, padding))
        return [(1, 1, 1), (8, 8, 8)]

    monkeypatch.setattr(compiler_module, "_load_stl_cells", fake_voxelizer)
    compiled = compile_world(
        [StlSource(source, scale=6.0, padding=1)],
        WorldExtent(x=8, y=8, z=8),
        tmp_path.joinpath("cache"),
        WorldCompilerConfig(chunk_shape=(4, 4, 4)),
    )

    assert calls == [(source, 6.0, 8, 1)]
    chunks = _read_chunks(compiled.root)
    assert chunks[(0, 0, 0)][0, 0, 0]
    assert chunks[(1, 1, 1)][3, 3, 3]
    assert compiled.manifest.overview is not None
    assert compiled.manifest.overview.source_type == "simplified_source_mesh"


def test_ascii_stl_compiles_with_native_voxelizer(tmp_path: Path) -> None:
    source = Path("usage", "geometries", "cube.stl")

    compiled = compile_world(
        [StlSource(source, scale=8.0, padding=2)],
        WorldExtent(x=16, y=16, z=16),
        tmp_path,
        WorldCompilerConfig(chunk_shape=(8, 8, 8)),
    )

    assert sum(chunk.occupied_voxels for chunk in compiled.manifest.chunks) > 0


def test_corruption_is_detected_and_recompiled_when_source_exists(
    tmp_path: Path,
) -> None:
    source_path = tmp_path.joinpath("source.npy")
    np.save(source_path, np.ones((4, 4, 4), dtype=np.uint8))
    cache = tmp_path.joinpath("cache")
    first = compile_world([NpySource(source_path)], WorldExtent(x=4, y=4, z=4), cache)
    first.pack_path.write_bytes(b"corrupt")

    rebuilt = compile_world([NpySource(source_path)], WorldExtent(x=4, y=4, z=4), cache)

    assert rebuilt.manifest.identity_sha256 == first.manifest.identity_sha256
    validate_compiled_world(rebuilt.root)


def test_short_pack_read_is_detected(tmp_path: Path) -> None:
    compiled = compile_world(
        [BoxSource((0, 0, 0), (3, 3, 3))],
        WorldExtent(x=8, y=8, z=8),
        tmp_path,
    )
    payload = compiled.pack_path.read_bytes()
    compiled.pack_path.write_bytes(payload[:-1])

    with pytest.raises(WorldPackCorruptError, match="pack checksum mismatch"):
        validate_compiled_world(compiled.root)


def test_tuple_index_corruption_is_detected(tmp_path: Path) -> None:
    compiled = compile_world(
        [BoxSource((0, 0, 0), (3, 3, 3))],
        WorldExtent(x=8, y=8, z=8),
        tmp_path,
    )
    index = json.loads(compiled.index_path.read_text(encoding="utf-8"))
    entry = next(iter(index.values()))
    entry["offset"] += 1
    compiled.index_path.write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(WorldPackCorruptError, match="chunk index mismatch"):
        validate_compiled_world(compiled.root)


def test_malformed_manifest_is_detected(tmp_path: Path) -> None:
    compiled = compile_world(
        [BoxSource((0, 0, 0), (0, 0, 0))],
        WorldExtent(x=2, y=2, z=2),
        tmp_path,
    )
    compiled.root.joinpath(MANIFEST_FILE).write_text("{", encoding="utf-8")

    with pytest.raises(WorldPackCorruptError, match="metadata is invalid"):
        validate_compiled_world(compiled.root)


def test_invalid_pack_without_source_fails_explicitly(tmp_path: Path) -> None:
    compiled = compile_world(
        [BoxSource((0, 0, 0), (0, 0, 0))],
        WorldExtent(x=2, y=2, z=2),
        tmp_path,
    )
    compiled.pack_path.write_bytes(b"corrupt")

    with pytest.raises(WorldPackUnavailableError, match="source data is required"):
        load_compiled_world(tmp_path, compiled.manifest.identity_sha256)


def test_incomplete_temporary_build_never_appears_valid(tmp_path: Path) -> None:
    incomplete = tmp_path.joinpath("interrupted")
    incomplete.mkdir()
    incomplete.joinpath(PACK_FILE).write_bytes(b"")
    incomplete.joinpath(MANIFEST_FILE).write_text("{}", encoding="utf-8")

    assert not incomplete.joinpath(COMPLETE_FILE).exists()
    with pytest.raises(WorldPackCorruptError, match="incomplete"):
        validate_compiled_world(incomplete)


def test_concurrent_requests_publish_one_entry(tmp_path: Path, monkeypatch) -> None:
    from theseo_anysearch.worlds import compiler as compiler_module

    calls = 0
    calls_lock = threading.Lock()
    original = compiler_module._write_pack

    def counted_write(*args, **kwargs) -> None:
        nonlocal calls
        with calls_lock:
            calls += 1
        time.sleep(0.05)
        original(*args, **kwargs)

    monkeypatch.setattr(compiler_module, "_write_pack", counted_write)
    results = []

    def build() -> None:
        results.append(
            compile_world(
                [BoxSource((0, 0, 0), (3, 3, 3))],
                WorldExtent(x=8, y=8, z=8),
                tmp_path,
                lock_timeout_seconds=2.0,
            )
        )

    threads = [threading.Thread(target=build), threading.Thread(target=build)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert calls == 1
    assert len(results) == 2
    assert results[0].root == results[1].root


@pytest.mark.parametrize("occupancy", [1, 64, 256])
def test_chunk_encodings_round_trip(occupancy: int, tmp_path: Path) -> None:
    grid = np.zeros((8, 8, 8), dtype=np.uint8)
    grid.reshape(-1)[:occupancy] = 1
    path = tmp_path.joinpath(f"{occupancy}.npy")
    np.save(path, grid)
    compiled = compile_world(
        [NpySource(path)], WorldExtent(x=8, y=8, z=8), tmp_path.joinpath("cache")
    )

    decoded = next(iter(_read_chunks(compiled.root).values()))
    assert np.array_equal(decoded, grid.astype(np.bool_))


def test_encoding_benchmark_reports_compression_and_throughput() -> None:
    sparse = np.zeros((16, 16, 16), dtype=np.bool_)
    sparse[0, 0, 0] = True
    dense = np.ones((16, 16, 16), dtype=np.bool_)

    results = benchmark_encodings([sparse, dense], repeats=2)

    assert [result["encoding"] for result in results] == ["sparse_u32", "uniform"]
    assert all(result["compression_ratio"] > 0 for result in results)
    assert all(result["encode_voxels_per_second"] > 0 for result in results)
    assert all(result["decode_voxels_per_second"] > 0 for result in results)
