from __future__ import annotations

import numpy as np

from theseo_anysearch.worlds.manifest import WorldExtent
from theseo_anysearch.worlds.overview import (
    CoarseCell,
    aggregate_cells,
    build_overview_from_chunks,
    build_overview_mesh,
    build_stl_overview_mesh,
    coarsen_cells,
    decode_overview_mesh,
    encode_overview_mesh,
    filter_components,
)


def test_sparse_isolated_component_is_removed() -> None:
    cells = {(0, 0, 0): CoarseCell(1, 8)}

    assert filter_components(cells) == {}


def test_long_thin_component_survives_size_rule() -> None:
    cells = {(x, 0, 0): CoarseCell(1, 8) for x in range(4)}

    assert filter_components(cells) == cells


def test_small_dense_component_survives_density_rule() -> None:
    occupied = {(x, y, z) for x in range(2) for y in range(2) for z in range(2)}
    cells = aggregate_cells(occupied, WorldExtent(x=2, y=2, z=2), scale=2)

    assert filter_components(cells) == cells


def test_diagonal_contact_does_not_join_sparse_components() -> None:
    cells = {
        (0, 0, 0): CoarseCell(1, 8),
        (1, 1, 1): CoarseCell(1, 8),
    }

    assert filter_components(cells) == {}


def test_filtered_result_feeds_next_coarser_candidate() -> None:
    cells = {(x, 0, 0): CoarseCell(1, 8) for x in range(4)}
    filtered = filter_components(cells)

    assert filtered == cells
    assert coarsen_cells(filtered, WorldExtent(x=8, y=2, z=2), 2) == {}


def test_overview_mesh_encoding_is_deterministic_and_round_trips() -> None:
    occupied = {(x, 0, 0) for x in range(8)}
    extent = WorldExtent(x=8, y=2, z=2)

    first = build_overview_mesh(occupied, extent)
    second = build_overview_mesh(set(occupied), extent)
    payload = encode_overview_mesh(first)

    assert payload == encode_overview_mesh(second)
    decoded = decode_overview_mesh(payload)
    assert decoded.vertices == first.vertices
    assert decoded.indices == first.indices
    assert first.triangle_count > 0


def test_chunk_overview_does_not_expand_dense_voxel_coordinates(monkeypatch) -> None:
    chunk = np.ones((32, 32, 32), dtype=np.bool_)

    def reject_coordinate_expansion(*_args, **_kwargs):
        raise AssertionError("dense chunk was expanded into source coordinates")

    monkeypatch.setattr(np, "argwhere", reject_coordinate_expansion)
    mesh = build_overview_from_chunks(
        {(1000, 1000, 1000): chunk},
        WorldExtent(x=60_000, y=40_000, z=40_000),
        (32, 32, 32),
    )

    assert mesh.triangle_count == 12


def test_stl_overview_uses_voxelizer_storage_transform(tmp_path) -> None:
    source = tmp_path.joinpath("triangle.stl")
    source.write_text(
        """solid triangle
facet normal 0 0 1
outer loop
vertex 10 20 30
vertex 12 20 30
vertex 10 22 30
endloop
endfacet
endsolid triangle
""",
        encoding="utf-8",
    )

    mesh = build_stl_overview_mesh(
        source, requested_scale=8.0, extent=WorldExtent(x=16, y=16, z=16), padding=2
    )

    assert mesh.vertices == ((2, 2, 2), (10, 2, 2), (2, 10, 2))
    assert mesh.indices == (0, 1, 2)
