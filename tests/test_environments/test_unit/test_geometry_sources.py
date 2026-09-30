"""Canonical small-world geometry composition tests."""

from theseo_anysearch.environments.geometry_sources import resolve_geometry_sources


def test_sources_use_deterministic_voxel_union() -> None:
    config = {
        "geometry_sources": [
            {"type": "stl", "path": "mesh.stl", "scale": 2.0, "padding": 4},
            {"type": "boxes", "boxes": [[2, 2, 2, 3, 2, 2]]},
        ]
    }
    calls = []

    def load_stl(path, scale, grid_size, *, padding):
        calls.append((path, scale, grid_size, padding))
        return [(2, 2, 2), (1, 1, 1)]

    result = resolve_geometry_sources(config, grid_size=8, load_stl=load_stl)

    assert result == [(1, 1, 1), (2, 2, 2), (3, 2, 2)]
    assert calls == [("mesh.stl", 2.0, 8, 4)]
