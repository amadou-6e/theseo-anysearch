from __future__ import annotations

import pytest
from pydantic import ValidationError

from theseo_anysearch.worlds.manifest import (
    WorldExtent,
    WorldManifest,
    world_contract,
    world_contract_fingerprint,
)


def test_cubic_shorthand_resolves_to_three_axis_extent() -> None:
    assert WorldExtent.from_value(32).as_tuple() == (32, 32, 32)
    assert world_contract({"grid_size": 32})["extent"] == [32, 32, 32]


def test_non_cubic_extent_is_preserved() -> None:
    extent = WorldExtent.from_value((100_000, 50_000, 10_000))
    assert extent.as_tuple() == (100_000, 50_000, 10_000)
    assert world_contract({"extent": [100_000, 50_000, 10_000]})["extent"] == [
        100_000,
        50_000,
        10_000,
    ]


def test_conflicting_cubic_shorthand_and_extent_are_rejected() -> None:
    with pytest.raises(ValueError, match="different world bounds"):
        world_contract({"grid_size": 32, "extent": [32, 64, 32]})


def test_world_fingerprint_tracks_coordinate_contract(monkeypatch) -> None:
    from theseo_anysearch.worlds import manifest as world_manifest

    first = world_contract_fingerprint(world_contract({"grid_size": 32}))
    monkeypatch.setattr(world_manifest, "COORDINATE_TYPE", "u64")
    second = world_contract_fingerprint(world_contract({"grid_size": 32}))

    assert first != second


@pytest.mark.parametrize("extent", [(0, 1, 1), (1, -1, 1), (1, 1, 2**32)])
def test_extent_rejects_invalid_axes(extent: tuple[int, int, int]) -> None:
    with pytest.raises(ValidationError):
        WorldExtent.from_value(extent)


def test_manifest_rejects_unsupported_schema_and_coordinate_contract() -> None:
    valid = {
        "extent": {"x": 8, "y": 4, "z": 2},
        "chunk_shape": {"x": 4, "y": 4, "z": 2},
        "chunks": [],
        "identity_sha256": "0" * 64,
    }
    with pytest.raises(ValidationError):
        WorldManifest.model_validate({**valid, "schema_version": 2})
    with pytest.raises(ValidationError):
        WorldManifest.model_validate({**valid, "coordinate_type": "u16"})
    with pytest.raises(ValidationError):
        WorldManifest.model_validate({**valid, "environment_min": [0, 0, 0]})


def test_manifest_rejects_ambiguous_duplicate_chunk_coordinates() -> None:
    chunk = {
        "coordinate": {"x": 0, "y": 0, "z": 0},
        "relative_path": "chunks/0-0-0.bin",
        "sha256": "1" * 64,
        "byte_length": 10,
        "occupied_voxels": 1,
    }
    with pytest.raises(ValidationError, match="unique tuple coordinates"):
        WorldManifest.model_validate(
            {
                "extent": {"x": 8, "y": 4, "z": 2},
                "chunk_shape": {"x": 4, "y": 4, "z": 2},
                "chunks": [chunk, chunk],
                "identity_sha256": "0" * 64,
            }
        )
