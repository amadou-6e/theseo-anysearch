import json
from pathlib import Path

import pytest

from theseo_anysearch.worlds.artifacts import (
    ARTIFACT_MANIFEST_FILE,
    GeometryArtifactError,
    load_geometry_artifact,
    migrate_compiled_world,
    publish_eager_geometry,
)
from theseo_anysearch.worlds.compiler import BoxSource, compile_world
from theseo_anysearch.worlds.manifest import WorldExtent


def test_eager_artifact_is_content_addressed_and_reusable(tmp_path: Path) -> None:
    extent = WorldExtent(x=8, y=7, z=6)
    coordinates = ((1, 2, 3), (4, 5, 5))
    first = publish_eager_geometry(coordinates, extent, tmp_path)
    second = publish_eager_geometry(tuple(reversed(coordinates)), extent, tmp_path)
    assert first.root == second.root
    assert first.eager_coordinates() == coordinates
    assert first.manifest.extent == extent


def test_eager_artifact_detects_corruption(tmp_path: Path) -> None:
    artifact = publish_eager_geometry(((1, 1, 1),), WorldExtent(x=4, y=4, z=4), tmp_path)
    artifact.root.joinpath("occupancy.json").write_text("[]", encoding="utf-8")
    with pytest.raises(GeometryArtifactError, match="checksum mismatch"):
        load_geometry_artifact(artifact.root)


def test_compiled_pack_projects_into_same_logical_contract(tmp_path: Path) -> None:
    world = compile_world(
        [BoxSource((1, 1, 1), (2, 2, 2))],
        WorldExtent(x=8, y=8, z=8),
        tmp_path,
    )
    manifest = migrate_compiled_world(world.root)
    loaded = load_geometry_artifact(world.root)
    assert manifest == loaded.manifest
    assert manifest.occupancy == "compiled_pack"
    assert loaded.compiled_world is not None


def test_manifest_tampering_fails_explicitly(tmp_path: Path) -> None:
    artifact = publish_eager_geometry(((1, 1, 1),), WorldExtent(x=4, y=4, z=4), tmp_path)
    path = artifact.root.joinpath(ARTIFACT_MANIFEST_FILE)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["difficulty"] = {"detour_ratio": 9.0}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(GeometryArtifactError, match="manifest is invalid"):
        load_geometry_artifact(artifact.root)


def test_eager_and_compiled_artifacts_have_equivalent_bounded_reads(
    tmp_path: Path,
) -> None:
    extent = WorldExtent(x=8, y=8, z=8)
    eager = publish_eager_geometry(((2, 2, 2),), extent, tmp_path.joinpath("eager"))
    compiled = load_geometry_artifact(
        compile_world(
            [BoxSource((1, 1, 1), (1, 1, 1))],
            extent,
            tmp_path.joinpath("compiled"),
        ).root
    )
    eager_read = eager.bounded_reader()
    compiled_read = compiled.bounded_reader(maximum_resident_chunks=1)

    for coordinate in ((1, 1, 1), (2, 2, 2), (3, 3, 3), (9, 1, 1)):
        assert eager_read.occupied(coordinate) == compiled_read.occupied(coordinate)
    assert compiled_read.resident_chunk_count <= 1


def test_bounded_artifact_reader_enforces_query_budget(tmp_path: Path) -> None:
    artifact = publish_eager_geometry(
        ((1, 1, 1),), WorldExtent(x=4, y=4, z=4), tmp_path
    )
    reader = artifact.bounded_reader(maximum_queries=1)
    assert reader.occupied((1, 1, 1))
    with pytest.raises(GeometryArtifactError, match="query budget exceeded"):
        reader.occupied((2, 2, 2))
