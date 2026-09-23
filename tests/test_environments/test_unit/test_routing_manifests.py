from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from pydantic import ValidationError

from theseo_anysearch.environments.routing_manifests import (
    ArtifactRef,
    ConversionRecord,
    GridFrame,
    RightsRecord,
    RoutingObservationRecord,
    RoutingReferenceRecord,
    RoutingSplitRecord,
    RoutingTaskRecord,
    RoutingWorldRecord,
    SourceFile,
    SourceRecord,
    SplitMember,
    read_sidecar,
    storage_to_task,
    task_to_storage,
    validate_observation_masks,
    validate_routing_bundle,
    validate_task_endpoints,
    verify_artifact,
    write_sidecar,
)
from theseo_anysearch.worlds.manifest import WorldExtent


def _source(*, rights: RightsRecord | None = None) -> SourceRecord:
    return SourceRecord(
        source_id="synthetic-plant",
        source_url="https://example.test/synthetic-plant",
        revision="fixture-1",
        files=(
            SourceFile(relative_path="maps/plant.3dmap", sha256="a" * 64, role="geometry"),
            SourceFile(relative_path="tasks/plant.3dscen", sha256="b" * 64, role="task"),
        ),
        rights=rights or RightsRecord(),
    )


def _bundle() -> dict:
    source = _source()
    conversion = ConversionRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        converter_name="synthetic-box-fixture",
        converter_version="1",
        parameters={"voxel_size_m": 0.25},
        output_occupancy_sha256="c" * 64,
    )
    world = RoutingWorldRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        conversion_identity_sha256=conversion.identity_sha256,
        occupancy_sha256=conversion.output_occupancy_sha256,
        extent=WorldExtent(x=5, y=4, z=3),
        frame=GridFrame(source_origin_m=(10.0, -2.0, 0.0), meters_per_voxel=0.25),
        root_geometry_id="synthetic-plant-01",
        topology_family="rooms",
        site_id="synthetic-site",
    )
    task = RoutingTaskRecord(
        world_identity_sha256=world.identity_sha256,
        provenance="upstream",
        family="point_path",
        start_storage=(0, 0, 0),
        goal_storage=(4, 3, 2),
        movement_model="discrete_6",
        source_query_id="query-001",
        source_task_sha256="b" * 64,
    )
    observation = RoutingObservationRecord(
        world_identity_sha256=world.identity_sha256,
        sensor_model="fixture-occlusion-v1",
        observed_occupied=ArtifactRef(relative_path="masks/occupied.npy", sha256="d" * 64),
        observed_free=ArtifactRef(relative_path="masks/free.npy", sha256="e" * 64),
        unknown=ArtifactRef(relative_path="masks/unknown.npy", sha256="f" * 64),
    )
    reference = RoutingReferenceRecord(
        task_identity_sha256=task.identity_sha256,
        claim="unverified",
        cost=11.0,
    )
    split = RoutingSplitRecord(
        dataset_id="synthetic-routing-v1",
        members=(
            SplitMember(
                world_identity_sha256=world.identity_sha256,
                root_geometry_id=world.root_geometry_id,
                topology_family=world.topology_family,
                site_id=world.site_id,
                partition="train",
            ),
        ),
    )
    return {
        "sources": (source,),
        "conversions": (conversion,),
        "worlds": (world,),
        "tasks": (task,),
        "observations": (observation,),
        "references": (reference,),
        "split": split,
    }


def test_rights_fail_closed_until_reviewed() -> None:
    source = _source()
    with pytest.raises(PermissionError, match="training"):
        source.rights.require_allowed("training")
    with pytest.raises(ValidationError, match="unreviewed"):
        RightsRecord(allowed_uses=("training",))
    with pytest.raises(ValidationError, match="evidence"):
        RightsRecord(status="reviewed", allowed_uses=("evaluation",))

    cleared = RightsRecord(
        status="reviewed",
        license_expression="fixture-only",
        allowed_uses=("evaluation",),
        evidence="reviewed fixture source",
    )
    cleared.require_allowed("evaluation")
    with pytest.raises(PermissionError, match="redistribution"):
        cleared.require_allowed("redistribution")
    assert _source(rights=cleared).content_identity_sha256 == source.content_identity_sha256
    assert _source(rights=cleared).identity_sha256 != source.identity_sha256


def test_source_identity_ignores_file_order_and_rejects_unsafe_paths() -> None:
    source = _source()
    reordered = SourceRecord.model_validate(
        {
            **source.model_dump(mode="json"),
            "files": list(reversed(source.model_dump(mode="json")["files"])),
        }
    )
    assert reordered.content_identity_sha256 == source.content_identity_sha256
    assert reordered.identity_sha256 == source.identity_sha256
    for path in ("../escape.npy", "/absolute.npy", "C:/drive.npy", "foo\\bar.npy", "a//b.npy"):
        with pytest.raises(ValidationError, match="relative POSIX"):
            ArtifactRef(relative_path=path, sha256="0" * 64)
    with pytest.raises(ValidationError, match="unique paths"):
        SourceRecord.model_validate(
            {**source.model_dump(mode="json"), "files": [source.files[0], source.files[0]]}
        )


def test_sidecar_bytes_ignore_unordered_source_files_and_split_members(tmp_path) -> None:
    source = _source()
    reversed_files = SourceRecord.model_validate(
        {
            **source.model_dump(mode="json"),
            "files": list(reversed(source.model_dump(mode="json")["files"])),
        }
    )
    first_path = tmp_path / "source-1.json"
    second_path = tmp_path / "source-2.json"
    write_sidecar(first_path, source)
    write_sidecar(second_path, reversed_files)
    assert first_path.read_bytes() == second_path.read_bytes()

    world = _bundle()["worlds"][0]
    members = (
        SplitMember(
            world_identity_sha256=world.identity_sha256,
            root_geometry_id="root-a",
            topology_family="room",
            partition="train",
        ),
        SplitMember(
            world_identity_sha256="9" * 64,
            root_geometry_id="root-b",
            topology_family="tunnel",
            partition="test",
        ),
    )
    split = RoutingSplitRecord(dataset_id="fixture", members=members)
    reverse_split = RoutingSplitRecord(dataset_id="fixture", members=members[::-1])
    assert split.identity_sha256 == reverse_split.identity_sha256
    write_sidecar(tmp_path / "split-1.json", split)
    write_sidecar(tmp_path / "split-2.json", reverse_split)
    assert (tmp_path / "split-1.json").read_bytes() == (tmp_path / "split-2.json").read_bytes()


def test_source_artifact_hash_is_checked_against_local_bytes(tmp_path) -> None:
    source_file = tmp_path / "fixture.bin"
    source_file.write_bytes(b"synthetic geometry")
    correct = ArtifactRef(
        relative_path="fixture.bin",
        sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
    )
    verify_artifact(tmp_path, correct)
    with pytest.raises(ValueError, match="SHA-256"):
        verify_artifact(
            tmp_path,
            ArtifactRef(relative_path="fixture.bin", sha256="0" * 64),
        )


@pytest.mark.parametrize(
    "axes",
    [
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ],
)
def test_metric_storage_and_one_based_task_round_trip(axes) -> None:
    frame = GridFrame(
        source_origin_m=(10.0, -2.0, 0.5),
        storage_axes_in_source=axes,
        meters_per_voxel=0.25,
    )
    extent = WorldExtent(x=7, y=5, z=3)
    coordinate = (6, 3, 1)
    source_m = frame.to_source_center(coordinate, extent)
    assert frame.from_source_center(source_m, extent) == coordinate
    assert task_to_storage(storage_to_task(coordinate, extent), extent) == coordinate
    assert storage_to_task(coordinate, extent) == (7, 4, 2)
    with pytest.raises(ValueError, match="voxel center"):
        frame.from_source_center((source_m[0] + 0.02, source_m[1], source_m[2]), extent)
    with pytest.raises(ValueError, match="outside"):
        task_to_storage((0, 1, 1), extent)


def test_frame_and_coordinates_reject_malformed_values() -> None:
    with pytest.raises(ValidationError, match="orthogonal"):
        GridFrame(
            source_origin_m=(0.0, 0.0, 0.0),
            storage_axes_in_source=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
            meters_per_voxel=1.0,
        )
    with pytest.raises(ValidationError):
        GridFrame(source_origin_m=(0.0, 0.0, float("nan")), meters_per_voxel=1.0)
    with pytest.raises(ValueError, match="integer"):
        storage_to_task((True, 1, 1), WorldExtent(x=3, y=3, z=3))
    with pytest.raises(ValidationError):
        RoutingTaskRecord.model_validate(
            {**_bundle()["tasks"][0].model_dump(mode="json"), "start_storage": [0.5, 0, 0]}
        )


def test_world_and_task_identities_track_roof_radius_ceiling_and_endpoints() -> None:
    bundle = _bundle()
    world = bundle["worlds"][0]
    task = bundle["tasks"][0]
    roof_conversion = ConversionRecord(
        source_content_identity_sha256=bundle["sources"][0].content_identity_sha256,
        converter_name="add-collision-roof",
        converter_version="1",
        parameters={"roof_underside_m": 3.5},
        output_occupancy_sha256="9" * 64,
        parent_world_identity_sha256=world.identity_sha256,
    )
    roof_world = world.model_copy(
        update={
            "conversion_identity_sha256": roof_conversion.identity_sha256,
            "occupancy_sha256": roof_conversion.output_occupancy_sha256,
            "parent_world_identity_sha256": world.identity_sha256,
        }
    )
    assert roof_world.identity_sha256 != world.identity_sha256
    assert (
        task.model_copy(
            update={"world_identity_sha256": roof_world.identity_sha256}
        ).identity_sha256
        != task.identity_sha256
    )
    assert task.model_copy(update={"body_radius_m": 0.3}).identity_sha256 != task.identity_sha256
    assert task.model_copy(update={"ceiling_source_m": 2.0}).identity_sha256 != task.identity_sha256
    assert (
        task.model_copy(update={"goal_storage": (3, 3, 2)}).identity_sha256
        != task.identity_sha256
    )


def test_provenance_and_reference_claims_fail_closed() -> None:
    task = _bundle()["tasks"][0]
    with pytest.raises(ValidationError, match="source query evidence"):
        RoutingTaskRecord.model_validate(
            {**task.model_dump(mode="json"), "source_query_id": None}
        )
    with pytest.raises(ValidationError, match="derivation"):
        RoutingTaskRecord.model_validate(
            {
                **task.model_dump(mode="json"),
                "provenance": "derived",
                "source_query_id": None,
                "source_task_sha256": None,
            }
        )
    with pytest.raises(ValidationError, match="route"):
        RoutingReferenceRecord(
            task_identity_sha256=task.identity_sha256, claim="feasible", cost=3.0
        )
    with pytest.raises(ValidationError, match="verification"):
        RoutingReferenceRecord(
            task_identity_sha256=task.identity_sha256,
            claim="certified_optimal",
            route_artifact=ArtifactRef(relative_path="routes/one.json", sha256="a" * 64),
        )
    with pytest.raises(ValidationError, match="verification"):
        RoutingReferenceRecord(
            task_identity_sha256=task.identity_sha256,
            claim="independently_validated",
            route_artifact=ArtifactRef(relative_path="routes/one.json", sha256="a" * 64),
        )


def test_split_rejects_roof_or_site_leakage() -> None:
    world = _bundle()["worlds"][0]
    first = SplitMember(
        world_identity_sha256=world.identity_sha256,
        root_geometry_id="same-scene",
        topology_family="rooms",
        site_id="site-one",
        partition="train",
    )
    roof = first.model_copy(
        update={"world_identity_sha256": "8" * 64, "partition": "test"}
    )
    with pytest.raises(ValidationError, match="root geometry"):
        RoutingSplitRecord(dataset_id="test", members=(first, roof))
    unrelated_root_same_site = roof.model_copy(update={"root_geometry_id": "other-scene"})
    with pytest.raises(ValidationError, match="site"):
        RoutingSplitRecord(dataset_id="test", members=(first, unrelated_root_same_site))


def test_validation_is_the_canonical_partition_and_calibration_is_a_legacy_alias() -> None:
    """See #493: `validation` replaces `calibration` as the canonical split name.

    `calibration` must remain an accepted `Partition` value so an already-written,
    content-addressed split.json naming it keeps loading and hashing to its
    original identity_sha256; new code must never write it.
    """

    world = _bundle()["worlds"][0]
    legacy = SplitMember(
        world_identity_sha256=world.identity_sha256,
        root_geometry_id="legacy-root",
        topology_family="room",
        partition="calibration",
    )
    canonical = SplitMember(
        world_identity_sha256="7" * 64,
        root_geometry_id="canonical-root",
        topology_family="room",
        partition="validation",
    )
    split = RoutingSplitRecord(dataset_id="fixture", members=(legacy, canonical))
    assert split.members[0].partition == "calibration"
    assert split.members[1].partition == "validation"


def test_legacy_calibration_split_sidecar_loads_and_hashes_unchanged(tmp_path) -> None:
    world = _bundle()["worlds"][0]
    legacy = RoutingSplitRecord(
        dataset_id="fixture",
        members=(
            SplitMember(
                world_identity_sha256=world.identity_sha256,
                root_geometry_id="legacy-root",
                topology_family="room",
                partition="calibration",
            ),
        ),
    )
    path = tmp_path / "split.json"
    write_sidecar(path, legacy)
    envelope = json.loads(path.read_bytes())
    assert envelope["payload"]["members"][0]["partition"] == "calibration"
    reloaded = read_sidecar(path, RoutingSplitRecord)
    assert reloaded.members[0].partition == "calibration"
    assert reloaded.identity_sha256 == legacy.identity_sha256 == envelope["identity_sha256"]


def test_legacy_and_canonical_partitions_still_reject_root_geometry_leakage() -> None:
    """A root geometry cannot be assigned to both the legacy and canonical name."""

    world = _bundle()["worlds"][0]
    legacy = SplitMember(
        world_identity_sha256=world.identity_sha256,
        root_geometry_id="same-scene",
        topology_family="room",
        partition="calibration",
    )
    relabeled = legacy.model_copy(
        update={"world_identity_sha256": "6" * 64, "partition": "validation"}
    )
    with pytest.raises(ValidationError, match="root geometry"):
        RoutingSplitRecord(dataset_id="test", members=(legacy, relabeled))


def test_box_observation_keeps_unknown_separate_from_free() -> None:
    truth = np.zeros((5, 5, 5), dtype=np.uint8)
    truth[2, 2, 2] = 1
    unknown = np.zeros_like(truth)
    unknown[1:4, 1:4, 1:4] = 1
    occupied = truth & ~unknown
    free = (1 - truth) & ~unknown
    validate_observation_masks(occupied, free, unknown, full_occupied=truth)
    contaminated_free = free.copy()
    contaminated_free[1, 1, 1] = 1
    with pytest.raises(ValueError, match="disjoint"):
        validate_observation_masks(occupied, contaminated_free, unknown, full_occupied=truth)
    with pytest.raises(ValueError, match="0/1"):
        validate_observation_masks(occupied, free.astype(float), unknown)
    with pytest.raises(ValueError, match="cover"):
        validate_observation_masks(occupied, np.zeros_like(free), unknown)
    false_obstacle = occupied.copy()
    false_obstacle[0, 0, 0] = 1
    free_without_false_obstacle = free.copy()
    free_without_false_obstacle[0, 0, 0] = 0
    with pytest.raises(ValueError, match="conflicts"):
        validate_observation_masks(
            false_obstacle,
            free_without_false_obstacle,
            unknown,
            full_occupied=truth,
        )


def test_synthetic_tunnel_and_invalid_query_endpoints() -> None:
    grid = np.ones((7, 7, 7), dtype=np.uint8)
    grid[1:6, 3, 3] = 0
    source = _source()
    conversion = ConversionRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        converter_name="synthetic-tunnel-fixture",
        converter_version="1",
        output_occupancy_sha256="7" * 64,
    )
    world = RoutingWorldRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        conversion_identity_sha256=conversion.identity_sha256,
        occupancy_sha256=conversion.output_occupancy_sha256,
        extent=WorldExtent(x=7, y=7, z=7),
        frame=GridFrame(source_origin_m=(0.0, 0.0, 0.0), meters_per_voxel=0.25),
        root_geometry_id="tunnel-fixture",
        topology_family="tunnel",
    )
    task = RoutingTaskRecord(
        world_identity_sha256=world.identity_sha256,
        provenance="derived",
        family="point_path",
        start_storage=(1, 3, 3),
        goal_storage=(5, 3, 3),
        movement_model="discrete_6",
        derivation_reason="synthetic tunnel endpoint fixture",
    )
    validate_task_endpoints(task, world, grid)
    with pytest.raises(ValueError, match="occupied goal"):
        validate_task_endpoints(
            task.model_copy(update={"goal_storage": (0, 0, 0)}), world, grid
        )


def test_bundle_linkage_and_deterministic_identity() -> None:
    bundle = _bundle()
    first = validate_routing_bundle(**bundle)
    assert len(first) == 64
    assert validate_routing_bundle(**bundle) == first
    bad_task = bundle["tasks"][0].model_copy(update={"goal_storage": (5, 0, 0)})
    with pytest.raises(ValueError, match="outside"):
        validate_routing_bundle(**{**bundle, "tasks": (bad_task,), "references": ()})
    missing_world = bundle["tasks"][0].model_copy(update={"world_identity_sha256": "0" * 64})
    with pytest.raises(ValueError, match="unknown world"):
        validate_routing_bundle(**{**bundle, "tasks": (missing_world,), "references": ()})
    bad_split = bundle["split"].model_copy(
        update={
            "members": (
                bundle["split"].members[0].model_copy(
                    update={"topology_family": "wrong"}
                ),
            )
        }
    )
    with pytest.raises(ValueError, match="split metadata"):
        validate_routing_bundle(**{**bundle, "split": bad_split})


def test_derived_world_cannot_change_source_lineage() -> None:
    bundle = _bundle()
    world = bundle["worlds"][0]
    conversion = ConversionRecord(
        source_content_identity_sha256=bundle["sources"][0].content_identity_sha256,
        converter_name="roof-fixture",
        converter_version="1",
        output_occupancy_sha256="9" * 64,
        parent_world_identity_sha256=world.identity_sha256,
    )
    wrong_root = world.model_copy(
        update={
            "conversion_identity_sha256": conversion.identity_sha256,
            "occupancy_sha256": conversion.output_occupancy_sha256,
            "parent_world_identity_sha256": world.identity_sha256,
            "root_geometry_id": "unrelated-layout",
        }
    )
    with pytest.raises(ValueError, match="source lineage"):
        validate_routing_bundle(
            **{
                **bundle,
                "conversions": (*bundle["conversions"], conversion),
                "worlds": (world, wrong_root),
            }
        )

    missing_parent_conversion = conversion.model_copy(
        update={"parent_world_identity_sha256": "f" * 64}
    )
    missing_parent_world = wrong_root.model_copy(
        update={
            "conversion_identity_sha256": missing_parent_conversion.identity_sha256,
            "parent_world_identity_sha256": "f" * 64,
            "root_geometry_id": world.root_geometry_id,
        }
    )
    with pytest.raises(ValueError, match="unknown parent world"):
        validate_routing_bundle(
            **{
                **bundle,
                "conversions": (*bundle["conversions"], missing_parent_conversion),
                "worlds": (world, missing_parent_world),
            }
        )


def test_sidecar_round_trip_rejects_tampering_and_duplicate_json(tmp_path) -> None:
    source = _source()
    path = tmp_path / "source.json"
    write_sidecar(path, source)
    assert read_sidecar(path, SourceRecord) == source
    with pytest.raises(FileExistsError):
        write_sidecar(path, source)
    with pytest.raises(ValueError, match="wrong record type"):
        read_sidecar(path, RoutingWorldRecord)

    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["payload"]["revision"] = "tampered"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        read_sidecar(path, SourceRecord)
    path.write_text(
        '{"payload":{},"payload":{},"record_type":"SourceRecord","identity_sha256":"0"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        read_sidecar(path, SourceRecord)
