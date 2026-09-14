"""No-network compact routing dataset tests with synthetic imported worlds."""

import hashlib
import json

import numpy as np
import pytest
import torch
from torch import nn
from types import SimpleNamespace

from theseo_anysearch.environments.routing_manifests import (
    ArtifactRef,
    ConversionRecord,
    GridFrame,
    RightsRecord,
    RoutingReferenceRecord,
    RoutingSplitRecord,
    RoutingTaskRecord,
    RoutingWorldRecord,
    SourceFile,
    SourceRecord,
    SplitMember,
    write_sidecar,
)
from theseo_anysearch.garden.external_routing import (
    axis_segment_clear,
    bind_matched_controls,
    compact_report,
    load_imported_world,
    prepare_routing_rows,
    read_prepared_dataset,
    synthetic_observation,
    write_prepared_dataset,
)
from theseo_anysearch.garden.external_routing_cli import run_plan
from theseo_anysearch.garden.compact import CompactEncoder
from theseo_anysearch.worlds.manifest import WorldExtent


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, name, *, rights=True, root_id=None, site_id=None, family="fixture-box"):
    source_root = tmp_path / f"{name}-source"
    source_root.mkdir()
    (source_root / "geometry.txt").write_text(name, encoding="utf-8")
    export = tmp_path / name
    export.mkdir()
    occupancy = np.zeros((25, 25, 25), dtype=np.uint8)
    occupancy[14, 12, 12] = 1
    np.save(export / "occupancy.npy", occupancy, allow_pickle=False)
    source = SourceRecord(
        source_id=name,
        source_url=f"fixture://{name}",
        revision="fixture-v1",
        files=(SourceFile(relative_path="geometry.txt", sha256=_sha(source_root / "geometry.txt"), role="geometry"),),
        rights=RightsRecord(
            status="reviewed" if rights else "unreviewed",
            license_expression="MIT" if rights else None,
            allowed_uses=("training",) if rights else (),
            evidence="fixture created in test" if rights else None,
        ),
    )
    conversion = ConversionRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        converter_name="fixture", converter_version="1",
        output_occupancy_sha256=_sha(export / "occupancy.npy"),
    )
    world = RoutingWorldRecord(
        source_content_identity_sha256=source.content_identity_sha256,
        conversion_identity_sha256=conversion.identity_sha256,
        occupancy_sha256=conversion.output_occupancy_sha256,
        extent=WorldExtent.from_value(occupancy.shape),
        frame=GridFrame(source_origin_m=(0, 0, 0), meters_per_voxel=1),
        root_geometry_id=root_id or name,
        site_id=site_id,
        topology_family=family,
    )
    task = RoutingTaskRecord(
        world_identity_sha256=world.identity_sha256,
        provenance="derived", family="drone_flight",
        start_storage=(12, 12, 12), goal_storage=(20, 12, 12),
        movement_model="6_axis_static_grid", body_radius_m=0,
        derivation_reason="fixture path",
    )
    route = [[12, 12, 12], [12, 13, 12], [20, 13, 12], [20, 12, 12]]
    (export / "route-storage.json").write_text(json.dumps(route), encoding="utf-8")
    reference = RoutingReferenceRecord(
        task_identity_sha256=task.identity_sha256,
        claim="independently_validated",
        route_artifact=ArtifactRef(relative_path="route-storage.json", sha256=_sha(export / "route-storage.json")),
        verification_evidence="fixture reference checked separately",
    )
    split = RoutingSplitRecord(
        dataset_id=f"fixture-{name}",
        members=(SplitMember(
            world_identity_sha256=world.identity_sha256,
            root_geometry_id=world.root_geometry_id,
            topology_family=world.topology_family,
            site_id=world.site_id, partition="test",
        ),),
    )
    for key, record in (
        ("source", source), ("conversion", conversion), ("world", world),
        ("task", task), ("reference", reference), ("split", split),
    ):
        write_sidecar(export / f"{key}.json", record)
    return export, source_root


def test_load_checks_rights_bytes_and_route(tmp_path):
    export, source_root = _fixture(tmp_path, "valid")
    item = load_imported_world(export, source_root=source_root)
    assert item.world.root_geometry_id == "valid"
    (export / "route-storage.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_imported_world(export, source_root=source_root)


def test_unreviewed_source_fails_closed(tmp_path):
    export, source_root = _fixture(tmp_path, "unreviewed", rights=False)
    with pytest.raises(PermissionError, match="training"):
        load_imported_world(export, source_root=source_root)


def test_source_and_occupancy_tampering_fail(tmp_path):
    export, source_root = _fixture(tmp_path, "tamper")
    (source_root / "geometry.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256"):
        load_imported_world(export, source_root=source_root)
    (source_root / "geometry.txt").write_text("tamper", encoding="utf-8")
    (export / "occupancy.npy").write_bytes(b"not a valid numpy file")
    with pytest.raises(ValueError, match="occupancy bytes"):
        load_imported_world(export, source_root=source_root)


def test_partial_observation_stops_at_first_obstacle():
    truth = np.zeros((33, 33, 33), dtype=np.uint8)
    truth[18, 16, 16] = 1
    occupied, free, unknown = synthetic_observation(truth, np.ones_like(truth, dtype=bool))
    assert occupied[18, 16, 16]
    assert free[17, 16, 16]
    assert unknown[19, 16, 16]
    assert unknown[29, 28, 20]
    assert np.all(occupied.astype(int) + free.astype(int) + unknown.astype(int) == 1)


def test_axis_segment_uses_full_truth_and_body_radius():
    truth = np.zeros((25, 25, 25), dtype=np.uint8)
    truth[14, 12, 12] = 1
    assert not axis_segment_clear(truth, (12, 12, 12), (20, 12, 12), body_radius_voxels=0)
    assert axis_segment_clear(truth, (12, 13, 12), (20, 13, 12), body_radius_voxels=0)
    assert not axis_segment_clear(truth, (12, 13, 12), (20, 13, 12), body_radius_voxels=0.5)
    assert not axis_segment_clear(truth, (12, 12, 12), (-1, 12, 12), body_radius_voxels=0)
    assert not axis_segment_clear(truth, (0, 12, 12), (4, 12, 12), body_radius_voxels=0.5)


def test_fresh_split_identity_rows_and_feature_alignment(tmp_path):
    items = []
    partitions = {}
    for partition in ("train", "calibration", "test"):
        for index in range(2):
            name = f"{partition}-{index}"
            export, source_root = _fixture(tmp_path, name)
            items.append(load_imported_world(export, source_root=source_root))
            partitions[name] = partition
    prepared = prepare_routing_rows(items, dataset_id="external-fixture-v1", partitions=partitions)
    reversed_input = prepare_routing_rows(list(reversed(items)), dataset_id="external-fixture-v1", partitions=partitions)
    assert prepared.dataset_identity_sha256 == reversed_input.dataset_identity_sha256
    assert prepared.query_identity_sha256 == reversed_input.query_identity_sha256
    assert len(prepared.rows) == 6 * 2 * 12
    assert {row.partition for row in prepared.rows} == {"train", "calibration", "test"}
    assert any(not row.traversable for row in prepared.rows)
    assert any(row.traversable for row in prepared.rows)
    assert compact_report(prepared)["claims"] == "preparation_only_no_training_or_cross_family_result"
    report = write_prepared_dataset(tmp_path / "prepared", prepared)
    assert report["rows"] == len(prepared.rows)
    loaded, metadata, arrays = read_prepared_dataset(tmp_path / "prepared")
    assert loaded["query_identity_sha256"] == prepared.query_identity_sha256
    assert metadata[0]["row_id"] == prepared.rows[0].row_id
    assert arrays["raw_grid"].shape == (12, 3, 33, 33, 33)
    assert arrays["observation_index"].shape == (len(prepared.rows),)
    with pytest.raises(FileExistsError):
        write_prepared_dataset(tmp_path / "prepared", prepared)
    (tmp_path / "prepared" / "rows.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="rows.json"):
        read_prepared_dataset(tmp_path / "prepared")

    obs = {row.observation_id: row for row in prepared.rows}
    codes = {key: np.array([index, index + 0.5], dtype=np.float32) for index, key in enumerate(obs)}
    spatial = {key: np.full((2, 3, 3, 3), index, dtype=np.float32) for index, key in enumerate(obs)}
    controls = bind_matched_controls(
        prepared, spatial_by_observation=spatial, code_by_observation=codes,
        spatial_state_sha256="a" * 64, code_state_sha256="b" * 64, shuffle_seed=9,
    )
    again = bind_matched_controls(
        prepared, spatial_by_observation=spatial, code_by_observation=codes,
        spatial_state_sha256="a" * 64, code_state_sha256="b" * 64, shuffle_seed=9,
    )
    assert controls.feature_identity_sha256 == again.feature_identity_sha256
    assert controls.row_ids == tuple(row.row_id for row in prepared.rows)
    assert np.array_equal(controls.labels, [row.traversable for row in prepared.rows])
    assert controls.raw_grid.shape == (len(prepared.rows), 3, 33, 33, 33)
    for index, row in enumerate(prepared.rows):
        assert np.array_equal(controls.frozen_code[index], codes[row.observation_id])
        donor = next(key for key, value in codes.items() if np.array_equal(value, controls.shuffled_code[index]))
        assert obs[donor].partition == row.partition
        assert obs[donor].root_geometry_id != row.root_geometry_id


def test_split_leakage_and_missing_control_rows_rejected(tmp_path):
    first, source_root = _fixture(tmp_path, "a", root_id="same-root")
    second, other_source = _fixture(tmp_path, "b", root_id="same-root")
    a = load_imported_world(first, source_root=source_root)
    b = load_imported_world(second, source_root=other_source)
    with pytest.raises(ValueError, match="partition assignment"):
        prepare_routing_rows((a, b), dataset_id="bad", partitions={"a": "train", "b": "test"})
    grouped = prepare_routing_rows((a, b), dataset_id="grouped", partitions={"same-root": "train"})
    assert {row.partition for row in grouped.rows} == {"train"}
    with pytest.raises(ValueError, match="two root geometries"):
        bind_matched_controls(
            grouped,
            spatial_by_observation={row.observation_id: np.zeros(2) for row in grouped.rows},
            code_by_observation={row.observation_id: np.ones(2) for row in grouped.rows},
            spatial_state_sha256="a" * 64, code_state_sha256="b" * 64, shuffle_seed=0,
        )
    with pytest.raises(ValueError, match="exactly"):
        bind_matched_controls(
            grouped, spatial_by_observation={}, code_by_observation={},
            spatial_state_sha256="a" * 64, code_state_sha256="b" * 64, shuffle_seed=0,
        )


def test_site_group_cannot_cross_partitions(tmp_path):
    first, first_source = _fixture(tmp_path, "site-a", site_id="plant-1")
    second, second_source = _fixture(tmp_path, "site-b", site_id="plant-1")
    a = load_imported_world(first, source_root=first_source)
    b = load_imported_world(second, source_root=second_source)
    with pytest.raises(ValueError, match="site crosses"):
        prepare_routing_rows(
            (a, b), dataset_id="bad-site",
            partitions={"site-a": "train", "site-b": "test"},
        )


def test_multiple_families_require_named_test_holdout(tmp_path):
    first, first_source = _fixture(tmp_path, "family-a")
    second, second_source = _fixture(tmp_path, "family-b", family="fixture-tunnel")
    a = load_imported_world(first, source_root=first_source)
    b = load_imported_world(second, source_root=second_source)
    with pytest.raises(ValueError, match="explicit test-family"):
        prepare_routing_rows(
            (a, b), dataset_id="no-holdout",
            partitions={"family-a": "train", "family-b": "test"},
        )
    with pytest.raises(ValueError, match="test-only"):
        prepare_routing_rows(
            (a, b), dataset_id="bad-holdout",
            partitions={"family-a": "train", "family-b": "calibration"},
            test_topology_families=("fixture-tunnel",),
        )
    valid = prepare_routing_rows(
        (a, b), dataset_id="family-holdout",
        partitions={"family-a": "train", "family-b": "test"},
        test_topology_families=("fixture-tunnel",),
    )
    assert compact_report(valid)["test_topology_families"] == ["fixture-tunnel"]


def test_cli_plan_pins_spec_and_fails_closed_on_rights(tmp_path):
    first, first_source = _fixture(tmp_path, "cli-reviewed")
    second, second_source = _fixture(tmp_path, "cli-unreviewed", rights=False)
    plan = tmp_path / "plan.json"
    payload = {
        "governing_spec_sha": "1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b",
        "dataset_id": "fixture-cli-v1",
        "imports": [
            {"export": first.name, "source_root": first_source.name, "partition": "train"},
        ],
    }
    plan.write_text(json.dumps(payload), encoding="utf-8")
    report = run_plan(plan, tmp_path / "cli-output")
    assert report["split_coverage"] == "incomplete_not_fit_ready"
    assert report["rows"] == 24
    payload["imports"].append(
        {"export": second.name, "source_root": second_source.name, "partition": "test"}
    )
    plan.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="training"):
        run_plan(plan, tmp_path / "blocked-output")
    assert not (tmp_path / "blocked-output").exists()
    payload["governing_spec_sha"] = "0" * 40
    plan.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="literal_error"):
        run_plan(plan, tmp_path / "wrong-spec")


def test_frozen_feature_extraction_uses_observed_channels_only(tmp_path):
    from theseo_anysearch.garden.external_routing import extract_frozen_features

    class FixtureBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = nn.Identity()

        def forward(self, level, hidden):
            assert torch.equal(level.features[:, 0] > 0, ~hidden[:, 0] & (level.features[:, 0] > 0))
            return SimpleNamespace(local_feature_volume=level.features.repeat(1, 2, 1, 1, 1))

    entries = []
    for name in ("extract-a", "extract-b"):
        export, source_root = _fixture(tmp_path, name)
        entries.append(load_imported_world(export, source_root=source_root))
    prepared = prepare_routing_rows(
        entries, dataset_id="extract-fixture",
        partitions={"extract-a": "train", "extract-b": "train"},
    )
    model = CompactEncoder(FixtureBackbone(), 64, "grid").requires_grad_(False)
    spatial, codes, spatial_hash, code_hash = extract_frozen_features(prepared, model)
    assert len(spatial) == len(codes) == 4
    controls = bind_matched_controls(
        prepared, spatial_by_observation=spatial, code_by_observation=codes,
        spatial_state_sha256=spatial_hash, code_state_sha256=code_hash, shuffle_seed=4,
    )
    assert controls.frozen_code.shape == (len(prepared.rows), 64)
    assert controls.spatial.shape == (len(prepared.rows), 8, 33, 33, 33)
    model.aggregation.project[0].weight.requires_grad_(True)
    with pytest.raises(ValueError, match="entirely frozen"):
        extract_frozen_features(prepared, model)
