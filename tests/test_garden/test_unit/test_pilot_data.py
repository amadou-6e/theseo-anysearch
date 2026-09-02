"""Tests for geometry-disjoint pilot data, targets, and immutable artifacts."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from theseo_anysearch.garden.collect import (
    load_pilot_observations,
    pilot_observation_sha256,
    save_pilot_observations,
)
from theseo_anysearch.garden.data_config import AugmentationConfig
from theseo_anysearch.garden.dataset import make_pilot_pool_datasets
from theseo_anysearch.garden.micro_scenes import MICRO_SCENE_ORACLES, make_micro_scenes
from theseo_anysearch.garden.splits import (
    CONFIRMATION_COUNTS,
    POOL_SIZES,
    GeometryDescriptor,
    assign_pilot_pools,
    make_fresh_draws,
)
from theseo_anysearch.garden.targets import (
    compute_geometry_targets,
    geodesic_distances,
    pair_targets,
)


FAMILIES = ("open", "thin_obstacle", "topology", "imported")
BANDS = ("low", "medium", "high")


def _geometry_records() -> list[GeometryDescriptor]:
    records: list[GeometryDescriptor] = []
    for index in range(168):
        records.append(
            GeometryDescriptor(
                geometry_id=f"base-{index:03d}",
                family=FAMILIES[index % len(FAMILIES)],
                occupancy_band=BANDS[(index // len(FAMILIES)) % len(BANDS)],
                source=f"source-{index:03d}",
            )
        )
    groups = (("ordinary", 16), ("heldout_topology", 8), ("heldout_imported", 8))
    offset = 0
    for group, count in groups:
        for local_index in range(count):
            if group == "heldout_topology":
                family = "topology"
            elif group == "heldout_imported":
                family = "imported"
            else:
                family = FAMILIES[local_index % len(FAMILIES)]
            records.append(
                GeometryDescriptor(
                    geometry_id=f"confirm-{offset + local_index:03d}",
                    family=family,
                    occupancy_band=BANDS[local_index % len(BANDS)],
                    source=f"confirm-source-{offset + local_index:03d}",
                    confirmation_group=group,
                )
            )
        offset += count
    return records


def test_pool_assignment_is_deterministic_disjoint_and_complete() -> None:
    records = _geometry_records()
    first = assign_pilot_pools(records, seed=90210)
    second = assign_pilot_pools(list(reversed(records)), seed=90210)

    assert first == second
    assert {name: len(ids) for name, ids in first.pools.items()} == POOL_SIZES
    flattened = [geometry_id for ids in first.pools.values() for geometry_id in ids]
    assert len(flattened) == len(set(flattened)) == 200
    by_id = {record.geometry_id: record for record in records}
    for ids in first.pools.values():
        selected = [by_id[geometry_id] for geometry_id in ids]
        assert {record.family for record in selected} == set(FAMILIES)
        assert {record.occupancy_band for record in selected} == set(BANDS)


def test_confirmation_composition_and_holdouts_never_enter_training() -> None:
    records = _geometry_records()
    assignment = assign_pilot_pools(records, seed=7)
    by_id = {record.geometry_id: record for record in records}
    confirmation = [by_id[geometry_id] for geometry_id in assignment.pools["pilot_confirm"]]

    assert {
        group: sum(record.confirmation_group == group for record in confirmation)
        for group in CONFIRMATION_COUNTS
    } == CONFIRMATION_COUNTS
    assert all(
        by_id[geometry_id].confirmation_group is None
        for geometry_id in assignment.pools["pilot_train"]
    )


def test_pool_assignment_rejects_duplicate_or_non_parent_geometry() -> None:
    records = _geometry_records()
    with pytest.raises(ValueError, match="globally unique"):
        assign_pilot_pools([*records, records[0]], seed=1)

    raw = records[0].__dict__ | {"parent_split": "test"}
    with pytest.raises(ValueError):
        GeometryDescriptor(**raw)


def test_fresh_draws_pin_seed_assignment_and_queries() -> None:
    assignment = assign_pilot_pools(_geometry_records(), seed=1)
    queries = {
        pool: [
            {"geometry_id": geometry_id, "query_id": f"{pool}-{index}", "weight": 1.0}
            for index, geometry_id in enumerate(assignment.pools[pool])
        ]
        for pool in ("pilot_dev_arch", "pilot_dev_interaction", "pilot_confirm")
    }

    first = make_fresh_draws(assignment, queries, seeds={"P4": 104, "P6": 106, "P7": 107})
    second = make_fresh_draws(
        assignment,
        {pool: list(reversed(values)) for pool, values in queries.items()},
        seeds={"P4": 104, "P6": 106, "P7": 107},
    )

    assert first == second
    assert first["P4"].pool == "pilot_dev_arch"
    assert first["P6"].pool == "pilot_dev_interaction"
    assert first["P7"].pool == "pilot_confirm"


def test_all_32_micro_scenes_match_frozen_oracles() -> None:
    scenes = make_micro_scenes()
    assert len(scenes) == 32
    assert {scene.name for scene in scenes} == set(MICRO_SCENE_ORACLES)
    for scene in scenes:
        targets = compute_geometry_targets(scene.occupancy, unknown_mask=scene.unknown_mask)
        actual = (
            int(targets.occupancy.sum()),
            int(targets.boundary.sum()),
            targets.free_components,
            targets.graph_cycle_rank,
            int(targets.valid_mask.sum()),
        )
        assert actual == MICRO_SCENE_ORACLES[scene.name], scene.name

    with pytest.raises(ValueError, match="defined only for side 9"):
        make_micro_scenes(side=11)


def test_esdf_has_exact_axis_and_diagonal_distances() -> None:
    occupancy = np.zeros((5, 5, 5), dtype=bool)
    occupancy[2, 2, 2] = True

    targets = compute_geometry_targets(occupancy, truncation=3.0)

    assert targets.signed_distance[2, 2, 2] == pytest.approx(-1.0)
    assert targets.signed_distance[2, 2, 3] == pytest.approx(1.0)
    assert targets.signed_distance[2, 3, 3] == pytest.approx(np.sqrt(2))
    assert targets.signed_distance[0, 0, 0] == pytest.approx(3.0)


def test_reachability_and_geodesic_targets_are_exact() -> None:
    scenes = {scene.name: scene for scene in make_micro_scenes()}
    corridor = compute_geometry_targets(scenes["corridor_x"].occupancy)
    reachable, distances = pair_targets(corridor, [((1, 4, 4), (7, 4, 4))])
    assert reachable.tolist() == [True]
    assert distances.tolist() == [6.0]

    disconnected = compute_geometry_targets(scenes["disconnected_x"].occupancy)
    reachable, distances = pair_targets(disconnected, [((1, 1, 1), (5, 1, 1))])
    assert reachable.tolist() == [False]
    assert np.isinf(distances[0])
    assert np.isinf(geodesic_distances(disconnected, (0, 0, 0))).all()


def test_pilot_dataset_uses_only_frozen_geometry_assignment() -> None:
    assignment = assign_pilot_pools(_geometry_records(), seed=2)
    selected = [ids[0] for ids in assignment.pools.values()]
    grids = np.arange(len(selected) * 27, dtype=np.float32).reshape(len(selected), 3, 3, 3)
    observation_ids = [f"observation-{index}" for index in range(len(selected))]
    targets = {"score": np.arange(len(selected), dtype=np.float32)}

    datasets = make_pilot_pool_datasets(
        grids,
        selected,
        observation_ids,
        assignment,
        train_augmentation=AugmentationConfig(rotate90=True),
        seed=42,
        targets=targets,
    )

    assert set(datasets) == set(POOL_SIZES)
    assert all(len(dataset) == 1 for dataset in datasets.values())
    geometry_sets = [dataset.geometry_ids for dataset in datasets.values()]
    assert all(not left & right for i, left in enumerate(geometry_sets) for right in geometry_sets[i + 1 :])
    first = datasets["pilot_train"][0]
    second = datasets["pilot_train"][0]
    assert torch.equal(first["grid"], second["grid"])
    assert first["geometry_id"] == selected[0]
    assert "score" in first


def test_pilot_dataset_rejects_unassigned_geometry() -> None:
    assignment = assign_pilot_pools(_geometry_records(), seed=2)
    with pytest.raises(ValueError, match="unassigned"):
        make_pilot_pool_datasets(
            np.zeros((1, 3, 3, 3), dtype=np.float32),
            ["not-assigned"],
            ["observation-0"],
            assignment,
        )


def test_observation_artifact_round_trip_and_freeze(tmp_path) -> None:
    grids = np.arange(54, dtype=np.uint8).reshape(2, 3, 3, 3)
    geometry_ids = ("geometry-a", "geometry-b")
    observation_ids = ("observation-a", "observation-b")
    path = tmp_path / "observations.npz"

    identity = save_pilot_observations(path, grids, geometry_ids, observation_ids)
    loaded = load_pilot_observations(path)

    assert identity == pilot_observation_sha256(grids, geometry_ids, observation_ids)
    assert loaded.identity_sha256 == identity
    np.testing.assert_array_equal(loaded.grids, grids)
    assert save_pilot_observations(path, grids, geometry_ids, observation_ids) == identity

    changed = grids.copy()
    changed[0, 0, 0, 0] += 1
    with pytest.raises(FileExistsError, match="refusing to replace"):
        save_pilot_observations(path, changed, geometry_ids, observation_ids)


def test_observation_artifact_detects_manifest_tampering(tmp_path) -> None:
    path = tmp_path / "observations.npz"
    save_pilot_observations(
        path,
        np.zeros((1, 3, 3, 3), dtype=np.uint8),
        ("geometry-a",),
        ("observation-a",),
    )
    manifest_path = path.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    manifest["identity_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="identity verification"):
        load_pilot_observations(path)
