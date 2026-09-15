"""Deterministic tests for voxel-pilot v2 calibration and health gates."""
from __future__ import annotations

import torch

from theseo_anysearch.garden.models.outputs import VoxelLevel
from theseo_anysearch.garden.pilots.corpus import V1_PROGRAM, V2_PROGRAM, make_pilot_observation
from theseo_anysearch.garden.pilots.diagnostics import classify_t3_replay
from theseo_anysearch.garden.pilots.v2 import (
    V2_POOL_SIZES,
    assign_v2_pools,
    build_v2_pool_identities,
    construct_score_anchors,
    frequency_control,
    random_projection_control,
    v2_geometry_records,
)


def test_v2_pools_are_complete_balanced_and_fresh_from_v1() -> None:
    records = v2_geometry_records()
    pools = assign_v2_pools(records, seed=290210)
    assert {name: len(values) for name, values in pools.items()} == V2_POOL_SIZES
    all_ids = [geometry_id for values in pools.values() for geometry_id in values]
    assert len(all_ids) == len(set(all_ids)) == 248
    assert not any(geometry_id.startswith("pilot-v1-") for geometry_id in all_ids)
    by_id = {record.geometry_id: record for record in records}
    for values in pools.values():
        selected = [by_id[value] for value in values]
        assert {record.family for record in selected} == {
            "open",
            "thin_obstacle",
            "topology",
            "imported",
        }
        assert {record.occupancy_band for record in selected} == {"low", "medium", "high"}


def test_v2_pool_and_query_hashes_are_deterministic() -> None:
    first = build_v2_pool_identities(seed=290210)
    second = build_v2_pool_identities(seed=290210)
    assert first[1] == second[1]
    assert first[2] == second[2]
    assert first[1]["pilot_calibration"].query_sha256 != first[1]["pilot_diagnostic"].query_sha256


def test_corpus_program_is_part_of_observation_identity() -> None:
    descriptor = v2_geometry_records()[0]
    v1 = make_pilot_observation(descriptor, 0, radius=8, program=V1_PROGRAM)
    v2 = make_pilot_observation(descriptor, 0, radius=8, program=V2_PROGRAM)
    assert v1.identity_sha256 != v2.identity_sha256
    assert not torch.equal(torch.from_numpy(v1.occupancy), torch.from_numpy(v2.occupancy))


def test_fixed_controls_have_frozen_contract_shapes() -> None:
    descriptor = v2_geometry_records()[0]
    observation = make_pilot_observation(descriptor, 0, radius=8, program=V2_PROGRAM)
    level = VoxelLevel.from_occupancy(
        torch.from_numpy(observation.occupancy[None]).float(),
        unknown_mask=torch.from_numpy(observation.unknown_mask[None]),
    )
    frequency = frequency_control()(level)
    random_first = random_projection_control(3270)(level)
    random_second = random_projection_control(3270)(level)
    assert frequency.global_embedding.shape == (1, 192)
    assert frequency.local_feature_volume.shape == (1, 16, 17, 17, 17)
    assert torch.count_nonzero(frequency.global_embedding) == 0
    assert torch.equal(random_first.global_embedding, random_second.global_embedding)


def _cell(learning_rate: float, losses: tuple[float, float]) -> dict[str, object]:
    return {
        "learning_rate": learning_rate,
        "health_labels": [],
        "implementation_errors": [],
        "telemetry": [
            {"update": 1_000, "pretext_loss": losses[0]},
            {"update": 2_000, "pretext_loss": losses[1]},
        ],
    }


def test_t3_health_classification_requires_cross_rate_failure() -> None:
    healthy = classify_t3_replay([_cell(0.0001, (1.0, 1.1)), _cell(0.0003, (1.0, 1.2))])
    assert healthy["passed"]

    unstable = classify_t3_replay([_cell(0.0001, (1.0, 1.3)), _cell(0.0003, (1.0, 1.4))])
    assert unstable["late_optimization_instability"]
    assert unstable["mechanism_health_failure"]
    assert not unstable["passed"]


def test_one_rate_collapse_is_a_limit_not_a_selector_or_shared_failure() -> None:
    first = _cell(0.0001, (1.0, 1.0))
    first["health_labels"] = ["target_collapse"]
    result = classify_t3_replay([first, _cell(0.0003, (1.0, 1.0))])
    assert result["labels_by_learning_rate"]["0.0001"] == ["target_collapse"]
    assert not result["mechanism_health_failure"]
    assert result["passed"]


def test_measured_anchor_failure_is_complete_and_blocks_before_p1() -> None:
    components = {
        "occupied_iou": 0.4,
        "boundary_f1": 0.3,
        "clearance_nmae": 0.8,
        "reachability_auprc": 0.5,
        "geodesic_nmae": 0.7,
    }
    measured = {
        "frequency": components,
        "pca": components,
        "fixed_random_projection": components,
        "supervised_reference": {
            **components,
            "occupied_iou": 0.45,
            "boundary_f1": 0.6,
            "clearance_nmae": 0.5,
            "reachability_auprc": 0.8,
            "geodesic_nmae": 0.4,
        },
    }
    anchors, failures = construct_score_anchors(measured)
    assert "occupied_iou" in failures
    assert "occupied_iou" not in anchors
    assert set(anchors) | set(failures) == set(components)
