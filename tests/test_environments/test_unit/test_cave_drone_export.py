"""Contract tests for the pinned native CaveDroneSim export boundary."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from theseo_anysearch.environments.cave_drone_export import (
    EXPECTED_EXTENT,
    clearance_mask,
    export_seed,
    parse_bridge_output,
    select_fixed_goal_pairs,
    topology_census,
)
from theseo_anysearch.environments.routing_manifests import (
    RoutingReferenceRecord,
    RoutingTaskRecord,
    RoutingWorldRecord,
    read_sidecar,
)


def _corridor() -> np.ndarray:
    occupied = np.ones(EXPECTED_EXTENT, dtype=np.uint8)
    occupied[70:120, 24:33, 91:102] = 0
    return occupied


def test_bridge_parser_uses_xyz_storage_order() -> None:
    occupied = np.zeros(EXPECTED_EXTENT, dtype=np.uint8)
    occupied[2, 3, 4] = 1
    payload = b"CAVEDRONE 1 192 56 192 0.5 7\n" + occupied.tobytes(order="C")
    actual = parse_bridge_output(payload, seed=7)
    assert actual.shape == EXPECTED_EXTENT
    assert actual[2, 3, 4] == 1
    assert actual[4, 3, 2] == 0


@pytest.mark.parametrize(
    "case", ["no_header", "wrong_seed", "truncated", "oversized"],
)
def test_bridge_parser_rejects_bad_payload(case: str) -> None:
    header = b"CAVEDRONE 1 192 56 192 0.5 7\n"
    payload = {
        "no_header": b"no header",
        "wrong_seed": b"CAVEDRONE 1 192 56 192 0.5 8\n" + bytes(np.prod(EXPECTED_EXTENT)),
        "truncated": header + b"\0",
        "oversized": header + bytes(np.prod(EXPECTED_EXTENT) + 1),
    }[case]
    with pytest.raises(ValueError):
        parse_bridge_output(payload, seed=7)


def test_clearance_is_conservative_at_walls_and_bounds() -> None:
    occupied = _corridor()
    mask = clearance_mask(occupied, body_radius_m=0.25)
    assert mask[96, 28, 96]
    assert not mask[96, 24, 96]
    assert mask[96, 25, 96]
    assert not mask[0, 0, 0]
    with pytest.raises(ValueError):
        clearance_mask(occupied, body_radius_m=float("nan"))


def test_fixed_goals_are_deterministic_and_clearance_connected() -> None:
    occupied = _corridor()
    first = select_fixed_goal_pairs(occupied, body_radius_m=0.25)
    assert first == select_fixed_goal_pairs(occupied, body_radius_m=0.25)
    pairs, rejections = first
    assert rejections == ["no_connected_goal_with_2_m_altitude_change"]
    assert len(pairs) == 1
    assert pairs[0][1] == (96, 28, 96)
    assert pairs[0][2] != pairs[0][1]
    census = topology_census(occupied, body_radius_m=0.25)
    assert census["free_components_6"] == 1
    assert census["start_clearance_component_voxels"] > 0


def test_blocked_source_start_is_recorded_not_repaired() -> None:
    occupied = np.ones(EXPECTED_EXTENT, dtype=np.uint8)
    pairs, rejections = select_fixed_goal_pairs(occupied, body_radius_m=0.25)
    assert pairs == []
    assert rejections == ["source_start_fails_clearance"]


def test_real_source_repeat_has_identical_content_hashes(tmp_path: Path) -> None:
    path = os.environ.get("CAVEDRONE_SOURCE")
    if not path:
        pytest.skip("external pinned CaveDroneSim checkout is not present in CI")
    source = Path(path)
    first = export_seed(source, tmp_path / "first", seed=1, partition="test")
    second = export_seed(source, tmp_path / "second", seed=1, partition="test")
    for key in (
        "source_identity_sha256", "world_identity_sha256",
        "dataset_identity_sha256", "occupancy_sha256", "task_ids",
        "reference_ids", "routes", "topology_census",
    ):
        assert first[key] == second[key]
    assert (tmp_path / "first" / "occupancy.npy").read_bytes() == (
        tmp_path / "second" / "occupancy.npy"
    ).read_bytes()
    world = read_sidecar(tmp_path / "first" / "world.json", RoutingWorldRecord)
    task = read_sidecar(tmp_path / "first" / "task-00.json", RoutingTaskRecord)
    reference = read_sidecar(tmp_path / "first" / "reference-00.json", RoutingReferenceRecord)
    assert task.world_identity_sha256 == world.identity_sha256
    assert reference.task_identity_sha256 == task.identity_sha256
    assert reference.claim == "independently_validated"
    assert task.provenance == "derived"
    assert task.family == "drone_flight"
    assert first["cross_family_holdout_supported"] is False


def test_invalid_source_revision_fails_closed(tmp_path: Path) -> None:
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        export_seed(tmp_path, tmp_path / "out", seed=1, partition="test")
    assert not (tmp_path / "out").exists()


def test_legacy_calibration_partition_is_rejected_not_silently_accepted(tmp_path: Path) -> None:
    """See #493: `validation` replaces `calibration`; export must reject the old name."""

    with pytest.raises(ValueError, match="train, validation or test"):
        export_seed(tmp_path, tmp_path / "out", seed=1, partition="calibration")
    assert not (tmp_path / "out").exists()


def test_validation_partition_is_accepted_past_input_checks(tmp_path: Path) -> None:
    """Partition validation runs before touching the source, so this needs no fixture."""

    with pytest.raises((subprocess.CalledProcessError, ValueError)) as excinfo:
        export_seed(tmp_path, tmp_path / "out", seed=1, partition="validation")
    assert "partition" not in str(excinfo.value)
