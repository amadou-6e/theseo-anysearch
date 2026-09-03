"""Integration tests for the amended v2r1 calibration foundation (F8)."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from experiments.perception_encoder.v2r1_foundation import freeze_protocol, run_cpu_smoke
from theseo_anysearch.garden.pilots.calibration_revision import (
    calibrate_revised_anchors,
    deterministic_smoke_datasets,
)
from theseo_anysearch.garden.pilots.io import contract_sha256, read_contract
from theseo_anysearch.garden.pilots.contracts import V2R1ProtocolPreregistration
from theseo_anysearch.garden.pilots.v2 import v2_geometry_records
from theseo_anysearch.garden.pilots.v2r1 import (
    build_v2r1_pool_identities,
    v2r1_geometry_records,
)


CONFIG = Path("experiments/perception_encoder/v2r1-foundation-config.yaml")


def _config() -> dict[str, object]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_v2r1_pool_ids_are_fresh_and_disjoint_from_v2() -> None:
    records, pools, _ = build_v2r1_pool_identities(seed=3290)
    ids = {record.geometry_id for record in records}
    assert len(ids) == 248
    assert ids.isdisjoint({record.geometry_id for record in v2_geometry_records()})
    assert all(geometry_id.startswith("pilot-v2r1-") for geometry_id in ids)
    assert sum(len(pool.geometry_ids) for pool in pools.values()) == 248


def test_protocol_freeze_round_trips_and_pins_amended_spec(tmp_path: Path) -> None:
    protocol = freeze_protocol(_config(), results_dir=tmp_path)
    restored = read_contract(
        tmp_path / "v2r1-protocol-preregistration.yaml",
        V2R1ProtocolPreregistration,
    )
    assert restored == protocol
    assert protocol.specs.commit_sha == "0c9e3c633799f5d42b7a603e0845cac0bd494cda"
    assert protocol.metric_plans["geodesic_nmae"].status == "deferred"
    assert contract_sha256(protocol)


def test_protocol_rejects_a_reused_v2_pool(tmp_path: Path) -> None:
    protocol = freeze_protocol(_config(), results_dir=tmp_path)
    _, old_pools, _ = build_v2r1_pool_identities(seed=3290)
    first = old_pools["pilot_train"]
    old_pools["pilot_train"] = first.model_copy(
        update={"geometry_ids": tuple(value.replace("pilot-v2r1-", "pilot-v2-") for value in first.geometry_ids)}
    )
    with pytest.raises(ValueError, match="must be fresh"):
        protocol.model_copy(update={"pools": old_pools}).model_validate(
            protocol.model_copy(update={"pools": old_pools}).model_dump()
        )


def test_full_revised_anchor_path_passes_deterministic_cpu_fixture() -> None:
    anchors, diagnostics = calibrate_revised_anchors(deterministic_smoke_datasets())
    assert set(anchors) == {
        "occupied_iou",
        "boundary_f1",
        "clearance_nmae",
        "reachability_auprc",
        "geodesic_nmae",
    }
    assert anchors["geodesic_nmae"].status == "deferred"
    assert all(anchors[name].status == "active" for name in diagnostics if name != "geodesic_nmae")


def test_cpu_smoke_content_addresses_report(tmp_path: Path) -> None:
    report = run_cpu_smoke(_config(), results_dir=tmp_path)
    assert report["status"] == "passed"
    assert len(report["report_payload_sha256"]) == 64
    assert (tmp_path / "v2r1-cpu-smoke-report.json").is_file()
