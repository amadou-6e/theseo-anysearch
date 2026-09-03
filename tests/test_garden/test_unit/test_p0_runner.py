"""Contract tests for P0 preregistration and analytic anchors."""
from __future__ import annotations

from pathlib import Path

import yaml

from theseo_anysearch.garden.pilots.runner import (
    build_preregistration,
    calibrate_score_anchors,
    pilot_geometry_records,
)


CONFIG = (
    Path(__file__).parents[3]
    / "experiments"
    / "perception_encoder"
    / "p0-config.yaml"
)


def test_p0_geometry_registry_is_complete_and_explicit_about_import_fixture() -> None:
    records = pilot_geometry_records()
    assert len(records) == len({record.geometry_id for record in records}) == 200
    assert {record.family for record in records} == {
        "open",
        "thin_obstacle",
        "topology",
        "imported",
    }
    imported = [record for record in records if record.family == "imported"]
    assert imported
    assert all("synthetic_mesh_import_fixture" in record.source for record in imported)


def test_p0_analytic_anchors_have_valid_nontrivial_denominators() -> None:
    anchors = calibrate_score_anchors()
    assert set(anchors) == {
        "occupied_iou",
        "boundary_f1",
        "clearance_nmae",
        "reachability_auprc",
        "geodesic_nmae",
    }
    for anchor in anchors.values():
        if anchor.higher_is_better:
            assert anchor.ceiling - anchor.floor >= 0.10
        else:
            assert anchor.floor > anchor.ceiling


def test_p0_preregistration_freezes_exact_pool_and_fresh_draw_identities() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    preregistration = build_preregistration(config)
    assert len(preregistration.pools["pilot_train"].geometry_ids) == 96
    assert len(preregistration.pools["pilot_confirm"].geometry_ids) == 32
    assert preregistration.fresh_draws["P4"].query_sha256 == preregistration.pools[
        "pilot_dev_arch"
    ].query_sha256
    assert preregistration.accelerator_caps.total_comparative_hours == 48
