"""Freeze and smoke-test the amended v2r1 calibration protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from theseo_anysearch.garden.pilots.calibration_revision import (
    calibrate_revised_anchors,
    deterministic_smoke_datasets,
)
from theseo_anysearch.garden.pilots.contracts import (
    ArtifactReference,
    SpecsReference,
    SupersededVerdict,
    V2R1MetricPlan,
    V2R1ProtocolPreregistration,
)
from theseo_anysearch.garden.pilots.io import contract_sha256, write_contract
from theseo_anysearch.garden.pilots.v2r1 import (
    V2R1_DATASET_ID,
    build_v2r1_pool_identities,
    v2r1_query_plan,
)


P0C_REPORT_SHA = "a7a149f9235b38f9ff1f1a230ce791367cf528fc6705e0f987674f1a48d4ea43"


def _canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _artifact(path: Path, role: str) -> ArtifactReference:
    return ArtifactReference(
        role=role,
        uri=path.as_posix(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
        media_type="application/json",
    )


def _metric_plans() -> dict[str, V2R1MetricPlan]:
    classification = {
        "higher_is_better": True,
        "floor_methods": ("frequency", "coordinates_only_ridge", "fixed_random_projection"),
        "ceiling_method": "bayes_error_knn",
        "null_input": "coordinates_only",
        "min_pvi_gain": 0.05,
        "minimum_absolute_headroom": 0.10,
    }
    return {
        "occupied_iou": V2R1MetricPlan(**classification),
        "boundary_f1": V2R1MetricPlan(**classification),
        "clearance_nmae": V2R1MetricPlan(
            higher_is_better=False,
            floor_methods=("frequency", "coordinates_only_ridge", "fixed_random_projection"),
            ceiling_method="knn_residual",
            null_input="coordinates_only",
            min_pvi_gain=0.05,
            minimum_relative_error_reduction=0.20,
            status="active",
        ),
        "reachability_auprc": V2R1MetricPlan(**classification),
        "geodesic_nmae": V2R1MetricPlan(
            higher_is_better=False,
            floor_methods=("frequency",),
            ceiling_method="knn_residual",
            null_input="coordinates_only",
            min_pvi_gain=0.05,
            status="deferred",
            deferral_reason=(
                "frequency NMAE is below 0.15 at pilot radii and the supervised "
                "reference does not improve it; revisit at Stage 2 wide context"
            ),
        ),
    }


def _validate_config(config: dict[str, Any]) -> None:
    expected = {
        "program": "voxel-encoder-pilot-v2r1",
        "dataset_id": V2R1_DATASET_ID,
        "preregistration_id": "voxel-encoder-pilot-v2r1-preregistration-1",
        "calibration_run_id": "voxel-encoder-pilot-v2r1-p0c-1",
        "data_sensitivity_run_id": "voxel-encoder-pilot-v2r1-p0d-1",
        "replacement_p1_run_id": "voxel-encoder-pilot-v2r1-p1-1",
        "p0d_observation_density_multiplier": 4,
        "reachability_false_open_margin": 0.02,
    }
    differences = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if differences:
        raise ValueError(f"v2r1 configuration contains incorrect frozen identities: {differences}")


def freeze_protocol(config: dict[str, Any], *, results_dir: Path) -> V2R1ProtocolPreregistration:
    """Write query identities and the pre-open protocol as immutable contracts."""

    _validate_config(config)
    _, pools, fresh_draws = build_v2r1_pool_identities(seed=int(config["generator_seed"]))
    query_path = results_dir / "v2r1-query-plan.json"
    _write_json(
        query_path,
        {pool: v2r1_query_plan(pool, identity.geometry_ids) for pool, identity in pools.items()},
    )
    frozen_at = datetime.fromisoformat(str(config["frozen_at"]).replace("Z", "+00:00"))
    protocol = V2R1ProtocolPreregistration(
        dataset_id=config["dataset_id"],
        preregistration_id=config["preregistration_id"],
        calibration_run_id=config["calibration_run_id"],
        data_sensitivity_run_id=config["data_sensitivity_run_id"],
        replacement_p1_run_id=config["replacement_p1_run_id"],
        frozen_at=frozen_at,
        specs=SpecsReference(
            repository="https://github.com/amadou-6e/specs",
            commit_sha=config["spec_commit"],
            files=(
                "projects/theseo-anysearch/python/perception-encoder-pilots.md",
                "projects/theseo-anysearch/python/perception-encoder-pretraining.md",
                "projects/theseo-anysearch/python/perception-encoders.md",
            ),
        ),
        generator_version=config["generator_version"],
        generator_seed=config["generator_seed"],
        calibration_cap_hours=config["calibration_cap_hours"],
        data_sensitivity_cap_hours=config["data_sensitivity_cap_hours"],
        p1_cap_hours=config["p1_cap_hours"],
        metric_plans=_metric_plans(),
        active_gate_components=(
            "occupied_iou",
            "boundary_f1",
            "clearance_nmae",
            "reachability_auprc",
        ),
        superseded_verdicts=(
            SupersededVerdict(
                superseded_program="voxel-encoder-pilot-v1",
                superseded_run_id="voxel-encoder-pilot-v1-p1-1",
                superseded_pilot="P1",
                superseded_decision="no_viable_direction",
                fired_veto="false_open_rate_max",
                void_reason=(
                    "P0C showed the absolute false-open veto and aggregate score "
                    "were structurally miscalibrated"
                ),
                evidence_run_id="voxel-encoder-pilot-v2-p0-calibration-1",
                evidence_report_sha256=P0C_REPORT_SHA,
                replacement_run_id=config["replacement_p1_run_id"],
                recorded_at=frozen_at,
            ),
        ),
        pools=pools,
        fresh_draws=fresh_draws,
        query_artifacts=(_artifact(query_path, "v2r1_query_plan"),),
    )
    write_contract(results_dir / "v2r1-protocol-preregistration.yaml", protocol)
    return protocol


def run_cpu_smoke(config: dict[str, Any], *, results_dir: Path) -> dict[str, object]:
    """Execute the same anchor integration path on deterministic fixtures."""

    protocol = freeze_protocol(config, results_dir=results_dir)
    anchors, diagnostics = calibrate_revised_anchors(
        deterministic_smoke_datasets(seed=int(config["cpu_smoke_seed"])),
        seed=int(config["cpu_smoke_seed"]),
    )
    report: dict[str, object] = {
        "stage": "F8 deterministic CPU smoke",
        "status": "passed",
        "protocol_sha256": contract_sha256(protocol),
        "dataset_id": protocol.dataset_id,
        "run_id": protocol.calibration_run_id,
        "spec_commit": protocol.specs.commit_sha,
        "active_gate_components": list(protocol.active_gate_components),
        "anchors": {name: anchor.model_dump(mode="json") for name, anchor in anchors.items()},
        "diagnostics": diagnostics,
    }
    report["report_payload_sha256"] = _canonical_sha(report)
    _write_json(results_dir / "v2r1-cpu-smoke-report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("freeze", "smoke"), default="smoke")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("runtime/perception_encoder/v2r1_foundation"),
    )
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    if arguments.stage == "freeze":
        protocol = freeze_protocol(config, results_dir=arguments.results_dir)
        print(contract_sha256(protocol))
    else:
        report = run_cpu_smoke(config, results_dir=arguments.results_dir)
        print(json.dumps({"status": report["status"], "report_payload_sha256": report["report_payload_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
