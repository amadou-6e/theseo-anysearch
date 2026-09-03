"""Execute the amended v2r1 P0C calibration on its fresh frozen pools."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from pydantic import ValidationError

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from theseo_anysearch.garden.evaluation.reachability import (
    calibrate_decision_threshold,
    derive_false_open_veto,
    false_open_false_closed,
    per_bin_auprc,
    two_way_agreement,
)
from theseo_anysearch.garden.pilots.calibration_revision import (
    ACTIVE_COMPONENTS,
    control_predictions,
    deferred_geodesic_anchor,
    measure_revised_component,
)
from theseo_anysearch.garden.pilots.contracts import (
    ArtifactReference,
    RevisedScoreAnchor,
    SeedAssignments,
    V2R1FrozenPreregistration,
    V2R1ProtocolPreregistration,
    V2R1VetoThresholds,
)
from theseo_anysearch.garden.pilots.io import contract_sha256, read_contract, write_contract
from theseo_anysearch.garden.pilots.v2r1 import build_v2r1_pool_identities
from theseo_anysearch.garden.pilots.v2r1_data import (
    ReachabilityMetadata,
    materialize_v2r1_calibration_datasets,
)


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


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


def _artifact(path: Path, role: str, media_type: str) -> ArtifactReference:
    return ArtifactReference(
        role=role,
        uri=path.as_posix(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
        media_type=media_type,
    )


def _descriptor_pools(protocol: V2R1ProtocolPreregistration) -> dict[str, list[Any]]:
    records, pools, _ = build_v2r1_pool_identities(seed=protocol.generator_seed)
    for name, identity in pools.items():
        if identity != protocol.pools[name]:
            raise RuntimeError(f"materialized {name} identity differs from preregistration")
    by_id = {record.geometry_id: record for record in records}
    return {
        name: [by_id[geometry_id] for geometry_id in identity.geometry_ids]
        for name, identity in pools.items()
    }


def _query_counts(query_plan: dict[str, Any], pool: str) -> tuple[int, int]:
    rows = {row["probe"]: int(row["count"]) for row in query_plan[pool]}
    if set(rows) != {"coordinate", "pair"}:
        raise ValueError(f"{pool} query plan must contain coordinate and pair probes")
    return rows["coordinate"], rows["pair"]


def _save_dataset_artifact(
    path: Path, datasets: dict[str, Any], metadata: ReachabilityMetadata
) -> None:
    values: dict[str, np.ndarray] = {
        "reachability_distance_bins": metadata.distance_bins,
        "reachability_kinds": metadata.kinds.astype("U32"),
    }
    for component, dataset in datasets.items():
        prefix = component.replace("_", "-")
        values[f"{prefix}-train-context"] = dataset.train_context
        values[f"{prefix}-train-null"] = dataset.train_null
        values[f"{prefix}-train-targets"] = dataset.train_targets
        values[f"{prefix}-evaluation-context"] = dataset.evaluation_context
        values[f"{prefix}-evaluation-null"] = dataset.evaluation_null
        values[f"{prefix}-evaluation-targets"] = dataset.evaluation_targets
        values[f"{prefix}-evaluation-geometry-ids"] = np.asarray(
            dataset.evaluation_geometry_ids, dtype="U40"
        )
        for control, array in dataset.train_controls.items():
            values[f"{prefix}-train-control-{control}"] = array
        for control, array in dataset.evaluation_controls.items():
            values[f"{prefix}-evaluation-control-{control}"] = array
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **values)


def _reachability_operating_point(dataset: Any, metadata: ReachabilityMetadata) -> dict[str, Any]:
    geometry_ids = np.asarray(dataset.evaluation_geometry_ids)
    unique = sorted(set(dataset.evaluation_geometry_ids))
    selection_ids = set(unique[::2])
    selection = np.asarray([value in selection_ids for value in geometry_ids])
    report = ~selection
    target = np.asarray(dataset.evaluation_targets, dtype=bool)
    baselines: dict[str, dict[str, Any]] = {}
    false_open_rates: dict[str, float] = {}
    for name in sorted(dataset.train_controls):
        scores = control_predictions(dataset, name, classification=True)
        threshold = calibrate_decision_threshold(scores[selection], target[selection])
        false_open, false_closed = false_open_false_closed(
            scores[report], target[report], threshold
        )
        reverse = scores.copy()
        agreement = two_way_agreement(scores[report], reverse[report], threshold)
        baselines[name] = {
            "threshold": threshold,
            "false_open": false_open,
            "false_closed": false_closed,
            "per_bin_auprc": per_bin_auprc(
                scores[report], target[report], metadata.distance_bins[report]
            ),
            "two_way_agreement_fraction": float(
                np.mean(agreement == (scores[report] >= threshold))
            ),
        }
        false_open_rates[name] = false_open
    veto = derive_false_open_veto(false_open_rates, margin=0.02)
    return {
        "selection_geometry_ids": sorted(selection_ids),
        "report_geometry_ids": sorted(set(unique) - selection_ids),
        "baselines": baselines,
        "veto": {
            "threshold": veto.threshold,
            "best_baseline": veto.best_baseline,
            "best_baseline_false_open": veto.best_baseline_false_open,
            "margin": veto.margin,
            "method": veto.method,
        },
        "pair_kind_counts": {
            str(kind): int(np.count_nonzero(metadata.kinds == kind))
            for kind in np.unique(metadata.kinds)
        },
    }


def execute_p0c(
    config: dict[str, Any], *, raw_dir: Path, results_dir: Path
) -> dict[str, Any]:
    started = time.perf_counter()
    protocol_path = Path(config["protocol_path"])
    protocol = read_contract(protocol_path, V2R1ProtocolPreregistration)
    if protocol.specs.commit_sha != config["spec_commit"]:
        raise ValueError("config and protocol specification commits differ")
    query_path = Path(config["query_plan_path"])
    query_artifact = _artifact(query_path, "v2r1_query_plan", "application/json")
    if query_artifact.sha256 != protocol.query_artifacts[0].sha256:
        raise ValueError("query plan differs from the frozen protocol artifact")
    query_plan = json.loads(query_path.read_text(encoding="utf-8"))
    coordinate_train, pair_train = _query_counts(query_plan, "pilot_train")
    coordinate_evaluation, pair_evaluation = _query_counts(
        query_plan, "pilot_calibration"
    )
    descriptors = _descriptor_pools(protocol)
    datasets, reachability_metadata = materialize_v2r1_calibration_datasets(
        descriptors["pilot_train"],
        descriptors["pilot_calibration"],
        coordinate_train_queries=coordinate_train,
        coordinate_evaluation_queries=coordinate_evaluation,
        pair_train_queries=pair_train,
        pair_evaluation_queries=pair_evaluation,
        seed=int(config["execution_seed"]),
    )
    dataset_path = raw_dir / "v2r1-p0c-datasets.npz"
    _save_dataset_artifact(dataset_path, datasets, reachability_metadata)
    dataset_artifact = _artifact(dataset_path, "v2r1_p0c_datasets", "application/x-npz")

    anchors: dict[str, RevisedScoreAnchor] = {}
    measurements: dict[str, Any] = {}
    failures: dict[str, str] = {}
    for index, component in enumerate(ACTIVE_COMPONENTS):
        print(f"calibrating {component}", flush=True)
        payload, diagnostics = measure_revised_component(
            component, datasets[component], seed=int(config["execution_seed"]) + index
        )
        serializable = dict(payload)
        serializable["triviality"] = payload["triviality"].model_dump(mode="json")
        measurements[component] = {"anchor": serializable, "diagnostics": diagnostics}
        try:
            anchors[component] = RevisedScoreAnchor(**payload)
        except ValidationError as error:
            failures[component] = str(error)
    geodesic = deferred_geodesic_anchor()
    anchors["geodesic_nmae"] = geodesic
    measurements["geodesic_nmae"] = {
        "anchor": geodesic.model_dump(mode="json"),
        "diagnostics": {"status": "deferred", "reason": geodesic.deferral_reason},
    }
    reachability = _reachability_operating_point(
        datasets["reachability_auprc"], reachability_metadata
    )
    elapsed_hours = (time.perf_counter() - started) / 3_600
    cap_passed = elapsed_hours <= protocol.calibration_cap_hours
    passed = not failures and cap_passed
    report: dict[str, Any] = {
        "issue": 333,
        "run_id": protocol.calibration_run_id,
        "status": "passed" if passed else "blocked",
        "disposition": "retain" if passed else "reject",
        "integration_base_sha": config["integration_base_sha"],
        "code_sha": _git("rev-parse", "HEAD"),
        "spec_commit": protocol.specs.commit_sha,
        "protocol_sha256": contract_sha256(protocol),
        "dataset_id": protocol.dataset_id,
        "dataset_artifact": dataset_artifact.model_dump(mode="json"),
        "query_artifact": query_artifact.model_dump(mode="json"),
        "query_counts": {
            "coordinate_train": coordinate_train,
            "coordinate_calibration": coordinate_evaluation,
            "pair_train": pair_train,
            "pair_calibration": pair_evaluation,
        },
        "measurements": measurements,
        "denominator_failures": failures,
        "reachability": reachability,
        "accelerator_hours": 0.0,
        "wall_hours": elapsed_hours,
        "accelerator_hour_cap": protocol.calibration_cap_hours,
        "cap_passed": cap_passed,
        "hardware": {
            "cpu": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "decision": {
            "decision": "tie" if passed else "blocked",
            "reason": (
                "all active revised denominators and triviality gates passed"
                if passed
                else "one or more revised denominator, triviality, or cap gates failed"
            ),
            "next": "E2 P0D" if passed else None,
        },
    }
    report["report_payload_sha256"] = _canonical_sha(report)
    _write_json(results_dir / "p0c-report.json", report)

    if passed:
        veto = reachability["veto"]
        preregistration = V2R1FrozenPreregistration(
            dataset_id=protocol.dataset_id,
            preregistration_id=protocol.preregistration_id,
            calibration_run_id=protocol.calibration_run_id,
            data_sensitivity_run_id=protocol.data_sensitivity_run_id,
            replacement_p1_run_id=protocol.replacement_p1_run_id,
            superseded_calibration_run_id="voxel-encoder-pilot-v2-p0-calibration-1",
            superseded_specs_sha="01eefc529016da48c4a1dd17b85391720542af14",
            frozen_at=datetime.now(timezone.utc),
            specs=protocol.specs,
            generator_version=protocol.generator_version,
            generator_seed=protocol.generator_seed,
            calibration_cap_hours=protocol.calibration_cap_hours,
            data_sensitivity_cap_hours=protocol.data_sensitivity_cap_hours,
            p1_cap_hours=protocol.p1_cap_hours,
            seeds=SeedAssignments(),
            vetoes=V2R1VetoThresholds(
                false_open_rate_max=veto["threshold"],
                false_open_baseline=veto["best_baseline_false_open"],
                false_open_baseline_name=veto["best_baseline"],
            ),
            revised_anchors=anchors,
            active_gate_components=protocol.active_gate_components,
            superseded_verdicts=protocol.superseded_verdicts,
            pools=protocol.pools,
            fresh_draws=protocol.fresh_draws,
            calibration_artifacts=(dataset_artifact, query_artifact),
            protocol_sha256=contract_sha256(protocol),
        )
        write_contract(
            results_dir / "v2r1-measured-preregistration.yaml", preregistration
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("runtime/perception_encoder/v2r1_p0c")
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/perception_encoder/results/v2r1_p0c"),
    )
    arguments = parser.parse_args()
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    report = execute_p0c(config, raw_dir=arguments.raw_dir, results_dir=arguments.results_dir)
    print(json.dumps(report["decision"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
