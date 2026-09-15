"""Execute and freeze the P0C/P0D foundation for voxel pilot v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from theseo_anysearch.garden.pilots.benchmark import ProbeProtocol
from theseo_anysearch.garden.pilots.contracts import (
    ArtifactReference,
    ScoreAnchor,
    SeedAssignments,
    SpecsReference,
    T3HealthGate,
    V2FrozenPreregistration,
    VetoThresholds,
)
from theseo_anysearch.garden.pilots.diagnostics import classify_t3_replay, run_t3_cell
from theseo_anysearch.garden.pilots.io import contract_sha256, write_contract
from theseo_anysearch.garden.pilots.v2 import (
    V2_CALIBRATION_RUN_ID,
    V2_DATASET_ID,
    V2_DIAGNOSTIC_RUN_ID,
    V2_P1_RUN_ID,
    V2_PREREGISTRATION_ID,
    build_v2_pool_identities,
    fit_pca_control,
    frequency_control,
    measure_score_anchors,
    random_projection_control,
    train_supervised_reference,
    v2_query_plan,
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
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path, role: str, *, root: Path) -> ArtifactReference:
    try:
        uri = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        uri = path.resolve().as_posix()
    return ArtifactReference(
        role=role,
        uri=uri,
        sha256=_sha256(path),
        size_bytes=path.stat().st_size,
        media_type="application/json" if path.suffix == ".json" else "application/x-pytorch",
    )


def _context(config: dict[str, Any]) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    records, pools, fresh_draws = build_v2_pool_identities(seed=int(config["generator_seed"]))
    by_id = {record.geometry_id: record for record in records}
    descriptors = {
        pool: [by_id[geometry_id] for geometry_id in identity.geometry_ids]
        for pool, identity in pools.items()
    }
    return records, descriptors, {"pools": pools, "fresh_draws": fresh_draws}


def run_p0c(
    config: dict[str, Any],
    *,
    raw_dir: Path,
    results_dir: Path,
    repo_root: Path,
    device: torch.device,
) -> dict[str, object]:
    started = time.perf_counter()
    _, descriptors, identities = _context(config)
    raw_dir.mkdir(parents=True, exist_ok=True)
    query_path = raw_dir / "v2-queries.json"
    _write_json(
        query_path,
        {
            pool: v2_query_plan(pool, identity.geometry_ids)
            for pool, identity in identities["pools"].items()
        },
    )
    controls = {
        "frequency": frequency_control(),
        "pca": fit_pca_control(
            descriptors["pilot_train"], seed=int(config["random_projection_seed"])
        ),
        "fixed_random_projection": random_projection_control(
            int(config["random_projection_seed"])
        ),
    }
    artifacts: list[ArtifactReference] = [_artifact(query_path, "v2_queries", root=repo_root)]
    for name, model in controls.items():
        path = raw_dir / f"{name}-state.pt"
        torch.save(model.state_dict(), path)
        artifacts.append(_artifact(path, f"{name}_model", root=repo_root))

    reference_config = config["supervised_reference"]
    reference = train_supervised_reference(
        descriptors["pilot_train"],
        device=device,
        seed=int(reference_config["seed"]),
        updates=int(reference_config["updates"]),
        learning_rate=float(reference_config["learning_rate"]),
        selection_interval=int(reference_config["selection_interval"]),
    )
    reference_path = raw_dir / "supervised-reference-state.pt"
    torch.save(reference.model.state_dict(), reference_path)
    artifacts.append(_artifact(reference_path, "supervised_reference_model", root=repo_root))
    protocol = ProbeProtocol(**config["probe"])
    anchors, evaluations, denominator_failures = measure_score_anchors(
        controls,
        reference.model,
        descriptors["pilot_train"],
        descriptors["pilot_calibration"],
        protocol=protocol,
        seed=3272,
        device=device,
    )
    evaluation_path = raw_dir / "p0c-evaluations.json"
    _write_json(evaluation_path, evaluations)
    artifacts.append(_artifact(evaluation_path, "calibration_predictions", root=repo_root))
    elapsed_hours = (time.perf_counter() - started) / 3_600
    cap_passed = elapsed_hours <= float(config["calibration_cap_hours"])
    passed = cap_passed and not denominator_failures
    report: dict[str, object] = {
        "issue": 327,
        "run_id": V2_CALIBRATION_RUN_ID,
        "status": "passed" if passed else "blocked",
        "code_sha": _git("rev-parse", "HEAD"),
        "spec_commit": config["spec_commit"],
        "dataset_id": V2_DATASET_ID,
        "dataset_sha256": _canonical_sha(
            {name: identity.model_dump(mode="json") for name, identity in identities["pools"].items()}
        ),
        "pool_hashes": {
            name: {
                "assignment_sha256": identity.assignment_sha256,
                "query_sha256": identity.query_sha256,
            }
            for name, identity in identities["pools"].items()
        },
        "anchor_selection_without_calibration": True,
        "calibration_used_for_candidate_ranking": False,
        "anchors": {name: anchor.model_dump(mode="json") for name, anchor in anchors.items()},
        "measured_components": {
            name: result["real"]["components"] for name, result in evaluations.items()
        },
        "supervised_reference": {
            "selected_update": reference.selected_update,
            "selection_error": reference.selection_error,
            "residual_training_error": reference.residual_training_error,
            "curve": list(reference.curve),
            "trainable_parameters": reference.trainable_parameters,
            "tiny_parameters": reference.tiny_parameters,
            "capacity_ratio": reference.trainable_parameters / reference.tiny_parameters,
            "selection_pool": "geometry-disjoint fold within pilot_train",
        },
        "denominator_gates": {
            name: name not in denominator_failures for name in (
                "occupied_iou",
                "boundary_f1",
                "clearance_nmae",
                "reachability_auprc",
                "geodesic_nmae",
            )
        },
        "denominator_failures": denominator_failures,
        "accelerator_hours": elapsed_hours,
        "accelerator_hour_cap": config["calibration_cap_hours"],
        "cap_passed": cap_passed,
        "artifacts": [artifact.model_dump(mode="json") for artifact in artifacts],
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "decision": {
            "decision": "tie" if passed else "blocked",
            "reason": (
                "all measured floor/ceiling denominators and the separate P0C cap passed"
                if passed
                else "one or more measured floor/ceiling denominator gates failed"
                if denominator_failures
                else "P0C exceeded its preregistered accelerator-hour cap"
            ),
            "next": "P0D" if passed else None,
        },
    }
    report["report_payload_sha256"] = _canonical_sha(report)
    _write_json(results_dir / "p0c-report.json", report)
    return report


def run_p0d(
    config: dict[str, Any],
    p0c: dict[str, object],
    *,
    raw_dir: Path,
    results_dir: Path,
    repo_root: Path,
    device: torch.device,
) -> tuple[dict[str, object], Path]:
    if p0c["status"] != "passed":
        raise RuntimeError("P0D cannot start because P0C is blocked")
    started = time.perf_counter()
    _, descriptors, _ = _context(config)
    anchors = {
        name: ScoreAnchor.model_validate(value) for name, value in p0c["anchors"].items()
    }
    protocol = ProbeProtocol(**config["probe"])
    cells = [
        run_t3_cell(
            descriptors["pilot_train"],
            descriptors["pilot_diagnostic"],
            anchors,
            learning_rate=learning_rate,
            protocol=protocol,
            device=device,
        )
        for learning_rate in (0.0001, 0.0003)
    ]
    classification = classify_t3_replay(cells)
    elapsed_hours = (time.perf_counter() - started) / 3_600
    cap_passed = elapsed_hours <= float(config["diagnostic_cap_hours"])
    passed = bool(classification["passed"]) and cap_passed
    raw_path = raw_dir / "p0d-full-report.json"
    raw_report = {
        "run_id": V2_DIAGNOSTIC_RUN_ID,
        "cells": cells,
        "classification": classification,
        "accelerator_hours": elapsed_hours,
    }
    _write_json(raw_path, raw_report)
    artifact = _artifact(raw_path, "t3_diagnostic_report", root=repo_root)
    report: dict[str, object] = {
        "issue": 327,
        "run_id": V2_DIAGNOSTIC_RUN_ID,
        "status": "passed" if passed else "blocked",
        "code_sha": _git("rev-parse", "HEAD"),
        "spec_commit": config["spec_commit"],
        "dataset_id": V2_DATASET_ID,
        "classification": classification,
        "cells": [
            {
                "learning_rate": cell["learning_rate"],
                "updates": cell["updates"],
                "telemetry_rows": len(cell["telemetry"]),
                "probe_checkpoints": [row["update"] for row in cell["frozen_probes"]],
                "health_labels": cell["health_labels"],
                "implementation_errors": cell["implementation_errors"],
                "wall_seconds": cell["wall_seconds"],
                "peak_allocated_bytes": cell["peak_allocated_bytes"],
                "final_loss": cell["telemetry"][-1]["pretext_loss"],
                "minimum_loss_after_1000": min(
                    row["pretext_loss"] for row in cell["telemetry"] if row["update"] >= 1_000
                ),
            }
            for cell in cells
        ],
        "accelerator_hours": elapsed_hours,
        "accelerator_hour_cap": config["diagnostic_cap_hours"],
        "cap_passed": cap_passed,
        "artifact": artifact.model_dump(mode="json"),
        "decision": {
            "decision": "tie" if passed else "blocked",
            "reason": (
                "both T3 mechanism-health cells and the separate P0D cap passed"
                if passed
                else "T3 implementation/mechanism health or the P0D cap failed"
            ),
            "next": "replacement P1" if passed else None,
        },
    }
    report["report_payload_sha256"] = _canonical_sha(report)
    _write_json(results_dir / "p0d-report.json", report)
    return report, raw_path


def freeze_preregistration(
    config: dict[str, Any],
    p0c: dict[str, object],
    p0d: dict[str, object],
    *,
    raw_dir: Path,
    results_dir: Path,
    repo_root: Path,
) -> V2FrozenPreregistration:
    if p0c["status"] != "passed" or p0d["status"] != "passed":
        raise RuntimeError("replacement P1 preregistration is blocked by P0C/P0D")
    _, _, identities = _context(config)
    calibration_artifacts = tuple(
        ArtifactReference.model_validate(value) for value in p0c["artifacts"]
    )
    diagnostic_artifact = ArtifactReference.model_validate(p0d["artifact"])
    classification = p0d["classification"]
    preregistration = V2FrozenPreregistration(
        dataset_id=V2_DATASET_ID,
        preregistration_id=V2_PREREGISTRATION_ID,
        calibration_run_id=V2_CALIBRATION_RUN_ID,
        diagnostic_run_id=V2_DIAGNOSTIC_RUN_ID,
        replacement_p1_run_id=V2_P1_RUN_ID,
        superseded_specs_sha=config["superseded_spec_commit"],
        frozen_at=datetime.fromisoformat(config["frozen_at"].replace("Z", "+00:00")),
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
        diagnostic_cap_hours=config["diagnostic_cap_hours"],
        p1_cap_hours=config["p1_cap_hours"],
        seeds=SeedAssignments(),
        vetoes=VetoThresholds(),
        score_anchors={
            name: ScoreAnchor.model_validate(value) for name, value in p0c["anchors"].items()
        },
        pools=identities["pools"],
        fresh_draws=identities["fresh_draws"],
        calibration_artifacts=calibration_artifacts,
        t3_health=T3HealthGate(
            run_id=V2_DIAGNOSTIC_RUN_ID,
            implementation_failure=classification["implementation_failure"],
            mechanism_health_failure=classification["mechanism_health_failure"],
            labels_by_learning_rate={
                key: tuple(value)
                for key, value in classification["labels_by_learning_rate"].items()
            },
            shared_labels=tuple(classification["shared_labels"]),
            report=diagnostic_artifact,
        ),
    )
    identity = write_contract(results_dir / "v2-preregistration.yaml", preregistration)
    summary = {
        "preregistration_id": V2_PREREGISTRATION_ID,
        "identity_sha256": identity,
        "contract_sha256": contract_sha256(preregistration),
        "replacement_p1_run_id": V2_P1_RUN_ID,
        "status": "frozen",
    }
    _write_json(results_dir / "v2-preregistration-summary.json", summary)
    return preregistration


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("P0C", "P0D", "all"), default="all")
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("runtime/perception_encoder/v2_foundation")
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/perception_encoder/results/v2_foundation"),
    )
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("v2 foundation execution requires CUDA")
    config = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    expected = {
        "program": "voxel-encoder-pilot-v2",
        "dataset_id": V2_DATASET_ID,
        "preregistration_id": V2_PREREGISTRATION_ID,
        "calibration_run_id": V2_CALIBRATION_RUN_ID,
        "diagnostic_run_id": V2_DIAGNOSTIC_RUN_ID,
        "replacement_p1_run_id": V2_P1_RUN_ID,
    }
    if any(config[key] != value for key, value in expected.items()):
        raise ValueError("v2 foundation configuration contains an incorrect frozen identity")
    device = torch.device("cuda:0")
    repo_root = Path(_git("rev-parse", "--show-toplevel"))
    p0c_path = arguments.results_dir / "p0c-report.json"
    if arguments.stage in {"P0C", "all"}:
        p0c = run_p0c(
            config,
            raw_dir=arguments.raw_dir,
            results_dir=arguments.results_dir,
            repo_root=repo_root,
            device=device,
        )
        print(json.dumps(p0c["decision"], sort_keys=True))
    else:
        p0c = json.loads(p0c_path.read_text(encoding="utf-8"))
    if arguments.stage in {"P0D", "all"}:
        p0d, _ = run_p0d(
            config,
            p0c,
            raw_dir=arguments.raw_dir,
            results_dir=arguments.results_dir,
            repo_root=repo_root,
            device=device,
        )
        print(json.dumps(p0d["decision"], sort_keys=True))
        freeze_preregistration(
            config,
            p0c,
            p0d,
            raw_dir=arguments.raw_dir,
            results_dir=arguments.results_dir,
            repo_root=repo_root,
        )


if __name__ == "__main__":
    main()
