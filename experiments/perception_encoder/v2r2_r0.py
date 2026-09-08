"""Freeze, execute and assess the registered v2r2 R0 identifiability audit."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import yaml

from theseo_anysearch.garden.pilots.io import contract_sha256, payload_sha256, read_contract, write_contract
from theseo_anysearch.garden.pilots.v2r2_audit import assess_r0
from theseo_anysearch.garden.pilots.v2r2_controls import fit_r0_controls
from theseo_anysearch.garden.pilots.v2r2_data import build_context, identity_plan, native_volume
from theseo_anysearch.garden.pilots.v2r2_protocol import V2R2AuditProtocol


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATHS = ("theseo_anysearch", "experiments/perception_encoder/v2r2_r0.py", "experiments/perception_encoder/v2r2-r0-config.yaml")
_BANKS = None
_WORKER_CONFIG = None


def _git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def resolve_protocol(config: dict, *, executable_sha: str, spec_sha: str) -> V2R2AuditProtocol:
    values = dict(config)
    values["executable_sha"] = executable_sha
    values["execution_addendum_spec_sha"] = spec_sha
    values["pools"] = identity_plan(values["root_seed"], values["training_geometries_per_stratum"],
        values["assessment"]["geometries_per_stratum_per_domain"], values["donor_geometries_per_configuration"])
    values["pool_plan_sha256"] = payload_sha256(values["pools"])
    values["query_plan_sha256"] = payload_sha256({
        "pool_plan_sha256": values["pool_plan_sha256"], "seed": values["root_seed"],
        "r0_rule": values["r0_strata"], "attempts": values["query_attempts"],
        "r1_positive": values["r1_positive_strata"], "r1_negative": values["r1_negative_strata"],
        "generator": values["generator"],
    })
    return V2R2AuditProtocol.model_validate(values)


def _initialize_worker(banks, config):
    global _BANKS, _WORKER_CONFIG
    _BANKS, _WORKER_CONFIG = banks, config


def _build(record):
    return build_context(record, _BANKS[record["configuration"]], **_WORKER_CONFIG)


def write_json_once(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        indent = 2 if path.name in {"r0-report.json", "run-started.json"} else None
        json.dump(value, stream, indent=indent, sort_keys=True, allow_nan=False)
        stream.write("\n")


def select_quota(candidates, quota: int, seen_observations: set[str]):
    """Selection reads eligibility and preassigned strata only, never labels."""
    counts = {}
    selected = []
    exclusions = {}
    for example, record in candidates:
        if example is None:
            exclusions[record["geometry_id"]] = record["status"]
            continue
        key = (example.stratum, example.bootstrap_stratum)
        if counts.get(key, 0) >= quota // 12:
            exclusions[example.geometry_id] = "quota_already_filled"
        elif example.observation_sha256 in seen_observations:
            exclusions[example.geometry_id] = "duplicate_selected_observation"
        else:
            selected.append(example)
            counts[key] = counts.get(key, 0) + 1
            seen_observations.add(example.observation_sha256)
    return tuple(selected), exclusions


def run(protocol: V2R2AuditProtocol, output: Path) -> dict:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("registered R0 requires CUDA; not starting a CPU substitute")
    if _git("diff", protocol.executable_sha, "--", *SOURCE_PATHS):
        raise ValueError("executable sources differ from the frozen implementation SHA")
    subprocess.check_call(["git", "merge-base", "--is-ancestor", protocol.executable_sha, "HEAD"], cwd=ROOT)
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    write_json_once(output / "run-started.json", {"run_id": protocol.r0_run_id,
        "protocol_sha256": contract_sha256(protocol), "started_at": started_at})

    def deadline():
        if time.perf_counter() - started > protocol.r0_wall_cap_seconds:
            raise TimeoutError("R0 wall cap exhausted")

    report = {
        "run_id": protocol.r0_run_id, "dataset_id": protocol.dataset_id,
        "protocol_sha256": contract_sha256(protocol), "code_sha": protocol.executable_sha,
        "runtime_head_sha": _git("rev-parse", "HEAD"),
        "spec_sha": protocol.governing_spec_sha, "execution_addendum_spec_sha": protocol.execution_addendum_spec_sha,
        "pool_plan_sha256": protocol.pool_plan_sha256, "query_plan_sha256": protocol.query_plan_sha256,
        "started_at": started_at, "candidate_training_updates": 0,
        "p0c_started": False, "p0d_started": False, "p1_started": False,
        "limitations": [
            "Conditional entropy is for the registered empirical patch prior, not a true native-world posterior.",
            "Completions are IID conditional on fixed donor banks; uncertainty conditions on those banks.",
            "Imported-family volumes are synthetic ellipsoids, not real imported assets.",
            "R0 bins use observed optimistic paths to avoid conditioning uncertainty on hidden labels; R1 positives use completed paths.",
            "The transfer configuration changes voxel sampling scale, not the whole generator family.",
            "A failed audit is insufficient evidence under this distribution/budget, not proof topology is impossible.",
        ],
    }
    try:
        banks = {}
        for configuration in ("A", "B"):
            print(f"Generating donor bank {configuration}", flush=True)
            volumes = []
            for record in protocol.pools[f"donors_{configuration}"]:
                volumes.append(native_volume(record, configuration))
                deadline()
            banks[configuration] = np.stack(volumes)
        report["donor_bank_sha256"] = {name: hashlib.sha256(bank.tobytes()).hexdigest() for name, bank in banks.items()}
        worker_config = {"seed": protocol.root_seed, "k": protocol.donor_neighbours,
            "completions": protocol.assessment.counterfactuals_per_context, "query_attempts": protocol.query_attempts}
        all_audits = {}
        pools = {}
        seen_observations = set()
        with ProcessPoolExecutor(max_workers=protocol.workers, initializer=_initialize_worker, initargs=(banks, worker_config)) as executor:
            for pool in ("r0_train", "r0_in_domain", "r0_transfer"):
                records = protocol.pools[pool]
                print(f"Generating {pool}: {len(records)} preassigned candidates", flush=True)
                candidates = []
                for i, result in enumerate(executor.map(_build, records, chunksize=4)):
                    candidates.append(result)
                    deadline()
                    if (i + 1) % 48 == 0:
                        print(f"  {pool}: {i + 1}/{len(records)}", flush=True)
                quota = protocol.training_geometries_per_stratum if pool == "r0_train" else protocol.assessment.geometries_per_stratum_per_domain
                pools[pool], exclusions = select_quota(candidates, quota, seen_observations)
                selected_ids = {example.geometry_id for example in pools[pool]}
                all_audits[pool] = [{**audit, "selected": audit["geometry_id"] in selected_ids,
                    "exclusion": exclusions.get(audit["geometry_id"])} for _, audit in candidates]
        write_json_once(output / "data-manifest.json", {"protocol_sha256": contract_sha256(protocol),
            "donor_bank_sha256": report["donor_bank_sha256"], "candidates": all_audits})
        report["data_manifest_sha256"] = hashlib.sha256((output / "data-manifest.json").read_bytes()).hexdigest()
        report["selected_contexts"] = {name: len(rows) for name, rows in pools.items()}
        fits = []
        for pool, domain in (("r0_in_domain", "in_domain"), ("r0_transfer", "heldout_generator")):
            if not pools["r0_train"] or not pools[pool]:
                continue
            print(f"Fitting CUDA controls: {domain}, {protocol.controls.updates} updates/control", flush=True)
            fits.append(fit_r0_controls(pools["r0_train"], pools[pool], fold_id=pool, domain=domain, recipe=protocol.controls, device="cuda"))
            deadline()
        gpu_hours = sum(fit.resources["accelerator_hours_upper_bound"] for fit in fits)
        if gpu_hours > protocol.r0_accelerator_hour_cap:
            raise TimeoutError("R0 accelerator-hour cap exhausted")
        predictions = tuple(row for fit in fits for row in fit.contexts)
        folds = tuple(fit.fold for fit in fits)
        print("Assessing frozen R0 gates with geometry-cluster uncertainty", flush=True)
        assessment = assess_r0(protocol.assessment, predictions, folds)
        if len(pools["r0_train"]) != 3 * protocol.training_geometries_per_stratum:
            assessment["reasons"].append("training:insufficient_support")
            assessment["decision"] = "defer"
            assessment.pop("assessment_payload_sha256")
            assessment["assessment_payload_sha256"] = payload_sha256(assessment)
        write_json_once(output / "predictions.json", {"contexts": [row.model_dump(mode="json") for row in predictions],
            "folds": [fold.model_dump(mode="json") for fold in folds]})
        write_json_once(output / "control-artifacts.json", {"fits": [fit.artifact for fit in fits], "resources": [fit.resources for fit in fits]})
        report.update({
            "status": "completed", "r0_decision": assessment["decision"], "assessment": assessment,
            "decision": "no_topology_identifiable" if assessment["decision"] == "defer" else "proceed_to_p0c",
            "next_stage": None if assessment["decision"] == "defer" else "P0C",
            "resources": {"control_fits": [fit.resources for fit in fits], "accelerator_hours_upper_bound": gpu_hours},
            "predictions_sha256": hashlib.sha256((output / "predictions.json").read_bytes()).hexdigest(),
            "control_artifacts_sha256": hashlib.sha256((output / "control-artifacts.json").read_bytes()).hexdigest(),
        })
        deadline()
    except Exception as error:
        report.update({"status": "failed", "decision": "blocked_execution_error", "error": f"{type(error).__name__}: {error}", "next_stage": None})
        raise
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report["report_payload_sha256"] = payload_sha256(report)
        write_json_once(output / "r0-report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--spec-sha", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    execute = subparsers.add_parser("run")
    execute.add_argument("--protocol", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        if _git("status", "--porcelain"):
            raise ValueError("commit changes before freezing the executable source")
        protocol = resolve_protocol(yaml.safe_load(args.config.read_text()), executable_sha=_git("rev-parse", "HEAD"), spec_sha=args.spec_sha)
        print(write_contract(args.output, protocol))
    else:
        report = run(read_contract(args.protocol, V2R2AuditProtocol), args.output)
        print(json.dumps({key: report[key] for key in ("status", "r0_decision", "decision", "report_payload_sha256")}, indent=2))


if __name__ == "__main__":
    main()
