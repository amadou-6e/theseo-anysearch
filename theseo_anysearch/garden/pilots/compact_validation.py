"""Preregistration and bounded no-retraining compact-code stress evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .. import compact_package as package
from ..evaluation.statistics import paired_stratified_geometry_bootstrap
from . import compact_replication as replication
from . import compact_validation_data as data

prior, base, search = replication.prior, replication.d.base, replication.search
PACKAGE_HASH = "2ddf188ecedff4699dba434fec05a851c5649119c07dcf30c9d9afd1292d4ed1"
PLANS = {
    "robustness": {"run_id": "compact-robustness-v1-run1", "cap_seconds": 7200, "overhead_reserve_seconds": 600,
                   "count_per_group": 24, "conditions": [list(c) for c in data.CONDITIONS], "seeds": [401, 402, 403]},
    "transfer": {"run_id": "compact-collision-transfer-v1-run1", "cap_seconds": 14400, "overhead_reserve_seconds": 900,
                 "fit_cap_seconds": 1800, "steps": 4096, "batch_geometry": 32, "paths_per_geometry": 4,
                 "counts": data.COUNTS, "seeds": [401, 402, 403], "representations": ["compact", "raw", "spatial", "null"]}}


def bootstrap(a, b, ids, families, densities):
    result = asdict(paired_stratified_geometry_bootstrap(a, b, geometry_ids=ids, geometry_families=families,
                                                        occupancy_bands=densities, resamples=1000, seed=405))
    result.pop("sample_indices")
    return result


def freeze(root, data_root, spec, reports):
    start = time.monotonic()
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec): raise ValueError("full spec SHA required")
    if base.git("status", "--porcelain", "--", "theseo_anysearch/garden"): raise ValueError("commit source first")
    if data_root.exists(): raise ValueError("fresh dataset directory required")
    parents = set(); predecessor_hashes = []
    for path in reports:
        report = json.loads(path.read_text()); digest = report.pop("report_payload_sha256")
        if base.payload_sha256(report) != digest: raise ValueError("predecessor hash mismatch")
        predecessor_hashes.append(digest)
        parents.update(h for split in report["registration"]["payload"]["data"].values() for h in split["parents"])
    if len(reports) != 6: raise ValueError("six compact predecessor reports required")
    data_root.mkdir(parents=True); root.mkdir(parents=True, exist_ok=True)
    for kind, factory in (("robustness", data.robustness), ("transfer", data.collision_data)):
        rows = factory()
        new_parents = {h for r in rows.values() for h in r["parents"]}
        if parents & new_parents: raise ValueError("prior/split parent overlap")
        parents.update(new_parents)
        file = data_root / f"{kind}.pt"; torch.save(rows, file)
        payload = {"plan": PLANS[kind], "source_commit": base.git("rev-parse", "HEAD"), "spec_commit": spec,
                   "spec_path": "projects/theseo-anysearch/python/perception-encoder-compact-robustness-transfer.md",
                   "package_manifest": PACKAGE_HASH, "predecessor_reports": predecessor_hashes,
                   "dataset_file": file.name, "dataset_sha256": package.file_sha(file), "data": data.identity(rows),
                   "preparation_seconds": time.monotonic() - start}
        base.write_json(root / f"compact-{kind}-preregistration.json", {"payload": payload, "identity_sha256": base.payload_sha256(payload)})
        print(json.dumps({"frozen": kind, "groups": len(rows), "geometries": sum(len(r["ids"]) for r in rows.values())}), flush=True)
        start = time.monotonic(); del rows


def registration(path, kind, data_root):
    env = json.loads(path.read_text()); payload = env["payload"]
    if base.payload_sha256(payload) != env["identity_sha256"] or payload["plan"] != PLANS[kind] or payload["package_manifest"] != PACKAGE_HASH:
        raise ValueError("validation registration mismatch")
    if base.git("diff", payload["source_commit"], "--", "theseo_anysearch/garden") or base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("frozen source mismatch")
    name = payload["dataset_file"]
    if Path(name).name != name or package.file_sha(data_root / name) != payload["dataset_sha256"]: raise ValueError("dataset hash mismatch")
    rows = torch.load(data_root / name, map_location="cpu", weights_only=True)
    if data.identity(rows) != payload["data"]: raise ValueError("dataset identity mismatch")
    return env, rows


def campaign(output, env, start):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "progress.json"; p = env["payload"]; plan = p["plan"]
    progress = json.loads(path.read_text()) if path.exists() else {
        "identity_sha256": env["identity_sha256"], "elapsed_seconds": p["preparation_seconds"] + plan["overhead_reserve_seconds"],
        "inflight": None, "stages": {}, "records": [], "artifacts": {}, "encoder_states": {}}
    if progress["identity_sha256"] != env["identity_sha256"]: raise ValueError("progress mismatch")
    c = prior.Campaign(output, env, progress, start, prior.resume_charge(progress), plan=plan)
    for name, digest in progress["artifacts"].items():
        if Path(name).name != name or package.file_sha(output / name) != digest: raise ValueError("artifact mismatch")
    replication.configure_cuda()
    return c


def completed(output, env):
    path = output / "report.json"
    if not path.exists(): return False
    report = json.loads(path.read_text()); digest = report.pop("report_payload_sha256")
    if base.payload_sha256(report) != digest or report["registration"] != env: raise ValueError("completed identity mismatch")
    return True


def finish(c, env, ledger, ledger_path, start, evidence):
    evidence.update(registration=env, status="completed", artifacts=c.progress["artifacts"], stages=c.progress["stages"],
                    encoder_states=c.progress["encoder_states"], elapsed_seconds=c.elapsed_before + time.monotonic() - start,
                    promotion_eligible=False)
    evidence["report_payload_sha256"] = base.payload_sha256(evidence)
    base.write_json(c.output / "report.json", evidence)
    ledger["runs"][env["payload"]["plan"]["run_id"]].update(charged_seconds=c.elapsed_before + time.monotonic() - start, settled=True)
    search.atomic_json(ledger_path, ledger)
    c.progress["inflight"] = None; c.save()
    print(json.dumps({"completed": True, "report": evidence["report_payload_sha256"]}), flush=True)


def geometry_metrics(prediction, row, thresholds):
    hidden = data.crop(row["hidden"], 17).flatten(1); truth = row["targets"]; free = truth[:, 0] < .5
    result = {}
    tasks = [("all_occupied_iou", 0, torch.ones_like(hidden), thresholds["occupied_iou"]),
             ("all_boundary_f1", 1, torch.ones_like(hidden), thresholds["boundary_f1"]),
             ("all_free_distance", 2, free, None), ("occupied_iou", 0, hidden, thresholds["occupied_iou"]),
             ("boundary_f1", 1, hidden, thresholds["boundary_f1"]), ("clearance_nmae", 2, free & ~hidden, None),
             ("recovery_nmae", 2, free & hidden, None)]
    for name, channel, valid, threshold in tasks:
        counts = valid.sum(1)
        if channel < 2:
            yes = prediction[:, channel] >= threshold; positive = truth[:, channel] > .5
            tp = (yes & positive & valid).sum(1); fp = (yes & ~positive & valid).sum(1); fn = (~yes & positive & valid).sum(1)
            numerator = tp * (2 if channel == 1 else 1); denominator = numerator + fp + fn
            score = float(numerator.sum() / denominator.sum()) if denominator.sum() else None
            per_geometry = [float(n / de) if de else None for n, de in zip(numerator, denominator)]
            support = {"positive": int((positive & valid).sum()), "negative": int((~positive & valid).sum())}
        else:
            error = ((prediction[:, channel] - truth[:, channel]).abs() * valid).sum(1)
            score = float(error.sum() / counts.sum()) if counts.sum() else None
            per_geometry = [float(e / n) if n else None for e, n in zip(error, counts)]; support = {}
        result[name] = {"score": score, "count": int(counts.sum()), "per_geometry": per_geometry, **support}
    return result


@torch.no_grad()
def robust_prediction(encoder, head, row, deadline):
    parts = []
    for first in range(0, len(row["ids"]), 16):
        replication.d.check_deadline(deadline)
        code = encoder(row["occupancy"][first:first+16].cuda(), row["hidden"][first:first+16].cuda())
        parts.append(prior.predict(head, "vector", code.cpu(), deadline))
    return torch.cat(parts)


def robustness(path, data_root, package_root, output, ledger_path):
    start = time.monotonic(); env, rows = registration(path, "robustness", data_root); plan = PLANS["robustness"]
    with search.exclusive_lock(ledger_path.with_suffix(".lock")):
        ledger = search.reserve_budget(ledger_path, plan["run_id"], plan["cap_seconds"])
        if completed(output, env): print("Completed; no repeat."); return
        c = campaign(output, env, start)
        for seed in plan["seeds"]:
            encoder, head, metadata = package.load_compact_package(package_root, seed=seed, device="cuda")
            if metadata["manifest_payload_sha256"] != PACKAGE_HASH: raise ValueError("package mismatch")
            before = package.encoder_state_sha256(encoder); head_before = package.encoder_state_sha256(head)
            for index, (key, row) in enumerate(rows.items()):
                if any(r["seed"] == seed and r["key"] == key for r in c.progress["records"]): continue
                cutoff = c.phase(f"robust-{seed}-{index}", 600)
                prediction = robust_prediction(encoder, head, row, cutoff)
                metrics = geometry_metrics(prediction, row, metadata["artifacts"][str(seed)]["thresholds"])
                record = {"seed": seed, "key": key, "metrics": metrics, "actual_density_mean": float(np.mean(row["actual_density"]))}
                c.artifact(f"robust-{seed}-{index}.pt", {"record": record, "prediction": prediction})
                c.progress["records"].append(record); c.progress["inflight"] = None; c.save()
                print(json.dumps({"seed": seed, "group": key, "scores": {k: v["score"] for k, v in metrics.items()}}), flush=True)
            if before != package.encoder_state_sha256(encoder) or head_before != package.encoder_state_sha256(head): raise ValueError("robustness changed weights")
            c.progress["encoder_states"][str(seed)] = {"before": before, "after": before, "head": head_before}
            del encoder, head
        comparisons, failed, unavailable = [], [], []
        for record in c.progress["records"]:
            key = record["key"]; row = rows[key]
            for task, target in data.prior.old.BARS.items():
                value = record["metrics"][task]["score"]
                if value is None: unavailable.append({"seed": record["seed"], "key": key, "task": task})
                elif (value < target if task in ("occupied_iou", "boundary_f1") else value > target):
                    failed.append({"seed": record["seed"], "key": key, "task": task, "score": value, "target": target})
            if row["condition"] != "baseline":
                reference = next(r for r in c.progress["records"] if r["seed"] == record["seed"] and r["key"] == f"baseline/{row['family']}")
                comparisons.append({"seed": record["seed"], "key": key, "negative_distance_error_difference":
                    bootstrap(-np.array(record["metrics"]["all_free_distance"]["per_geometry"]),
                              -np.array(reference["metrics"]["all_free_distance"]["per_geometry"]), row["ids"],
                              [row["family"]] * len(row["ids"]), [str(row["density"])] * len(row["ids"]))})
        finish(c, env, ledger, ledger_path, start, {"records": c.progress["records"], "comparisons": comparisons,
                                                   "target_failures": failed, "unavailable": unavailable})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "robustness"])
    parser.add_argument("--registration", type=Path); parser.add_argument("--registration-root", type=Path)
    parser.add_argument("--data-root", type=Path, required=True); parser.add_argument("--spec-sha")
    parser.add_argument("--prior-reports", type=Path, nargs=6); parser.add_argument("--package", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--budget-ledger", type=Path)
    args = parser.parse_args()
    if args.command == "freeze": freeze(args.registration_root, args.data_root, args.spec_sha, args.prior_reports)
    else: robustness(args.registration, args.data_root, args.package, args.output, args.budget_ledger)
