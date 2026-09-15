"""Frozen-code readout comparison with fresh uncertain-path calibration."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from ..collision_readout import NativeCollisionHead, UpsamplingCollisionHead, thresholds, uncertain
from ..collision_transfer import CollisionHead, probability_metrics
from . import compact_validation as common
from . import compact_collision_transfer as prior

PLAN = {"run_id": "compact-collision-readout-v1-run1", "program": "compact-collision-readout-v1",
        "cap_seconds": 14400, "overhead_reserve_seconds": 1200, "fit_cap_seconds": 1800,
        "steps": 4096, "batch_geometry": 32, "paths_per_geometry": 4,
        "counts": {"train": 1536, "calibration": 768, "test": 768, "ood": 576},
        "seeds": [401, 402, 403], "representations": ["original", "resize_conv", "transpose", "native", "raw", "spatial", "null"]}
CANDIDATES = ("resize_conv", "transpose", "native")
CODE_HEADS = ("original", *CANDIDATES)
PREDECESSORS = {"3c058b813540b08c7eb8f7499a13c2c6b08b44549689a2fc1d9f2a76ec7f8d67",
                "6f3036bd4f6a83443324ca44a7ac9c234b82748a31b89c71be4052349de09b7e"}


def make_head(seed, kind):
    torch.manual_seed(seed + 10000)
    if kind in ("resize_conv", "transpose", "null"):
        return UpsamplingCollisionHead("resize_conv" if kind == "null" else kind)
    if kind == "native": return NativeCollisionHead()
    return CollisionHead({"original": 1, "raw": 2, "spatial": 8}[kind])


def freeze(path, data_root, spec, reports):
    start = time.monotonic()
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec): raise ValueError("full spec SHA required")
    if common.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"): raise ValueError("commit source first")
    if data_root.exists(): raise ValueError("fresh dataset directory required")
    parents = set(); hashes = []
    for file in reports:
        report = json.loads(file.read_text()); digest = report.pop("report_payload_sha256")
        if common.base.payload_sha256(report) != digest: raise ValueError("predecessor hash mismatch")
        hashes.append(digest)
        parents.update(h for row in report["registration"]["payload"]["data"].values() for h in row["parents"])
    if len(hashes) != 8 or len(set(hashes)) != 8 or not PREDECESSORS.issubset(hashes):
        raise ValueError("eight distinct predecessor reports required")
    rows = common.data.collision_data(PLAN["counts"], program=PLAN["program"])
    if parents.intersection(h for row in rows.values() for h in row["parents"]): raise ValueError("prior parent overlap")
    thresholds(torch.full_like(rows["calibration"]["labels"], .5, dtype=torch.float32), rows["calibration"])
    data_root.mkdir(parents=True)
    file = data_root / "dataset.pt"; torch.save(rows, file)
    payload = {"plan": PLAN, "source_commit": common.base.git("rev-parse", "HEAD"), "spec_commit": spec,
               "spec_path": "projects/theseo-anysearch/python/perception-encoder-collision-readout.md",
               "package_manifest": common.PACKAGE_HASH, "predecessor_reports": hashes,
               "dataset_file": file.name, "dataset_sha256": common.package.file_sha(file),
               "data": common.data.identity(rows), "preparation_seconds": time.monotonic()-start}
    common.base.write_json(path, {"payload": payload, "identity_sha256": common.base.payload_sha256(payload)})
    print(json.dumps({"frozen": PLAN["run_id"], "geometries": sum(len(r["ids"]) for r in rows.values())}), flush=True)


def registration(path, data_root):
    env = json.loads(path.read_text()); p = env["payload"]
    if common.base.payload_sha256(p) != env["identity_sha256"] or p["plan"] != PLAN or p["package_manifest"] != common.PACKAGE_HASH:
        raise ValueError("readout registration mismatch")
    if common.base.git("diff", p["source_commit"], "--", "theseo_anysearch/garden") or common.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("frozen source mismatch")
    name = p["dataset_file"]
    if Path(name).name != name or common.package.file_sha(data_root / name) != p["dataset_sha256"]: raise ValueError("dataset hash mismatch")
    rows = torch.load(data_root / name, map_location="cpu", weights_only=True)
    if common.data.identity(rows) != p["data"]: raise ValueError("dataset identity mismatch")
    return env, rows


@torch.no_grad()
def inputs(c, package_root, rows, seed, memo):
    prior.inputs(c, package_root, rows, seed, memo)
    encoder, decoder, _ = common.package.load_compact_package(package_root, seed=seed, device="cuda")
    del encoder
    decoder.eval().requires_grad_(False); before = common.package.encoder_state_sha256(decoder)
    cutoff = c.phase(f"native-inputs-{seed}", 900)
    names = {f"inputs-{split}-native-{seed}.pt": memo[f"inputs-{split}-compact-{seed}.pt"].flatten(1) for split in rows}
    names[f"native-zero-{seed}.pt"] = torch.zeros(1, 729)
    for name, code in names.items():
        if name not in c.progress["artifacts"]:
            value = common.prior.predict(decoder, "vector", code, cutoff).reshape(-1, 3, 17, 17, 17)
            c.artifact(name, value); memo[name] = value
        elif name not in memo: memo[name] = c.load(name)
    if common.package.encoder_state_sha256(decoder) != before: raise ValueError("native decoder changed")
    c.progress.setdefault("native_states", {})[str(seed)] = {"before": before, "after": before,
                                                            "frozen_parameters": sum(p.numel() for p in decoder.parameters())}
    c.progress["inflight"] = None; c.save()


def values(memo, split, kind, seed):
    if kind == "native": return memo[f"inputs-{split}-native-{seed}.pt"]
    return prior.values(memo, split, "compact" if kind in CODE_HEADS else kind, seed)


def variants(memo, split, kind, seed):
    grids = values(memo, split, kind, seed); result = {kind: grids}
    if kind in CODE_HEADS:
        permutation = torch.randperm(len(grids), generator=torch.Generator().manual_seed(seed + 30000))
        zero = memo[f"native-zero-{seed}.pt"].expand(len(grids), -1, -1, -1, -1) if kind == "native" else torch.zeros_like(grids)
        result.update({f"{kind}-shuffled": grids[permutation], f"{kind}-zeroed": zero})
    return result


def metrics(prediction, row, cuts):
    mask = uncertain(row); definite = ~mask
    decisions = torch.where(definite, row["visible_hit"], prediction >= cuts["uncertain"])
    y = row["labels"]
    return {"pooled": prior.metrics(prediction, row, cuts["pooled"]),
            "uncertain_operating_point": probability_metrics(prediction[mask].numpy(), y[mask].numpy(), cuts["uncertain"]),
            "observability_assisted": {"definite_coverage": float(definite.float().mean()),
                "definite_errors": int(((row["visible_hit"] != y) & definite).sum()),
                "false_safe": float((~decisions & y).sum() / y.sum()),
                "false_alarm": float((decisions & ~y).sum() / (~y).sum())}}


def uncertain_scores(prediction, row):
    p = np.clip(prediction.numpy().astype(np.float64), 1e-6, 1 - 1e-6)
    y = row["labels"].numpy(); mask = uncertain(row).numpy(); counts = mask.sum(1)
    score = ((y*np.log(p) + (~y)*np.log1p(-p))*mask).sum(1) / np.maximum(counts, 1)
    return score, counts > 0


def comparisons(predictions, rows):
    result = []
    for seed in PLAN["seeds"]:
        for split in ("test", "ood"):
            row = rows[split]
            for kind in CANDIDATES:
                candidate, keep = uncertain_scores(predictions[seed, kind, split], row)
                for control in ("original", "raw", "spatial", "null"):
                    reference, valid = uncertain_scores(predictions[seed, control, split], row)
                    if not np.array_equal(keep, valid): raise ValueError("comparison support mismatch")
                    result.append({"seed": seed, "split": split, "kind": kind, "reference": control,
                        "included_geometries": int(keep.sum()), "excluded_geometries": int((~keep).sum()),
                        "negative_uncertain_log_loss_difference": common.bootstrap(candidate[keep], reference[keep],
                            [v for v, k in zip(row["ids"], keep) if k], [v for v, k in zip(row["families"], keep) if k],
                            [v for v, k in zip(row["densities"], keep) if k])})
    return result


def assessment(records, contrasts):
    result = {}
    for kind in CANDIDATES:
        failures = []; competitive = []
        for seed in PLAN["seeds"]:
            lookup = {r["kind"]: r for r in records if r["seed"] == seed and r["split"] == "test"}
            c = lookup[kind]["metrics"]; original = lookup["original"]["metrics"]
            ci = next(r["negative_uncertain_log_loss_difference"] for r in contrasts if r["seed"] == seed and r["split"] == "test" and r["kind"] == kind and r["reference"] == "original")
            checks = {"primary_ci": ci["lower_95"] > 0,
                      "uncertain_auprc_gain": c["uncertain_operating_point"]["auprc"] >= original["uncertain_operating_point"]["auprc"] + .03,
                      "uncertain_false_safe": c["uncertain_operating_point"]["false_safe"] <= .10,
                      "uncertain_false_alarm": c["uncertain_operating_point"]["false_alarm"] <= .25}
            failures.extend({"seed": seed, "check": name} for name, passed in checks.items() if not passed)
            for control in ("raw", "spatial"):
                for scope in ("overall", "uncertain"):
                    a = c["pooled"]["all"] if scope == "overall" else c["uncertain_operating_point"]
                    b = lookup[control]["metrics"]["pooled"]["all"] if scope == "overall" else lookup[control]["metrics"]["uncertain_operating_point"]
                    if a["auprc"] < b["auprc"] - .03: competitive.append({"seed": seed, "control": control, "scope": scope})
        result[kind] = {"id_improvement_candidate": not failures, "improvement_failures": failures,
                        "competitive": not failures and not competitive, "competitive_margin_failures": competitive}
    return {"candidates": result, "promotion_eligible": False}


def run(path, data_root, package_root, output, ledger_path):
    start = time.monotonic(); env, rows = registration(path, data_root)
    with common.search.exclusive_lock(ledger_path.with_suffix(".lock")):
        ledger = common.search.reserve_budget(ledger_path, PLAN["run_id"], PLAN["cap_seconds"])
        if common.completed(output, env): print("Completed; no repeat."); return
        c = common.campaign(output, env, start); memo = {}
        for seed in PLAN["seeds"]:
            inputs(c, package_root, {s: rows[s] for s in ("train", "calibration")}, seed, memo)
            for kind in PLAN["representations"]:
                if any(r["seed"] == seed and r["kind"] == kind for r in c.progress["records"]): continue
                name = f"head-{seed}-{kind}"
                model = prior.fit(c, make_head(seed, kind).cuda(), name, values(memo, "train", kind, seed), rows["train"], seed, plan=PLAN)
                cutoff = c.phase(f"calibrate-{seed}-{kind}", 900)
                prediction = prior.predict(model, values(memo, "calibration", kind, seed), rows["calibration"], cutoff)
                cuts = thresholds(prediction, rows["calibration"])
                record = {"seed": seed, "kind": kind, "thresholds": cuts, "metrics": metrics(prediction, rows["calibration"], cuts),
                          "parameters": sum(p.numel() for p in model.parameters())}
                c.artifact(f"calibration-{seed}-{kind}.pt", {"record": record, "prediction": prediction})
                c.progress["records"].append(record); c.progress["inflight"] = None; c.save(); del model
                print(json.dumps({"calibrated": name, "uncertain": record["metrics"]["uncertain_operating_point"]}), flush=True)
        lock = {"identity_sha256": env["identity_sha256"], "thresholds": {f"{r['seed']}-{r['kind']}": r["thresholds"] for r in c.progress["records"]}}
        if len(lock["thresholds"]) != 21: raise ValueError("all heads must calibrate before tests")
        lock_path = output / "selection-lock.json"
        if lock_path.exists() and json.loads(lock_path.read_text()) != lock: raise ValueError("threshold lock changed")
        common.base.write_json(lock_path, lock)
        records = []; predictions = {}
        for seed in PLAN["seeds"]:
            inputs(c, package_root, {s: rows[s] for s in ("test", "ood")}, seed, memo)
            for kind in PLAN["representations"]:
                state = c.progress["stages"][f"head-{seed}-{kind}"]; saved = c.load(state["checkpoint"])
                if saved["state"] != state or state["steps"] != PLAN["steps"]: raise ValueError("unfinished head")
                model = make_head(seed, kind).cuda(); model.load_state_dict(saved["model"])
                if common.package.encoder_state_sha256(model) != state["model_state_sha256"]: raise ValueError("restored head mismatch")
                cuts = lock["thresholds"][f"{seed}-{kind}"]
                for split in ("test", "ood"):
                    cutoff = c.phase(f"test-{seed}-{kind}-{split}", 900)
                    for variant, grids in variants(memo, split, kind, seed).items():
                        prediction = prior.predict(model, grids, rows[split], cutoff)
                        record = {"seed": seed, "kind": variant, "split": split, "thresholds": cuts, "metrics": metrics(prediction, rows[split], cuts)}
                        c.artifact(f"test-{seed}-{variant}-{split}.pt", {"record": record, "prediction": prediction})
                        records.append(record); predictions[seed, variant, split] = prediction
                        print(json.dumps({"test": f"{seed}-{variant}-{split}", "uncertain": record["metrics"]["uncertain_operating_point"]}), flush=True)
                del model
        contrasts = comparisons(predictions, rows)
        common.finish(c, env, ledger, ledger_path, start, {"calibration": c.progress["records"], "records": records,
            "comparisons": contrasts, "assessment": assessment(records, contrasts), "native_states": c.progress["native_states"]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    for name in ("registration", "data-root", "package", "output", "budget-ledger"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--spec-sha"); parser.add_argument("--prior-reports", type=Path, nargs=8)
    args = parser.parse_args()
    if args.command == "freeze": freeze(args.registration, args.data_root, args.spec_sha, args.prior_reports)
    else: run(args.registration, args.data_root, args.package, args.output, args.budget_ledger)
