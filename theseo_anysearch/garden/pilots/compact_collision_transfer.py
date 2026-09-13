"""Fresh local collision supervision on frozen code/raw/spatial representations."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from ..collision_transfer import CollisionHead, decision_threshold, probability_metrics
from . import compact_validation as common

PLAN = common.PLANS["transfer"]
CHANNELS = {"compact": 1, "null": 1, "raw": 2, "spatial": 8}


def make_head(seed, kind):
    torch.manual_seed(seed + 10000)
    return CollisionHead(CHANNELS[kind])


@torch.no_grad()
def inputs(c, package_root, rows, seed, memo):
    encoder, head, metadata = common.package.load_compact_package(package_root, seed=seed, device="cuda")
    del head
    if metadata["manifest_payload_sha256"] != common.PACKAGE_HASH: raise ValueError("package mismatch")
    before = common.package.encoder_state_sha256(encoder)
    cutoff = c.phase(f"inputs-{seed}", 1800)
    for split, row in rows.items():
        names = {"compact": f"inputs-{split}-compact-{seed}.pt", "raw": f"inputs-{split}-raw.pt", "spatial": f"inputs-{split}-spatial.pt"}
        missing = [kind for kind, name in names.items() if name not in c.progress["artifacts"]]
        collected = {kind: [] for kind in missing}
        for first in range(0, len(row["ids"]), 16) if missing else ():
            common.replication.d.check_deadline(cutoff)
            occ = row["occupancy"][first:first+16].cuda(); mask = row["hidden"][first:first+16].cuda()[:, None]
            level = common.base.VoxelLevel.from_occupancy(occ.float(), unknown_mask=mask)
            if "raw" in missing: collected["raw"].append(level.features[:, [0, 2]].cpu())
            if "spatial" in missing: collected["spatial"].append(encoder.backbone(level, mask).local_feature_volume.cpu())
            if "compact" in missing: collected["compact"].append(encoder(occ, mask).reshape(-1, 1, 9, 9, 9).cpu())
        for kind, name in names.items():
            if kind in missing:
                memo[name] = torch.cat(collected[kind]); c.artifact(name, memo[name])
            elif name not in memo:
                memo[name] = c.load(name)
    after = common.package.encoder_state_sha256(encoder)
    if before != after: raise ValueError("extraction changed frozen encoder")
    c.progress["encoder_states"][str(seed)] = {"before": before, "after": after}
    c.progress["inflight"] = None; c.save()
    del encoder


def values(memo, split, kind, seed):
    suffix = f"compact-{seed}" if kind in ("compact", "null") else kind
    result = memo[f"inputs-{split}-{suffix}.pt"]
    return torch.zeros_like(result) if kind == "null" else result


def fit(c, model, name, grids, row, seed):
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    rng = torch.Generator().manual_seed(seed + 40000); first = 0; elapsed = 0.; curve = []
    if name in c.progress["stages"]:
        state = c.progress["stages"][name]; saved = c.load(state["checkpoint"])
        if saved["state"] != state: raise ValueError("checkpoint state mismatch")
        model.load_state_dict(saved["model"])
        if common.package.encoder_state_sha256(model) != state["model_state_sha256"]: raise ValueError("head hash mismatch")
        if state["steps"] == PLAN["steps"]: return model
        optimizer.load_state_dict(saved["optimizer"]); rng.set_state(saved["rng"])
        first, elapsed, curve = state["steps"], state["fit_seconds"], state["curve"]
    cutoff = c.phase(name, PLAN["fit_cap_seconds"]); start = time.monotonic()
    cutoff = min(cutoff, start + PLAN["fit_cap_seconds"] - elapsed)
    gpu = grids.cuda(); model.train(); torch.cuda.reset_peak_memory_stats()
    for step in range(first + 1, PLAN["steps"] + 1):
        common.replication.d.check_deadline(cutoff)
        rate = common.prior.nonlinear.learning_rate({"steps": PLAN["steps"], "lr": .001}, step)
        for group in optimizer.param_groups: group["lr"] = rate
        ids = torch.randint(len(grids), (PLAN["batch_geometry"],), generator=rng)
        path_ids = torch.randint(32, (len(ids), PLAN["paths_per_geometry"]), generator=rng)
        selected = ids[:, None].expand_as(path_ids)
        paths = row["paths"][selected, path_ids].reshape(-1, 9, 3).cuda()
        valid = row["valid"][selected, path_ids].reshape(-1, 9).cuda()
        labels = row["labels"][selected, path_ids].flatten().float().cuda()
        geometry_indices = torch.arange(len(ids), device="cuda").repeat_interleave(PLAN["paths_per_geometry"])
        logits = model(gpu[ids.cuda()], paths, valid, geometry_indices)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite collision loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); optimizer.step()
        if step % 512 == 0:
            curve.append({"step": step, "loss": float(loss.detach()), "lr": rate})
            checkpoint = f"{name}-step-{step}.pt"
            state = {"steps": step, "checkpoint": checkpoint, "curve": curve[:], "fit_seconds": elapsed + time.monotonic() - start,
                     "model_state_sha256": common.package.encoder_state_sha256(model), "sampled_paths": step * 128,
                     "parameters": sum(p.numel() for p in model.parameters()), "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            c.artifact(checkpoint, {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng.get_state(), "state": state})
            c.progress["stages"][name] = state; c.save()
            print(json.dumps({"stage": name, **curve[-1]}), flush=True)
    del gpu
    c.progress["inflight"] = None; c.save()
    return model


@torch.no_grad()
def predict(model, grids, row, cutoff):
    model.eval(); parts = []
    for first in range(0, len(grids), 16):
        common.replication.d.check_deadline(cutoff)
        n = min(16, len(grids) - first)
        logits = model(grids[first:first+n].cuda(), row["paths"][first:first+n].reshape(-1, 9, 3).cuda(),
                       row["valid"][first:first+n].reshape(-1, 9).cuda(), torch.arange(n, device="cuda").repeat_interleave(32))
        parts.append(logits.sigmoid().reshape(n, 32).cpu())
    return torch.cat(parts)


def metrics(prediction, row, threshold):
    p = prediction.numpy(); y = row["labels"].numpy(); visible = row["visible_hit"].numpy(); unknown = row["unknown_path"].numpy()
    masks = {"all": np.ones_like(y), "fully_observed": ~unknown, "uncertain_no_visible_hit": unknown & ~visible,
             "visible_hit": visible, "hidden_only_collision": y & ~visible}
    for family in sorted(set(row["families"])):
        masks[f"family/{family}"] = np.broadcast_to(np.array([f == family for f in row["families"]])[:, None], y.shape)
    lengths = row["valid"].sum(-1).numpy()
    for length in (2, 3, 5, 9): masks[f"length/{length}"] = lengths == length
    return {key: probability_metrics(p[mask], y[mask], threshold) for key, mask in masks.items()}


def per_geometry_score(p, row):
    p = np.clip(p.numpy(), 1e-6, 1 - 1e-6); y = row["labels"].numpy()
    return (y * np.log(p) + (~y) * np.log1p(-p)).mean(1)


def engineering_screen(records):
    failures = []
    for seed in PLAN["seeds"]:
        fitted = {r["kind"]: r["metrics"]["all"] for r in records if r["seed"] == seed and r["split"] == "test"}
        candidate = fitted["compact"]
        for control in ("raw", "spatial"):
            if candidate["auprc"] < fitted[control]["auprc"] - .03: failures.append({"seed": seed, "comparison": f"auprc_vs_{control}"})
        if candidate["auprc"] < fitted["null"]["auprc"] + .10: failures.append({"seed": seed, "comparison": "auprc_vs_null"})
        if candidate["false_safe"] > .10: failures.append({"seed": seed, "comparison": "false_safe"})
        if candidate["false_alarm"] > .25: failures.append({"seed": seed, "comparison": "false_alarm"})
    return {"competitive_id_collision_head": not failures, "failures": failures, "promotion_eligible": False}


def run(path, data_root, package_root, output, ledger_path):
    start = time.monotonic(); env, rows = common.registration(path, "transfer", data_root)
    with common.search.exclusive_lock(ledger_path.with_suffix(".lock")):
        ledger = common.search.reserve_budget(ledger_path, PLAN["run_id"], PLAN["cap_seconds"])
        if common.completed(output, env): print("Completed; no repeat."); return
        c = common.campaign(output, env, start); memo = {}
        for seed in PLAN["seeds"]:
            inputs(c, package_root, {s: rows[s] for s in ("train", "calibration")}, seed, memo)
            for kind in PLAN["representations"]:
                if any(r["seed"] == seed and r["kind"] == kind for r in c.progress["records"]): continue
                name = f"head-{seed}-{kind}"
                model = fit(c, make_head(seed, kind).cuda(), name, values(memo, "train", kind, seed), rows["train"], seed)
                cutoff = c.phase(f"calibrate-{seed}-{kind}", 1800)
                prediction = predict(model, values(memo, "calibration", kind, seed), rows["calibration"], cutoff)
                threshold = decision_threshold(prediction.numpy(), rows["calibration"]["labels"].numpy())
                record = {"seed": seed, "kind": kind, "threshold": threshold, "metrics": metrics(prediction, rows["calibration"], threshold),
                          "parameters": sum(p.numel() for p in model.parameters())}
                c.artifact(f"calibration-{seed}-{kind}.pt", {"record": record, "prediction": prediction})
                c.progress["records"].append(record); c.progress["inflight"] = None; c.save()
                print(json.dumps({"calibrated": name, "all": record["metrics"]["all"]}), flush=True)
                del model
        lock = {"identity_sha256": env["identity_sha256"], "thresholds": {f"{r['seed']}-{r['kind']}": r["threshold"] for r in c.progress["records"]}}
        if len(lock["thresholds"]) != 12: raise ValueError("all heads must calibrate before test")
        lock_path = output / "selection-lock.json"
        if lock_path.exists() and json.loads(lock_path.read_text()) != lock: raise ValueError("threshold lock changed")
        common.base.write_json(lock_path, lock)
        results = []; comparisons = []; predictions = {}
        for seed in PLAN["seeds"]:
            inputs(c, package_root, {s: rows[s] for s in ("test", "ood")}, seed, memo)
            for kind in PLAN["representations"]:
                model = make_head(seed, kind).cuda(); state = c.progress["stages"][f"head-{seed}-{kind}"]
                saved = c.load(state["checkpoint"])
                if saved["state"] != state or state["steps"] != PLAN["steps"]: raise ValueError("unfinished stage")
                model.load_state_dict(saved["model"])
                if common.package.encoder_state_sha256(model) != state["model_state_sha256"]: raise ValueError("restored head mismatch")
                threshold = lock["thresholds"][f"{seed}-{kind}"]
                for split in ("test", "ood"):
                    cutoff = c.phase(f"test-{seed}-{kind}-{split}", 1800)
                    grids = values(memo, split, kind, seed)
                    variants = {kind: grids}
                    if kind == "compact":
                        permutation = torch.randperm(len(grids), generator=torch.Generator().manual_seed(seed + 30000))
                        variants.update(shuffled=grids[permutation], zeroed=torch.zeros_like(grids))
                    for variant, value in variants.items():
                        prediction = predict(model, value, rows[split], cutoff)
                        record = {"seed": seed, "kind": variant, "split": split, "threshold": threshold,
                                  "metrics": metrics(prediction, rows[split], threshold)}
                        c.artifact(f"test-{seed}-{variant}-{split}.pt", {"record": record, "prediction": prediction})
                        results.append(record); predictions[(seed, variant, split)] = prediction
                        print(json.dumps({"test": f"{seed}-{variant}-{split}", "all": record["metrics"]["all"]}), flush=True)
                del model
            for split in ("test", "ood"):
                for kind in ("raw", "spatial", "null", "shuffled"):
                    row = rows[split]
                    comparisons.append({"seed": seed, "split": split, "reference": kind, "negative_log_loss_difference":
                        common.bootstrap(per_geometry_score(predictions[(seed, "compact", split)], row),
                                         per_geometry_score(predictions[(seed, kind, split)], row), row["ids"], row["families"], row["densities"])})
        rules = {}
        for split in ("test", "ood"):
            r = rows[split]; definite = r["visible_hit"] | ~r["unknown_path"]
            rules[split] = {"coverage": float(definite.float().mean()), "abstain_fraction": float((~definite).float().mean()),
                            "definite_errors": int(((r["visible_hit"] != r["labels"]) & definite).sum()), "definite_count": int(definite.sum())}
        common.finish(c, env, ledger, ledger_path, start, {"calibration": c.progress["records"], "records": results,
                       "comparisons": comparisons, "visible_grid_rule": rules, "engineering_screen": engineering_screen(results)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True); parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    args = parser.parse_args(); run(args.registration, args.data_root, args.package, args.output, args.budget_ledger)
