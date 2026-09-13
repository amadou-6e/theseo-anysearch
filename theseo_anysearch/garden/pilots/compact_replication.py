"""Fresh-seed replication of the selection-locked fine729 native-code recipe."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from . import compact_structured as parent

prior, corpus, d, search = parent.prior, parent.corpus, parent.d, parent.search
SEEDS = (401, 402, 403)
CONFIG = {"kind": "structured", "side": 9, "channels": 1, "dimension": 729, "lr": .0003}
PLAN = {**parent.PLAN, "run_id": "compact-replication-v1-run1", "cap_seconds": 14400,
        "sampling_seed": 40100, "seeds": list(SEEDS),
        "source_report": "f4c46fe17c0c767e7e43e374ee940f70d83d760f472c18c1fd84a5ee02711b94"}
TARGETS = {"occupied_iou": .60, "boundary_f1": .70, "clearance_nmae": .10, "recovery_nmae": .10}


def data(splits=None):
    return corpus.data(splits, study_id="compact-replication-v1", counts=PLAN["counts"],
                       bank_splits=("train", "probe"), bank_sizes=PLAN["bank_sizes"])


def make_model(seed):
    torch.manual_seed(seed)
    return parent.models.StructuredModel(CONFIG)


def make_head(seed):
    torch.manual_seed(seed + 10000)
    return parent.models.StructuredReadout(9, 1)


def make_spatial(seed):
    torch.manual_seed(seed + 20000)
    return prior.models.SpatialReference()


def assess(records):
    if [r["seed"] for r in records] != list(SEEDS):
        raise ValueError("all frozen seeds required")
    failures = []
    for record in records:
        native = record["native"]
        for task, target in TARGETS.items():
            families = native[task]["families"]
            if set(families) != {"random_field", "oblique_sheets", "sphere_shells", "box_shells"}:
                raise ValueError("missing assessment family")
            for family, value in families.items():
                if not torch.isfinite(torch.tensor(value)):
                    raise ValueError("nonfinite assessment")
                if (value < target if task in ("occupied_iou", "boundary_f1") else value > target):
                    failures.append({"seed": record["seed"], "task": task, "family": family,
                                     "score": value, "target": target})
        for family, value in native["boundary_f1"]["families"].items():
            for control in ("shuffled", "null"):
                baseline = record[control]["boundary_f1"]["families"][family]
                if not torch.isfinite(torch.tensor(baseline)):
                    raise ValueError("nonfinite control")
                if value - baseline < .10:
                    failures.append({"seed": record["seed"], "task": "embedding_necessity",
                                     "family": family, "control": control, "gain": value - baseline, "target": .10})
    ranges = {task: {family: {"min": min(r["native"][task]["families"][family] for r in records),
                              "max": max(r["native"][task]["families"][family] for r in records)}
                     for family in records[0]["native"][task]["families"]} for task in TARGETS}
    return {"experimental_packaging_ready": not failures, "failures": failures,
            "native_seed_ranges": ranges, "promotion_eligible": False}


def check_registration(env):
    p = env["payload"]
    if d.base.payload_sha256(p) != env["identity_sha256"] or p["plan"] != PLAN or p["config"] != CONFIG:
        raise ValueError("replication registration mismatch")
    return p


def freeze(path, spec, report_paths):
    start = time.monotonic()
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec):
        raise ValueError("full spec SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit source first")
    reports = []
    for path_in in report_paths:
        report = json.loads(path_in.read_text()); digest = report.pop("report_payload_sha256")
        if digest != d.base.payload_sha256(report):
            raise ValueError("prior report mismatch")
        reports.append((report, digest))
    if len(reports) != 5 or reports[-1][1] != PLAN["source_report"] or reports[-1][0]["selected_id"] != 8:
        raise ValueError("require five predecessor reports and locked winner")
    rows = data(); corpus.support(rows)
    parents = {h for r, _ in reports for split in r["registration"]["payload"]["data"].values() for h in split["parents"]}
    if any(h in parents for row in rows.values() for h in row["parents"]):
        raise ValueError("reused parent")
    payload = {"plan": PLAN, "config": CONFIG, "source_commit": d.base.git("rev-parse", "HEAD"),
               "spec_commit": spec, "spec_path": "projects/theseo-anysearch/python/perception-encoder-compact-replication.md",
               "prior_reports": [h for _, h in reports], "data": corpus.old.identity(rows),
               "source_encoders": reports[-1][0]["registration"]["payload"]["source_encoders"],
               "preparation_seconds": time.monotonic() - start}
    d.base.write_json(path, {"payload": payload, "identity_sha256": d.base.payload_sha256(payload)})


def configure_cuda():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False


def lock_thresholds(env, records):
    if [r["seed"] for r in records] != list(SEEDS):
        raise ValueError("all calibration records required")
    return {"identity_sha256": env["identity_sha256"], "seeds": list(SEEDS),
            "thresholds": {str(r["seed"]): {k: r[k]["thresholds"] for k in ("native", "spatial", "null")} for r in records}}


def run(path, source, output, ledger_path):
    start = time.monotonic(); env = json.loads(path.read_text()); payload = check_registration(env)
    if d.base.git("diff", payload["source_commit"], "--", "theseo_anysearch/garden") or d.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("source mismatch")
    if payload["source_encoders"] != corpus.old.helpers.prior.sources(source):
        raise ValueError("initializer mismatch")
    output.mkdir(parents=True, exist_ok=True)
    with search.exclusive_lock(ledger_path.with_suffix(".lock")):
        ledger = search.reserve_budget(ledger_path, PLAN["run_id"], PLAN["cap_seconds"])
        if (output / "report.json").exists():
            done = json.loads((output / "report.json").read_text()); digest = done.pop("report_payload_sha256")
            if d.base.payload_sha256(done) != digest or done["registration"] != env:
                raise ValueError("completed report mismatch")
            print("Completed; no fits restarted.", flush=True); return
        progress_path = output / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "identity_sha256": env["identity_sha256"], "elapsed_seconds": payload["preparation_seconds"] + PLAN["overhead_reserve_seconds"],
            "inflight": None, "stages": {}, "records": [], "caches": {}, "artifacts": {}}
        if progress["identity_sha256"] != env["identity_sha256"]:
            raise ValueError("progress identity mismatch")
        campaign = prior.Campaign(output, env, progress, start, prior.resume_charge(progress), plan=dict(PLAN))
        cutoff = campaign.phase("preparation", PLAN["preparation_cap_seconds"])
        for name, digest in progress["artifacts"].items():
            if prior.previous.file_hash(output / name) != digest:
                raise ValueError("artifact mismatch")
        configure_cuda()
        rows = data(["train", "probe", "selection"]); corpus.support(rows)
        if corpus.old.identity(rows) != {s: payload["data"][s] for s in rows}:
            raise ValueError("data mismatch")
        backbone = d.base.make_encoder(0, torch.device("cuda"))
        backbone.load_state_dict(torch.load(source / payload["source_encoders"][0]["path"], map_location="cuda", weights_only=True))
        backbone.eval().requires_grad_(False)
        if d.base.encoder_state_sha256(backbone) != payload["source_encoders"][0]["state_hash"]:
            raise ValueError("backbone mismatch")
        cache = {s: campaign.cache(s, backbone, rows[s], cutoff) for s in rows}
        progress["inflight"] = None; campaign.save()
        for seed in SEEDS:
            if any(r["seed"] == seed for r in progress["records"]):
                continue
            campaign.plan["sampling_seed"] = seed * 100
            model = campaign.fit(f"train-{seed}", make_model(seed).cuda(), "volume", cache["train"], rows["train"], lr=CONFIG["lr"])
            cutoff = campaign.phase(f"extract-{seed}", PLAN["fit_cap_seconds"])
            aggregation = model.aggregation.eval().requires_grad_(False)
            aggregation_hash = d.base.encoder_state_sha256(aggregation)
            name = f"vectors-{seed}.pt"
            if name in progress["artifacts"]:
                stored = campaign.load(name)
                if stored["aggregation_hash"] != aggregation_hash:
                    raise ValueError("vector aggregation mismatch")
            else:
                stored = {"aggregation_hash": aggregation_hash,
                          "probe": prior.vectors(aggregation, cache["probe"], cutoff),
                          "selection": prior.vectors(aggregation, cache["selection"], cutoff)}
                campaign.artifact(name, stored)
            z, stats = parent.models.normalized_vectors(stored["probe"], CONFIG)
            selected = parent.heads.normalize(stored["selection"], stats)
            native = campaign.fit(f"probe-{seed}", make_head(seed).cuda(), "vector", z, rows["probe"])
            spatial = campaign.fit(f"spatial-{seed}", make_spatial(seed).cuda(), "spatial", cache["probe"], rows["probe"])
            null = campaign.fit(f"null-{seed}", make_head(seed).cuda(), "vector", torch.zeros_like(z), rows["probe"])
            cutoff = campaign.phase(f"assess-{seed}", PLAN["fit_cap_seconds"])
            predictions = {"native": prior.predict(native, "vector", selected, cutoff),
                           "spatial": prior.predict(spatial, "spatial", cache["selection"], cutoff),
                           "null": prior.predict(null, "vector", torch.zeros_like(selected), cutoff)}
            record = {"seed": seed, "aggregation_state_sha256": aggregation_hash,
                      **{k: prior.metrics(v, rows["selection"]) for k, v in predictions.items()}}
            permutation = torch.randperm(len(selected), generator=torch.Generator().manual_seed(seed + 30000))
            for name, values in (("shuffled", selected[permutation]), ("zeroed", torch.zeros_like(selected))):
                predictions[name] = prior.predict(native, "vector", values, cutoff)
                record[name] = prior.diagnosis.compact_metrics(predictions[name], rows["selection"], record["native"]["thresholds"])
            if d.base.encoder_state_sha256(aggregation) != aggregation_hash:
                raise ValueError("frozen aggregation mutated")
            campaign.artifact(f"assessment-{seed}.pt", {"record": record, "statistics": stats, "predictions": predictions})
            progress["records"].append(record); progress["inflight"] = None; campaign.save()
            print(json.dumps({"seed": seed, "selection": record["native"]["selection"]}), flush=True)
            del model, aggregation, native, spatial, null, stored, z, selected
        lock = lock_thresholds(env, progress["records"]); lock_path = output / "selection-lock.json"
        if lock_path.exists() and json.loads(lock_path.read_text()) != lock:
            raise ValueError("threshold lock changed")
        d.base.write_json(lock_path, lock)
        cutoff = campaign.phase("development", PLAN["preparation_cap_seconds"])
        development = data(["development"])
        if corpus.old.identity(development) != {"development": payload["data"]["development"]}:
            raise ValueError("development identity mismatch")
        dev = development["development"]; dev_cache = campaign.cache("development", backbone, dev, cutoff)
        results = []
        for record in progress["records"]:
            seed = record["seed"]
            model = parent.restore(campaign, f"train-{seed}", make_model(seed).cuda())
            stats = campaign.load(f"assessment-{seed}.pt")["statistics"]
            z = parent.heads.normalize(prior.vectors(model.aggregation, dev_cache, cutoff), stats)
            native = parent.restore(campaign, f"probe-{seed}", make_head(seed).cuda())
            spatial = parent.restore(campaign, f"spatial-{seed}", make_spatial(seed).cuda())
            null = parent.restore(campaign, f"null-{seed}", make_head(seed).cuda())
            permutation = torch.randperm(len(z), generator=torch.Generator().manual_seed(seed + 30000))
            predictions = {"native": prior.predict(native, "vector", z, cutoff),
                           "shuffled": prior.predict(native, "vector", z[permutation], cutoff),
                           "zeroed": prior.predict(native, "vector", torch.zeros_like(z), cutoff),
                           "spatial": prior.predict(spatial, "spatial", dev_cache, cutoff),
                           "null": prior.predict(null, "vector", torch.zeros_like(z), cutoff)}
            result = {"seed": seed, **{k: prior.diagnosis.compact_metrics(v, dev, record[k if k in ("spatial", "null") else "native"]["thresholds"])
                                      for k, v in predictions.items()}}
            campaign.artifact(f"development-{seed}.pt", {"record": result, "predictions": predictions})
            results.append(result)
        d.check_deadline(cutoff)
        evidence = {"registration": env, "status": "completed", "records": progress["records"], "stages": progress["stages"],
                    "development": results, "assessment": assess(results), "artifacts": progress["artifacts"], "caches": progress["caches"],
                    "elapsed_seconds": campaign.elapsed_before + time.monotonic() - start, "promotion_eligible": False,
                    "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()}}
        evidence["report_payload_sha256"] = d.base.payload_sha256(evidence); d.base.write_json(output / "report.json", evidence)
        ledger["runs"][PLAN["run_id"]].update(charged_seconds=campaign.elapsed_before + time.monotonic() - start, settled=True)
        search.atomic_json(ledger_path, ledger)
        progress["inflight"] = None; campaign.save()
        print(json.dumps({"completed": True, "assessment": evidence["assessment"], "development": results}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--registration", type=Path, required=True); parser.add_argument("--spec-sha")
    parser.add_argument("--prior-reports", type=Path, nargs=5); parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--budget-ledger", type=Path)
    args = parser.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "freeze":
        freeze(args.registration, args.spec_sha, args.prior_reports)
    else:
        run(args.registration, args.source, args.output, args.budget_ledger)
