"""Budgeted, checkpointed compact-encoder structural and hyperparameter search."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from . import compact_search_data as corpus

d = corpus.d
PLAN = {"study_id": "compact-search-v1", "run_id": "compact-search-v1-run1",
        "cap_seconds": 8*3600, "trial_cap_seconds": 900, "protected": [5, 16],
        "rungs": [512, 2048, 8192], "campaign_authorized_seconds": 48*3600,
        "historical_reserve_seconds": 3600, "counts": corpus.COUNTS, "bars": corpus.BARS}


def atomic_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


@contextmanager
def exclusive_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            if path.stat().st_size == 0: handle.write(b"0"); handle.flush()
            handle.seek(0); msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally:
            handle.seek(0)
            if os.name == "nt": msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(handle, fcntl.LOCK_UN)


def reserve_budget(path, run_id, seconds):
    ledger = json.loads(path.read_text()) if path.exists() else {
        "authorized_seconds": 48*3600, "historical_reserve_seconds": 3600,
        "authorization": "User explicitly approved up to48 GPU-hours on2026-09-12; no merges.", "runs": {}}
    if ledger["authorized_seconds"] != PLAN["campaign_authorized_seconds"]: raise ValueError("budget authorization mismatch")
    if run_id not in ledger["runs"]:
        used = ledger["historical_reserve_seconds"] + sum(r["charged_seconds"] if r["settled"] else r["reserved_seconds"] for r in ledger["runs"].values())
        if seconds <= 0 or used+seconds > ledger["authorized_seconds"]: raise ValueError("campaign budget exhausted")
        ledger["runs"][run_id] = {"reserved_seconds": seconds, "charged_seconds": 0, "settled": False}
        atomic_json(path, ledger)
    elif ledger["runs"][run_id]["reserved_seconds"] != seconds: raise ValueError("reservation mismatch")
    return ledger


def structural_configs():
    configs = []
    for mode in ("grid", "strided", "attention"):
        for dimension in (64, 128, 192):
            for joint in (False, True):
                configs.append({"id": len(configs), "mode": mode, "dimension": dimension, "joint": joint,
                                "lr": .003, "weight_decay": .01, "mask": .2, "boundary_weight": 1., "distance_weight": 10.})
    return configs


def templates():
    rng = np.random.default_rng(389)
    return [{"lr": float(np.exp(rng.uniform(np.log(1e-4), np.log(.003)))),
             "weight_decay": float(np.exp(rng.uniform(np.log(1e-5), np.log(.01)))),
             "mask": float(rng.choice([.1, .2, .35])), "boundary_weight": float(rng.choice([.5, 1, 2])),
             "distance_weight": float(rng.choice([3, 10, 30]))} for _ in range(18)]


def check_support(rows):
    for split, v in rows.items():
        eligible = corpus.masks(v)
        for family in range(4):
            ids = [i for i in range(len(v["ids"])) if (i%12)//3 == family]
            for channel, task in enumerate(d.base.TASKS[:2]):
                y = v["targets"][ids, channel][eligible[task][ids]]
                if min(float(y.sum()), float((1-y).sum())) < 20: raise ValueError(f"insufficient class support: {split}/{family}/{task}")
            for task in d.base.TASKS[2:]:
                if not eligible[task][ids].any(): raise ValueError("empty family distance query pool")


def freeze(path, spec, source):
    started = time.monotonic()
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec): raise ValueError("full spec SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"): raise ValueError("commit source first")
    rows = corpus.data(); check_support(rows)
    p = {"plan": PLAN, "source_commit": d.base.git("rev-parse", "HEAD"), "spec_commit": spec,
         "spec_path": "projects/theseo-anysearch/python/perception-encoder-compact-search.md",
         "structural_configs": structural_configs(), "templates": templates(),
         "source_encoders": corpus.helpers.prior.sources(source), "data": corpus.identity(rows),
         "preparation_seconds": time.monotonic()-started}
    d.base.write_json(path, {"payload": p, "identity_sha256": d.base.payload_sha256(p)})


def objective(logits, targets, hidden, class_weights, config):
    if not hidden.any(): raise ValueError("empty training mask")
    losses = [F.binary_cross_entropy_with_logits(logits[:, c][hidden], targets[:, c][hidden], pos_weight=class_weights[c]) for c in (0, 1)]
    free = targets[:, 0] < .5
    distance = F.smooth_l1_loss(logits[:, 2][free], targets[:, 2][free]) if free.any() else logits[:, 2].sum()*0
    return losses[0]+config["boundary_weight"]*losses[1]+config["distance_weight"]*distance


def stage(config, steps, previous_record, rows, payload, source, output, deadline):
    started = time.monotonic(); deadline = min(deadline, started+PLAN["trial_cap_seconds"])
    model = corpus.make_encoder(config, source, payload["source_encoders"][0]).train()
    initial_backbone = d.base.encoder_state_sha256(model.backbone)
    decoder = nn.Linear(config["dimension"], 3*4913).cuda()
    opt = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad]+list(decoder.parameters()), lr=config["lr"], weight_decay=config["weight_decay"])
    rng = torch.Generator(device="cuda").manual_seed(38900)
    first = 0; curve = []
    if previous_record is not None:
        old = torch.load(output/previous_record["checkpoint"], map_location="cuda", weights_only=True)
        if old["record"] != previous_record: raise ValueError("resume record mismatch")
        model.load_state_dict(old["model"]); decoder.load_state_dict(old["decoder"]); opt.load_state_dict(old["optimizer"])
        if d.base.encoder_state_sha256(model) != previous_record["encoder_state_sha256"]: raise ValueError("resume encoder hash mismatch")
        rng.set_state(old["rng"].cpu()); first = previous_record["steps"]; curve = previous_record["curve"][:]
    target_train = rows["train"]["targets"].cuda()
    prevalence = target_train[:, :2].mean(dim=(0, 2))
    class_weights = (1-prevalence)/prevalence.clamp_min(1e-5)
    torch.cuda.reset_peak_memory_stats()
    for step in range(first, steps):
        d.check_deadline(deadline)
        idx = torch.randint(len(target_train), (4,), generator=rng, device="cuda")
        occ = rows["train"]["occupancy"][idx.cpu()].to("cuda", dtype=torch.float32)
        hidden = torch.rand((4, 1, 33, 33, 33), generator=rng, device="cuda") < config["mask"]
        z = model(d.base.VoxelLevel.from_occupancy(occ, unknown_mask=hidden), hidden)
        logits = decoder(z).reshape(4, 3, 4913)
        loss = objective(logits, target_train[idx], corpus.helpers.prior.crop(hidden[:, 0], 17).flatten(1), class_weights, config)
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite training loss")
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step+1)%512 == 0:
            curve.append({"step": step+1, "loss": float(loss.detach())})
            print(json.dumps({"trial": config["id"], "step": step+1, "loss": float(loss.detach())}), flush=True)
    torch.cuda.synchronize(); train_seconds = time.monotonic()-started
    train_peak = torch.cuda.max_memory_allocated()
    if not config["joint"] and d.base.encoder_state_sha256(model.backbone) != initial_backbone: raise ValueError("frozen backbone mutated")
    model.eval().requires_grad_(False); state = d.base.encoder_state_sha256(model)
    d.check_deadline(deadline)
    probe_vectors = corpus.vectors(model, rows["probe"], deadline)
    selection_vectors = corpus.vectors(model, rows["selection"], deadline)
    readout = corpus.ridge_fit(probe_vectors.cuda(), rows["probe"]["targets"].cuda())
    prediction = corpus.ridge_predict(readout, selection_vectors.cuda()).cpu()
    thresholds = corpus.select_thresholds(prediction, rows["selection"])
    selected = corpus.evaluate(prediction, rows["selection"], thresholds)
    d.check_deadline(deadline)
    if d.base.encoder_state_sha256(model) != state: raise ValueError("readout mutated encoder")
    name = f"trial-{config['id']:03d}-step-{steps}.pt"
    record = {"config": config, "steps": steps, "curve": curve, "selection": selected, "thresholds": thresholds,
              "checkpoint": name, "seconds": time.monotonic()-started, "train_seconds": train_seconds,
              "train_peak_allocated_bytes": train_peak, "encoder_state_sha256": state,
              "readout_parameters": (config["dimension"]+1)*14739}
    torch.save({"model": model.state_dict(), "decoder": decoder.state_dict(), "optimizer": opt.state_dict(),
                "rng": rng.get_state(), "record": record, "readout": readout,
                "selection_predictions": prediction}, output/name)
    print(json.dumps({"completed_trial": config["id"], "steps": steps, "scores": {k: v["score"] for k, v in selected.items()}}), flush=True)
    return record


def run(path, source, output, budget_path):
    start = time.monotonic(); env = json.loads(path.read_text()); p = env["payload"]
    if set(env) != {"payload", "identity_sha256"} or d.base.payload_sha256(p) != env["identity_sha256"] or p["plan"] != PLAN or p["structural_configs"] != structural_configs() or p["templates"] != templates() or p["source_encoders"] != corpus.helpers.prior.sources(source): raise ValueError("registration mismatch")
    if d.base.git("diff", p["source_commit"], "--", "theseo_anysearch/garden") or d.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"): raise ValueError("source mismatch")
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(budget_path.with_suffix(".lock")):
        ledger = reserve_budget(budget_path, PLAN["run_id"], PLAN["cap_seconds"])
        if (output/"report.json").exists():
            completed = json.loads((output/"report.json").read_text()); digest = completed.pop("report_payload_sha256")
            if d.base.payload_sha256(completed) != digest or completed["registration"] != env: raise ValueError("existing report mismatch")
            print("Completed report exists; no trials restarted.", flush=True); return
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
        progress_path = output/"progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "registration_sha256": env["identity_sha256"], "elapsed_seconds": p["preparation_seconds"], "records": [], "artifacts": {}, "inflight": None}
        if progress["registration_sha256"] != env["identity_sha256"]: raise ValueError("progress belongs to another run")
        for name, digest in progress["artifacts"].items():
            if hashlib.sha256((output/name).read_bytes()).hexdigest() != digest: raise ValueError("completed checkpoint hash mismatch")
        elapsed_before = progress["elapsed_seconds"] + (PLAN["trial_cap_seconds"] if progress["inflight"] else 0)
        deadline = start+PLAN["cap_seconds"]-elapsed_before
        rows = corpus.data(["train", "probe", "selection"]); check_support(rows)
        if corpus.identity(rows) != {k: p["data"][k] for k in rows}: raise ValueError("data mismatch")
        configs = p["structural_configs"][:]

        def execute(config, steps):
            d.check_deadline(deadline)
            if any(r["config"]["id"] == config["id"] and r["steps"] == steps for r in progress["records"]): return
            progress["inflight"] = {"id": config["id"], "steps": steps}
            progress["elapsed_seconds"] = elapsed_before+time.monotonic()-start; atomic_json(progress_path, progress)
            checkpoint = output/f"trial-{config['id']:03d}-step-{steps}.pt"
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location="cpu", weights_only=True); record = saved["record"]
                if record["config"] != config or record["steps"] != steps: raise ValueError("existing checkpoint mismatch")
            else:
                previous_records = [r for r in progress["records"] if r["config"]["id"] == config["id"]]
                old = max(previous_records, key=lambda x: x["steps"]) if previous_records else None
                record = stage(config, steps, old, rows, p, source, output, deadline)
            progress["records"].append(record); progress["inflight"] = None
            progress["artifacts"][record["checkpoint"]] = hashlib.sha256((output/record["checkpoint"]).read_bytes()).hexdigest()
            progress["elapsed_seconds"] = elapsed_before+time.monotonic()-start; atomic_json(progress_path, progress)

        for config in configs: execute(config, 512)
        first = sorted([r for r in progress["records"] if r["config"]["id"] < 18 and r["steps"] == 512], key=corpus.rank, reverse=True)[:2]
        for i, template in enumerate(p["templates"]):
            config = {**first[i%2]["config"], **template, "id": 18+i}
            configs.append(config); execute(config, 512)
        initial = sorted([r for r in progress["records"] if r["steps"] == 512], key=corpus.rank, reverse=True)
        promoted = sorted(set([r["config"]["id"] for r in initial[:6]]+PLAN["protected"]))
        for idx in promoted: execute(configs[idx], 2048)
        finalists = sorted([r for r in progress["records"] if r["steps"] == 2048], key=corpus.rank, reverse=True)[:2]
        for r in finalists: execute(r["config"], 8192)
        finalists = sorted([r for r in progress["records"] if r["steps"] == 8192], key=corpus.rank, reverse=True)
        atomic_json(output/"selection-lock.json", {"winner": finalists[0]["config"]["id"], "finalists": [r["config"]["id"] for r in finalists], "registration_sha256": env["identity_sha256"]})
        development = corpus.data(["development"])
        if corpus.identity(development) != {"development": p["data"]["development"]}: raise ValueError("development identity mismatch")
        assessments = []
        for record in finalists:
            d.check_deadline(deadline)
            saved = torch.load(output/record["checkpoint"], map_location="cuda", weights_only=True)
            model = corpus.make_encoder(record["config"], source, p["source_encoders"][0]); model.load_state_dict(saved["model"])
            z = corpus.vectors(model, development["development"], deadline)
            prediction = corpus.ridge_predict(saved["readout"], z.cuda()).cpu()
            assessments.append({"id": record["config"]["id"], "metrics": corpus.evaluate(prediction, development["development"], record["thresholds"])})
            torch.save(prediction, output/f"development-{record['config']['id']}.pt")
            del saved, model, z, prediction
        elapsed = elapsed_before+time.monotonic()-start; d.check_deadline(deadline)
        report = {"registration": env, "status": "search_tranche_completed", "records": progress["records"],
                  "winner_selection_only": finalists[0]["config"]["id"], "development": assessments,
                  "elapsed_seconds": elapsed, "promotion_eligible": False,
                  "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
                  "artifacts": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in output.glob("*.pt")}}
        report["report_payload_sha256"] = d.base.payload_sha256(report); d.base.write_json(output/"report.json", report)
        ledger["runs"][PLAN["run_id"]].update(charged_seconds=elapsed, settled=True); atomic_json(budget_path, ledger)
        print(json.dumps({"complete": True, "winner": report["winner_selection_only"], "elapsed_seconds": elapsed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--registration", type=Path, required=True); parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spec-sha"); parser.add_argument("--output", type=Path); parser.add_argument("--budget-ledger", type=Path)
    a = parser.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if a.command == "freeze": freeze(a.registration, a.spec_sha, a.source)
    else: run(a.registration, a.source, a.output, a.budget_ledger)


if __name__ == "__main__": main()
