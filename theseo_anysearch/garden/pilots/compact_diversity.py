"""Budgeted compact training with cached exact-mask features and diversity penalties."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from torch import nn

from . import compact_diagnosis as diagnosis
from . import compact_diversity_data as corpus
from . import compact_search as search

d = corpus.d
PLAN = {"run_id": "compact-diversity-v1-run1", "cap_seconds": 4 * 3600,
        "recipe_cap_seconds": 1800, "overhead_reserve_seconds": 300,
        "rungs": [256, 512, 1024], "batch": 128, "bank": 8, "counts": corpus.COUNTS}


def configs():
    return [{"id": i, "side": 9 if i >= 4 else 5, "lr": .003 if i == 0 else .0003,
             "schedule": "constant" if i == 0 else "cosine",
             "regularizer": [0., 0., .1, 1., .1, 1.][i]} for i in range(6)]


def learning_rate(config, step):
    if not 1 <= step <= 1024:
        raise ValueError("step outside frozen schedule")
    if config["schedule"] == "constant":
        return config["lr"]
    return config["lr"] * (step / 32 if step <= 32 else .1 + .9 * (1 + math.cos(math.pi * (step - 32) / 992)) / 2)


def diversity_penalty(z):
    if z.ndim != 2 or len(z) < 2:
        raise ValueError("batch of at least two vectors required")
    centered = z - z.mean(0)
    covariance = centered.T @ centered / (len(z) - 1)
    variance = torch.relu(1 - torch.sqrt(z.var(0, unbiased=True) + 1e-4)).mean()
    off_diagonal = covariance - torch.diag_embed(covariance.diagonal())
    decorrelation = off_diagonal.square().sum() / z.shape[1]
    return 25 * variance + decorrelation


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def resume_elapsed(progress):
    return progress["elapsed_seconds"] + (PLAN["recipe_cap_seconds"] if progress["inflight"] is not None else 0)


def freeze(path, spec, source, old_registration):
    start = time.monotonic()
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec):
        raise ValueError("full spec SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit source first")
    rows = corpus.data(); corpus.support(rows)
    previous = json.loads(old_registration.read_text())
    if d.base.payload_sha256(previous["payload"]) != previous["identity_sha256"]:
        raise ValueError("prior registration mismatch")
    old_parents = {h for s in previous["payload"]["data"].values() for h in s["parents"]}
    if any(h in old_parents for v in rows.values() for h in v["parents"]):
        raise ValueError("reused source parent")
    payload = {"plan": PLAN, "configs": configs(), "source_commit": d.base.git("rev-parse", "HEAD"),
               "spec_commit": spec, "spec_path": "projects/theseo-anysearch/python/perception-encoder-compact-diversity.md",
               "source_encoders": corpus.old.helpers.prior.sources(source), "data": corpus.old.identity(rows),
               "previous_registration_sha256": previous["identity_sha256"], "preparation_seconds": time.monotonic() - start}
    d.base.write_json(path, {"payload": payload, "identity_sha256": d.base.payload_sha256(payload)})


@torch.no_grad()
def evaluate(aggregation, decoder, features, rows, deadline):
    d.check_deadline(deadline); aggregation.eval()
    original_state = d.base.encoder_state_sha256(aggregation)
    probe = aggregation(features["probe"].cuda())
    selected = aggregation(features["selection"].cuda())
    readout = corpus.old.ridge_fit(probe, rows["probe"]["targets"].cuda())
    prediction = corpus.old.ridge_predict(readout, selected).cpu()
    thresholds = corpus.old.select_thresholds(prediction, rows["selection"])
    permutation = torch.randperm(len(selected), generator=torch.Generator().manual_seed(393)).cuda()
    shuffled = corpus.old.ridge_predict(readout, selected[permutation]).cpu()
    trained = diagnosis.trained_prediction(decoder.state_dict(), selected).cpu()
    trained_thresholds = corpus.old.select_thresholds(trained, rows["selection"])
    result = {"selection": corpus.old.evaluate(prediction, rows["selection"], thresholds), "thresholds": thresholds,
              "spectrum": diagnosis.spectrum(probe.cpu()),
              "shuffled": diagnosis.compact_metrics(shuffled, rows["selection"], thresholds),
              "trained_decoder": diagnosis.compact_metrics(trained, rows["selection"], trained_thresholds),
              "trained_decoder_thresholds": trained_thresholds}
    if d.base.encoder_state_sha256(aggregation) != original_state:
        raise ValueError("readout evaluation mutated aggregation")
    aggregation.train(); d.check_deadline(deadline)
    return result, readout, prediction


def run(path, source, output, ledger_path):
    start = time.monotonic(); env = json.loads(path.read_text()); payload = env["payload"]
    if d.base.payload_sha256(payload) != env["identity_sha256"] or payload["plan"] != PLAN or payload["configs"] != configs():
        raise ValueError("registration mismatch")
    if d.base.git("diff", payload["source_commit"], "--", "theseo_anysearch/garden") or d.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("source mismatch")
    if payload["source_encoders"] != corpus.old.helpers.prior.sources(source):
        raise ValueError("initializer identity mismatch")
    output.mkdir(parents=True, exist_ok=True)
    with search.exclusive_lock(ledger_path.with_suffix(".lock")):
        ledger = search.reserve_budget(ledger_path, PLAN["run_id"], PLAN["cap_seconds"])
        if (output / "report.json").exists():
            finished = json.loads((output / "report.json").read_text()); digest = finished.pop("report_payload_sha256")
            if d.base.payload_sha256(finished) != digest or finished["registration"] != env:
                raise ValueError("completed report mismatch")
            print("Completed report exists; no training restarted.", flush=True); return
        progress_path = output / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "identity_sha256": env["identity_sha256"], "elapsed_seconds": payload["preparation_seconds"] + PLAN["overhead_reserve_seconds"],
            "inflight": None, "records": [], "artifacts": {}}
        if progress["identity_sha256"] != env["identity_sha256"]:
            raise ValueError("progress mismatch")
        for name, digest in progress["artifacts"].items():
            if file_hash(output / name) != digest:
                raise ValueError("saved artifact mismatch")
        elapsed_before = resume_elapsed(progress)
        deadline = start + PLAN["cap_seconds"] - elapsed_before
        preparation_deadline = min(deadline, start + PLAN["recipe_cap_seconds"])
        progress["inflight"] = "preparation"
        progress["elapsed_seconds"] = elapsed_before; search.atomic_json(progress_path, progress)
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
        rows = corpus.data(["train", "probe", "selection"]); corpus.support(rows)
        d.check_deadline(preparation_deadline)
        identity = corpus.old.identity(rows)
        if identity != {s: payload["data"][s] for s in rows}:
            raise ValueError("data mismatch")
        backbone = d.base.make_encoder(0, torch.device("cuda"))
        backbone.load_state_dict(torch.load(source / payload["source_encoders"][0]["path"], map_location="cuda", weights_only=True))
        backbone.eval().requires_grad_(False)
        backbone_hash = d.base.encoder_state_sha256(backbone)
        if backbone_hash != payload["source_encoders"][0]["state_hash"]:
            raise ValueError("source backbone mismatch")
        cache_path = output / "features.pt"
        if cache_path.exists():
            if progress["artifacts"].get(cache_path.name) != file_hash(cache_path):
                raise ValueError("unverified feature cache")
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cache["identity"] != identity or cache["backbone_hash"] != backbone_hash:
                raise ValueError("cache key mismatch")
        else:
            cache = {"identity": identity, "backbone_hash": backbone_hash,
                     "features": {s: corpus.features(backbone, row, preparation_deadline) for s, row in rows.items()}}
            if d.base.encoder_state_sha256(backbone) != backbone_hash:
                raise ValueError("cache extraction mutated backbone")
            torch.save(cache, cache_path); progress["artifacts"][cache_path.name] = file_hash(cache_path)
            search.atomic_json(progress_path, progress)
        target = rows["train"]["targets"].cuda()
        hidden = corpus.old.helpers.prior.crop(rows["train"]["hidden"], 17).reshape(-1, 4913).cuda()
        prevalence = target[:, :2].mean(dim=(0, 2)); weights = (1 - prevalence) / prevalence.clamp_min(1e-5)
        for config in payload["configs"]:
            existing = [r for r in progress["records"] if r["config"] == config]
            if any(r["steps"] == 1024 for r in existing):
                continue
            recipe_start = time.monotonic(); recipe_deadline = min(deadline, recipe_start + PLAN["recipe_cap_seconds"])
            progress["inflight"] = config["id"]; progress["elapsed_seconds"] = elapsed_before + time.monotonic() - start
            search.atomic_json(progress_path, progress)
            torch.manual_seed(393)
            aggregation = corpus.aggregation(config["side"]).cuda(); decoder = nn.Linear(64, 3 * 4913).cuda()
            parameters = list(aggregation.parameters()) + list(decoder.parameters())
            optimizer = torch.optim.AdamW(parameters, lr=config["lr"], weight_decay=.01)
            rng = torch.Generator(device="cuda").manual_seed(39300)
            features = {s: values[config["side"]] for s, values in cache["features"].items()}
            train_features = features["train"].cuda(); seen = torch.zeros(len(train_features), dtype=torch.bool)
            first = 0; curve = []; prior_recipe_seconds = 0.
            if existing:
                previous = max(existing, key=lambda r: r["steps"])
                saved = torch.load(output / previous["checkpoint"], map_location="cuda", weights_only=True)
                if saved["record"] != previous:
                    raise ValueError("resume record mismatch")
                aggregation.load_state_dict(saved["aggregation"]); decoder.load_state_dict(saved["decoder"])
                if d.base.encoder_state_sha256(aggregation) != previous["aggregation_state_sha256"]:
                    raise ValueError("resumed aggregation state mismatch")
                optimizer.load_state_dict(saved["optimizer"]); rng.set_state(saved["rng"].cpu())
                seen = saved["seen"].cpu(); first = previous["steps"]; curve = previous["curve"][:]
                prior_recipe_seconds = previous["recipe_elapsed_seconds"]
                recipe_deadline = min(deadline, recipe_start + PLAN["recipe_cap_seconds"] - prior_recipe_seconds)
                del saved
            torch.cuda.reset_peak_memory_stats()
            for step in range(first + 1, 1025):
                d.check_deadline(recipe_deadline)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(config, step)
                ids = torch.randint(len(train_features), (PLAN["batch"],), device="cuda", generator=rng)
                seen[ids.cpu()] = True
                z = aggregation(train_features[ids]); logits = decoder(z).reshape(-1, 3, 4913)
                reconstruction = search.objective(logits, target[ids // corpus.BANK], hidden[ids], weights,
                                                  {"boundary_weight": 1., "distance_weight": 10.})
                penalty = diversity_penalty(z); loss = reconstruction + config["regularizer"] * penalty
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training objective")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True); optimizer.step()
                if step % 128 == 0:
                    curve.append({"step": step, "reconstruction": float(reconstruction.detach()),
                                  "penalty": float(penalty.detach()), "preclip_grad_norm": float(norm),
                                  "lr": learning_rate(config, step)})
                if step in PLAN["rungs"]:
                    result, readout, prediction = evaluate(aggregation, decoder, features, rows, recipe_deadline)
                    name = f"recipe-{config['id']}-step-{step}.pt"
                    record = {"config": config, "steps": step, "curve": curve[:], "checkpoint": name,
                              "aggregation_parameters": sum(p.numel() for p in aggregation.parameters()),
                              "aggregation_state_sha256": d.base.encoder_state_sha256(aggregation),
                              "examples": step * PLAN["batch"], "unique_bank_entries": int(seen.sum()),
                              "recipe_elapsed_seconds": prior_recipe_seconds + time.monotonic() - recipe_start,
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), **result}
                    torch.save({"aggregation": aggregation.state_dict(), "decoder": decoder.state_dict(),
                                "optimizer": optimizer.state_dict(), "rng": rng.get_state(), "seen": seen,
                                "readout": readout, "selection_predictions": prediction, "record": record}, output / name)
                    progress["records"].append(record); progress["artifacts"][name] = file_hash(output / name)
                    progress["elapsed_seconds"] = elapsed_before + time.monotonic() - start
                    search.atomic_json(progress_path, progress)
                    print(json.dumps({"recipe": config["id"], "step": step, "rank": result["spectrum"]["effective_rank"],
                                      "scores": {k: v["score"] for k, v in result["selection"].items()}}), flush=True)
                    del readout, prediction
            progress["inflight"] = None; search.atomic_json(progress_path, progress)
            del aggregation, decoder, parameters, optimizer, train_features
        best = [max((r for r in progress["records"] if r["config"] == c), key=corpus.rank) for c in configs()]
        finalists = sorted(best, key=corpus.rank, reverse=True)[:2]
        search.atomic_json(output / "selection-lock.json", {"identity_sha256": env["identity_sha256"],
                           "checkpoints": [r["checkpoint"] for r in finalists]})
        progress["inflight"] = "assessment"; search.atomic_json(progress_path, progress)
        development = corpus.data(["development"])
        if corpus.old.identity(development) != {"development": payload["data"]["development"]}:
            raise ValueError("development identity mismatch")
        dev_features = corpus.features(backbone, development["development"], deadline)
        if d.base.encoder_state_sha256(backbone) != backbone_hash:
            raise ValueError("assessment mutated backbone")
        assessments = []
        for record in finalists:
            d.check_deadline(deadline)
            saved = torch.load(output / record["checkpoint"], map_location="cuda", weights_only=True)
            aggregation = corpus.aggregation(record["config"]["side"]).cuda().eval().requires_grad_(False)
            aggregation.load_state_dict(saved["aggregation"])
            with torch.no_grad():
                vectors = aggregation(dev_features[record["config"]["side"]].cuda())
                prediction = corpus.old.ridge_predict(saved["readout"], vectors).cpu()
            assessments.append({"checkpoint": record["checkpoint"],
                                "metrics": corpus.old.evaluate(prediction, development["development"], record["thresholds"])})
            name = f"development-{record['config']['id']}.pt"; torch.save(prediction, output / name)
            progress["artifacts"][name] = file_hash(output / name)
        prior = rows["probe"]["targets"].mean(0, keepdim=True).expand(len(rows["selection"]["ids"]), -1, -1)
        prior_thresholds = corpus.old.select_thresholds(prior, rows["selection"])
        d.check_deadline(deadline)
        evidence = {"registration": env, "status": "completed", "records": progress["records"],
                    "selected_checkpoint": finalists[0]["checkpoint"], "development": assessments,
                    "spatial_prior": diagnosis.compact_metrics(prior, rows["selection"], prior_thresholds),
                    "artifacts": progress["artifacts"], "promotion_eligible": False,
                    "elapsed_seconds": elapsed_before + time.monotonic() - start,
                    "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}}
        evidence["report_payload_sha256"] = d.base.payload_sha256(evidence)
        d.base.write_json(output / "report.json", evidence)
        ledger["runs"][PLAN["run_id"]].update(charged_seconds=elapsed_before + time.monotonic() - start, settled=True)
        search.atomic_json(ledger_path, ledger)
        progress["inflight"] = None; progress["elapsed_seconds"] = ledger["runs"][PLAN["run_id"]]["charged_seconds"]
        search.atomic_json(progress_path, progress)
        print(json.dumps({"complete": True, "selected": evidence["selected_checkpoint"], "elapsed": evidence["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--registration", type=Path, required=True); parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spec-sha"); parser.add_argument("--old-registration", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--budget-ledger", type=Path)
    args = parser.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "freeze":
        freeze(args.registration, args.spec_sha, args.source, args.old_registration)
    else:
        run(args.registration, args.source, args.output, args.budget_ledger)


if __name__ == "__main__":
    main()
