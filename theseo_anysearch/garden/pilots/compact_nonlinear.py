"""Independent nonlinear head training on frozen compact voxel encoders."""
import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from .. import compact_readout as heads
from . import compact_diversity as previous
from . import compact_diversity_data as corpus
from . import compact_diagnosis as diagnosis
from . import compact_search as search

d = corpus.d
COUNTS = {"probe": 1536, "selection": 192, "development": 192, "calibration": 12}
PLAN = {"run_id": "compact-readout-v1-run1", "cap_seconds": 14400, "fit_cap_seconds": 900,
        "overhead_reserve_seconds": 300, "batch": 32, "queries": 256, "counts": COUNTS,
        "source_report": "02701cbde1957a598460a86b0a4321f3ff39344a3a468c0f54dd3b943787fbe0",
        "checkpoints": ["recipe-0-step-512.pt", "recipe-5-step-1024.pt"]}


def data(splits=None):
    return corpus.data(splits, study_id="compact-readout-v1", counts=COUNTS, bank_splits=("probe",))


def configs():
    result = []
    for source in range(2):
        for kind in ("conv", "film"):
            for lr in (.0003, .001, .003):
                result.append({"id": len(result), "source": source, "kind": kind, "lr": lr, "mode": "candidate", "steps": 4096})
    for kind in ("conv", "film"):
        result.append({"id": len(result), "source": 0, "kind": kind, "lr": .001, "mode": "null", "steps": 4096})
    for kind in ("conv", "film"):
        for lr in (.0003, .001, .003):
            result.append({"id": len(result), "source": None, "kind": kind, "lr": lr, "mode": "oracle", "steps": 2048})
    return result[14:] + result[:14]


def learning_rate(config, step):
    if not 1 <= step <= config["steps"]:
        raise ValueError("step outside frozen schedule")
    return config["lr"] * (step / 64 if step <= 64 else .1 + .9 * (1 + math.cos(math.pi * (step - 64) / (config["steps"] - 64))) / 2)


def read_source(directory):
    report = json.loads((directory / "report.json").read_text()); digest = report.pop("report_payload_sha256")
    if digest != PLAN["source_report"] or d.base.payload_sha256(report) != digest:
        raise ValueError("source report mismatch")
    return report


def freeze(path, spec, directory, old_registration):
    start = time.monotonic()
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec):
        raise ValueError("full spec SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit source first")
    source = read_source(directory)
    rows = data(); corpus.support(rows)
    old = json.loads(old_registration.read_text())
    if d.base.payload_sha256(old["payload"]) != old["identity_sha256"]:
        raise ValueError("old registration mismatch")
    parents = {h for p in (old["payload"], source["registration"]["payload"]) for s in p["data"].values() for h in s["parents"]}
    if any(h in parents for r in rows.values() for h in r["parents"]):
        raise ValueError("reused parent")
    sources = []
    for name in PLAN["checkpoints"]:
        record = next(r for r in source["records"] if r["checkpoint"] == name)
        if previous.file_hash(directory / name) != source["artifacts"][name]:
            raise ValueError("source checkpoint mismatch")
        sources.append({"checkpoint": name, "artifact_sha256": source["artifacts"][name], "record": record})
    payload = {"plan": PLAN, "configs": configs(), "source_commit": d.base.git("rev-parse", "HEAD"), "spec_commit": spec,
               "spec_path": "projects/theseo-anysearch/python/perception-encoder-compact-readout.md", "sources": sources,
               "data": corpus.old.identity(rows), "source_encoders": source["registration"]["payload"]["source_encoders"],
               "preparation_seconds": time.monotonic() - start}
    d.base.write_json(path, {"payload": payload, "identity_sha256": d.base.payload_sha256(payload)})


@torch.no_grad()
def predict(head, vectors, deadline):
    head.eval(); predictions = []
    for i in range(0, len(vectors), 8):
        d.check_deadline(deadline)
        z = vectors[i:i + 8].cuda()
        indices = torch.arange(4913, device="cuda")[None].expand(len(z), -1)
        logits = head(z, indices)
        predictions.append(torch.cat((logits[:, :2].sigmoid(), logits[:, 2:].clamp(0, 1)), 1).cpu())
    head.train()
    return torch.cat(predictions)


def oracle_scores(prediction, targets):
    scores = {}; thresholds = {}
    for channel, task in enumerate(d.base.TASKS[:2]):
        p, y = prediction[:, channel].numpy(), targets[:, channel].numpy()
        threshold = d.threshold(p.flatten(), y.flatten()); thresholds[task] = threshold
        yes, truth = p >= threshold, y > .5
        tp = (yes & truth).sum(); fp = (yes & ~truth).sum(); fn = (~yes & truth).sum()
        scores[task] = float(tp / (tp + fp + fn) if channel == 0 else 2 * tp / (2 * tp + fp + fn))
    free = targets[:, 0] < .5
    scores["free_nmae"] = float((prediction[:, 2][free] - targets[:, 2][free]).abs().mean())
    return {"scores": scores, "thresholds": thresholds,
            "overfit_targets_met": scores["occupied_iou"] >= .95 and scores["boundary_f1"] >= .97}


def rank(record):
    return corpus.rank({**record, "aggregation_parameters": record["head_parameters"]})


def resume_elapsed(progress):
    return progress["elapsed_seconds"] + (PLAN["fit_cap_seconds"] if progress["inflight"] is not None else 0)


def run(path, directory, source, output, ledger_path):
    start = time.monotonic(); env = json.loads(path.read_text()); payload = env["payload"]
    if d.base.payload_sha256(payload) != env["identity_sha256"] or payload["plan"] != PLAN or payload["configs"] != configs():
        raise ValueError("registration mismatch")
    if d.base.git("diff", payload["source_commit"], "--", "theseo_anysearch/garden") or d.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("source mismatch")
    read_source(directory)
    if payload["source_encoders"] != corpus.old.helpers.prior.sources(source):
        raise ValueError("initializer mismatch")
    output.mkdir(parents=True, exist_ok=True)
    with search.exclusive_lock(ledger_path.with_suffix(".lock")):
        ledger = search.reserve_budget(ledger_path, PLAN["run_id"], PLAN["cap_seconds"])
        if (output / "report.json").exists():
            finished = json.loads((output / "report.json").read_text()); digest = finished.pop("report_payload_sha256")
            if d.base.payload_sha256(finished) != digest or finished["registration"] != env:
                raise ValueError("completed report mismatch")
            print("Completed report exists; no fits restarted.", flush=True); return
        progress_path = output / "progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {
            "identity_sha256": env["identity_sha256"], "elapsed_seconds": payload["preparation_seconds"] + PLAN["overhead_reserve_seconds"],
            "inflight": None, "records": [], "artifacts": {}}
        if progress["identity_sha256"] != env["identity_sha256"]:
            raise ValueError("progress mismatch")
        for name, digest in progress["artifacts"].items():
            if previous.file_hash(output / name) != digest:
                raise ValueError("saved artifact mismatch")
        elapsed_before = resume_elapsed(progress); deadline = start + PLAN["cap_seconds"] - elapsed_before
        preparation_deadline = min(deadline, start + PLAN["fit_cap_seconds"])
        progress["inflight"] = "preparation"; progress["elapsed_seconds"] = elapsed_before
        search.atomic_json(progress_path, progress)
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
        rows = data(["probe", "selection", "calibration"]); corpus.support(rows)
        identity = corpus.old.identity(rows)
        if identity != {s: payload["data"][s] for s in rows}:
            raise ValueError("data identity mismatch")
        d.check_deadline(preparation_deadline)
        backbone = d.base.make_encoder(0, torch.device("cuda"))
        backbone.load_state_dict(torch.load(source / payload["source_encoders"][0]["path"], map_location="cuda", weights_only=True))
        backbone.eval().requires_grad_(False)
        backbone_hash = d.base.encoder_state_sha256(backbone)
        if backbone_hash != payload["source_encoders"][0]["state_hash"]:
            raise ValueError("backbone mismatch")
        aggregations = []
        for source_record in payload["sources"]:
            if previous.file_hash(directory / source_record["checkpoint"]) != source_record["artifact_sha256"]:
                raise ValueError("aggregation artifact mismatch")
            saved = torch.load(directory / source_record["checkpoint"], map_location="cuda", weights_only=True)
            aggregation = corpus.aggregation(source_record["record"]["config"]["side"]).cuda().eval().requires_grad_(False)
            aggregation.load_state_dict(saved["aggregation"])
            if d.base.encoder_state_sha256(aggregation) != source_record["record"]["aggregation_state_sha256"]:
                raise ValueError("aggregation state mismatch")
            aggregations.append(aggregation)
        del saved

        def extract(row, cutoff):
            features = corpus.features(backbone, row, cutoff)
            with torch.no_grad():
                result = [a(features[r["record"]["config"]["side"]].cuda()).cpu() for a, r in zip(aggregations, payload["sources"])]
            if d.base.encoder_state_sha256(backbone) != backbone_hash or any(d.base.encoder_state_sha256(a) != r["record"]["aggregation_state_sha256"] for a, r in zip(aggregations, payload["sources"])):
                raise ValueError("vector extraction mutated encoder")
            return result

        cache_path = output / "vectors.pt"
        if cache_path.exists():
            if progress["artifacts"].get(cache_path.name) != previous.file_hash(cache_path):
                raise ValueError("unverified vector cache")
            cache = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cache["identity"] != identity or cache["sources"] != payload["sources"] or cache["backbone_hash"] != backbone_hash:
                raise ValueError("vector cache key mismatch")
        else:
            cache = {"identity": identity, "sources": payload["sources"], "backbone_hash": backbone_hash,
                     "vectors": {s: extract(rows[s], preparation_deadline) for s in ("probe", "selection")}}
            torch.save(cache, cache_path); progress["artifacts"][cache_path.name] = previous.file_hash(cache_path)
            search.atomic_json(progress_path, progress)
        normalized = []
        for i in range(2):
            z, statistics = heads.normalized_vectors(cache["vectors"]["probe"][i])
            normalized.append({"probe": z.cuda(), "selection": heads.normalize(cache["vectors"]["selection"][i], statistics), "statistics": statistics})
        target = rows["probe"]["targets"].cuda()
        hidden = corpus.old.helpers.prior.crop(rows["probe"]["hidden"], 17).reshape(-1, 4913).cuda()
        oracle_target = rows["calibration"]["targets"][[1, 4, 7, 10]].cuda()
        oracle_vectors = torch.zeros(4, 64, device="cuda"); oracle_vectors[:, :4] = torch.eye(4, device="cuda")
        for config in payload["configs"]:
            existing = [r for r in progress["records"] if r["config"] == config]
            if any(r["steps"] == config["steps"] for r in existing):
                continue
            fit_start = time.monotonic(); fit_deadline = min(deadline, fit_start + PLAN["fit_cap_seconds"])
            progress["inflight"] = config["id"]; progress["elapsed_seconds"] = elapsed_before + time.monotonic() - start
            search.atomic_json(progress_path, progress)
            torch.manual_seed(395); head = heads.make_readout(config["kind"]).cuda()
            optimizer = torch.optim.AdamW(head.parameters(), lr=config["lr"], weight_decay=.01)
            rng = torch.Generator(device="cuda").manual_seed(39500)
            first = 0; curve = []; prior_seconds = 0.
            if existing:
                previous_record = max(existing, key=lambda r: r["steps"])
                saved = torch.load(output / previous_record["checkpoint"], map_location="cuda", weights_only=True)
                if saved["record"] != previous_record:
                    raise ValueError("resume record mismatch")
                head.load_state_dict(saved["head"]); optimizer.load_state_dict(saved["optimizer"]); rng.set_state(saved["rng"].cpu())
                if d.base.encoder_state_sha256(head) != previous_record["head_state_sha256"]:
                    raise ValueError("resumed head mismatch")
                first = previous_record["steps"]; curve = previous_record["curve"][:]; prior_seconds = previous_record["fit_elapsed_seconds"]
                fit_deadline = min(deadline, fit_start + PLAN["fit_cap_seconds"] - prior_seconds)
                del saved
            oracle = config["mode"] == "oracle"
            fit_target = oracle_target if oracle else target
            fit_vectors = oracle_vectors if oracle else normalized[config["source"]]["probe"]
            prevalence = fit_target[:, :2].mean(dim=(0, 2)); weights = (1 - prevalence) / prevalence.clamp_min(1e-5)
            torch.cuda.reset_peak_memory_stats()
            for step in range(first + 1, config["steps"] + 1):
                d.check_deadline(fit_deadline)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(config, step)
                ids = torch.randint(len(fit_vectors), (PLAN["batch"],), generator=rng, device="cuda")
                indices = torch.randint(4913, (PLAN["batch"], PLAN["queries"]), generator=rng, device="cuda")
                z = fit_vectors[ids]
                if config["mode"] == "null":
                    z = torch.zeros_like(z)
                logits = head(z, indices)
                query_target = fit_target[ids if oracle else ids // 8].gather(2, indices[:, None].expand(-1, 3, -1))
                query_hidden = torch.ones_like(indices, dtype=torch.bool) if oracle else hidden[ids].gather(1, indices)
                loss = search.objective(logits, query_target, query_hidden, weights, {"boundary_weight": 1., "distance_weight": 10.})
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite head loss")
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5., error_if_nonfinite=True); optimizer.step()
                if step % 512 == 0:
                    curve.append({"step": step, "minibatch_loss": float(loss.detach()), "lr": learning_rate(config, step)})
                    print(json.dumps({"fit": config["id"], "mode": config["mode"], "step": step}), flush=True)
                if step == config["steps"] or (not oracle and step == 1024):
                    state_hash = d.base.encoder_state_sha256(head)
                    if oracle:
                        prediction = predict(head, oracle_vectors, fit_deadline)
                        result = {"oracle": oracle_scores(prediction, oracle_target.cpu())}
                    else:
                        selected_vectors = normalized[config["source"]]["selection"]
                        if config["mode"] == "null":
                            selected_vectors = torch.zeros_like(selected_vectors)
                        prediction = predict(head, selected_vectors, fit_deadline)
                        thresholds = corpus.old.select_thresholds(prediction, rows["selection"])
                        result = {"selection": diagnosis.compact_metrics(prediction, rows["selection"], thresholds), "thresholds": thresholds}
                        if config["mode"] == "candidate":
                            permutation = torch.randperm(len(selected_vectors), generator=torch.Generator().manual_seed(395))
                            shuffled = predict(head, selected_vectors[permutation], fit_deadline)
                            result["shuffled"] = diagnosis.compact_metrics(shuffled, rows["selection"], thresholds)
                    if d.base.encoder_state_sha256(head) != state_hash:
                        raise ValueError("evaluation mutated head")
                    name = f"fit-{config['id']:02d}-step-{step}.pt"
                    record = {"config": config, "steps": step, "curve": curve[:], "checkpoint": name,
                              "head_state_sha256": state_hash, "head_parameters": sum(p.numel() for p in head.parameters()),
                              "sampled_queries": step * PLAN["batch"] * PLAN["queries"], "fit_elapsed_seconds": prior_seconds + time.monotonic() - fit_start,
                              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), **result}
                    torch.save({"head": head.state_dict(), "optimizer": optimizer.state_dict(), "rng": rng.get_state(),
                                "normalization": None if oracle else normalized[config["source"]]["statistics"],
                                "prediction": prediction, "record": record}, output / name)
                    progress["records"].append(record); progress["artifacts"][name] = previous.file_hash(output / name)
                    progress["elapsed_seconds"] = elapsed_before + time.monotonic() - start
                    search.atomic_json(progress_path, progress)
                    print(json.dumps({"completed_fit": config["id"], "step": step, "result": result.get("oracle", {k: v["score"] for k, v in result.get("selection", {}).items()})}), flush=True)
            progress["inflight"] = None; search.atomic_json(progress_path, progress)
            del head, optimizer
        candidates = [c for c in configs() if c["mode"] == "candidate"]
        best = [max((r for r in progress["records"] if r["config"] == c), key=rank) for c in candidates]
        finalists = sorted(best, key=rank, reverse=True)[:2]
        search.atomic_json(output / "selection-lock.json", {"identity_sha256": env["identity_sha256"], "checkpoints": [r["checkpoint"] for r in finalists]})
        progress["inflight"] = "assessment"; search.atomic_json(progress_path, progress)
        ridge = []
        for i in range(2):
            d.check_deadline(deadline)
            readout = heads.grouped_ridge(cache["vectors"]["probe"][i].cuda(), target, 8)
            prediction = corpus.old.ridge_predict(readout, cache["vectors"]["selection"][i].cuda()).cpu()
            thresholds = corpus.old.select_thresholds(prediction, rows["selection"])
            ridge.append({"source": i, "selection": diagnosis.compact_metrics(prediction, rows["selection"], thresholds), "thresholds": thresholds})
            name = f"ridge-{i}.pt"; torch.save({"readout": readout, "prediction": prediction}, output / name)
            progress["artifacts"][name] = previous.file_hash(output / name)
        development = data(["development"])
        if corpus.old.identity(development) != {"development": payload["data"]["development"]}:
            raise ValueError("development data mismatch")
        dev_vectors = extract(development["development"], deadline)
        assessments = []
        for record in finalists:
            saved = torch.load(output / record["checkpoint"], map_location="cpu", weights_only=True)
            head = heads.make_readout(record["config"]["kind"]).cuda(); head.load_state_dict(saved["head"])
            vectors = heads.normalize(dev_vectors[record["config"]["source"]], saved["normalization"])
            prediction = predict(head, vectors, deadline)
            assessments.append({"checkpoint": record["checkpoint"], "metrics": diagnosis.compact_metrics(prediction, development["development"], record["thresholds"])})
            name = f"development-{record['config']['id']}.pt"; torch.save(prediction, output / name)
            progress["artifacts"][name] = previous.file_hash(output / name)
        d.check_deadline(deadline)
        evidence = {"registration": env, "status": "completed", "records": progress["records"], "ridge_controls": ridge,
                    "selected_checkpoint": finalists[0]["checkpoint"], "development": assessments, "artifacts": progress["artifacts"],
                    "elapsed_seconds": elapsed_before + time.monotonic() - start, "promotion_eligible": False,
                    "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}}
        evidence["report_payload_sha256"] = d.base.payload_sha256(evidence); d.base.write_json(output / "report.json", evidence)
        ledger["runs"][PLAN["run_id"]].update(charged_seconds=elapsed_before + time.monotonic() - start, settled=True)
        search.atomic_json(ledger_path, ledger)
        progress["inflight"] = None; progress["elapsed_seconds"] = ledger["runs"][PLAN["run_id"]]["charged_seconds"]
        search.atomic_json(progress_path, progress)
        print(json.dumps({"complete": True, "selected": evidence["selected_checkpoint"], "elapsed": evidence["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--registration", type=Path, required=True); parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--spec-sha"); parser.add_argument("--old-registration", type=Path); parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--budget-ledger", type=Path)
    args = parser.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "freeze":
        freeze(args.registration, args.spec_sha, args.source_run, args.old_registration)
    else:
        run(args.registration, args.source_run, args.source, args.output, args.budget_ledger)


if __name__ == "__main__":
    main()
