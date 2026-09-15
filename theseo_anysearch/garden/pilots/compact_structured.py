"""Fresh spatial-layout, capacity and learning-rate compact encoder search."""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from .. import compact_structured as models
from . import compact_reconstruction as prior

corpus, d, search, heads = prior.corpus, prior.d, prior.search, prior.heads
COUNTS = {"train": 6144, "probe": 1536, "selection": 384, "development": 384}
PLAN = {"run_id": "compact-structured-v1-run1", "cap_seconds": 28800,
        "fit_cap_seconds": 3600, "preparation_cap_seconds": 5400,
        "overhead_reserve_seconds": 900, "batch": 64, "queries": 256, "steps": 8192,
        "sampling_seed": 39900, "counts": COUNTS, "bank_sizes": {"train": 2, "probe": 4},
        "source_report": "5ca3268a17847dde3bd30e6e019d48994f7140752d113bc1162ffeb594318474"}


def data(splits=None):
    return corpus.data(splits, study_id="compact-structured-v1", counts=COUNTS,
                       bank_splits=("train", "probe"), bank_sizes=PLAN["bank_sizes"])


def configs():
    result = []
    for side, channels in ((5, 1), (5, 2), (5, 4), (5, 8), (9, 1), (0, 0)):
        for lr in (.0003, .001):
            result.append({"id": len(result), "kind": "structured" if side else "grid",
                           "side": side, "channels": channels, "dimension": channels * side**3 if side else 128,
                           "lr": lr})
    return result


def check_registration(env):
    payload = env["payload"]
    if d.base.payload_sha256(payload) != env["identity_sha256"] or payload["plan"] != PLAN or payload["configs"] != configs():
        raise ValueError("structured registration mismatch")
    return payload


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
    if len(reports) != 4 or reports[-1][1] != PLAN["source_report"]:
        raise ValueError("require search/diversity/readout/reconstruction reports")
    rows = data(); corpus.support(rows)
    parents = {h for report, _ in reports for split in report["registration"]["payload"]["data"].values() for h in split["parents"]}
    if any(h in parents for row in rows.values() for h in row["parents"]):
        raise ValueError("reused parent")
    payload = {"plan": PLAN, "configs": configs(), "source_commit": d.base.git("rev-parse", "HEAD"),
               "spec_commit": spec, "spec_path": "projects/theseo-anysearch/python/perception-encoder-compact-structured.md",
               "prior_reports": [digest for _, digest in reports], "data": corpus.old.identity(rows),
               "source_encoders": reports[-1][0]["registration"]["payload"]["source_encoders"],
               "preparation_seconds": time.monotonic() - start}
    d.base.write_json(path, {"payload": payload, "identity_sha256": d.base.payload_sha256(payload)})


def restore(campaign, name, model):
    saved = campaign.load(campaign.progress["stages"][name]["checkpoint"])
    if saved["state"] != campaign.progress["stages"][name] or saved["state"]["steps"] != PLAN["steps"]:
        raise ValueError("unfinished or mismatched stage")
    model.load_state_dict(saved["model"])
    if d.base.encoder_state_sha256(model) != saved["state"]["model_state_sha256"]:
        raise ValueError("restored model mismatch")
    return model.eval().requires_grad_(False)


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
            "inflight": None, "stages": {}, "records": [], "controls": {}, "common": [], "caches": {}, "artifacts": {}}
        if progress["identity_sha256"] != env["identity_sha256"]:
            raise ValueError("progress identity mismatch")
        campaign = prior.Campaign(output, env, progress, start, prior.resume_charge(progress), plan=PLAN)
        cutoff = campaign.phase("preparation", PLAN["preparation_cap_seconds"])
        for name, digest in progress["artifacts"].items():
            if prior.previous.file_hash(output / name) != digest:
                raise ValueError("artifact mismatch")
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
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
        torch.manual_seed(399)
        spatial = campaign.fit("spatial", prior.models.SpatialReference().cuda(), "spatial", cache["probe"], rows["probe"])
        if "spatial" not in progress["controls"]:
            cutoff = campaign.phase("spatial-assessment", PLAN["fit_cap_seconds"])
            prediction = prior.predict(spatial, "spatial", cache["selection"], cutoff)
            progress["controls"]["spatial"] = prior.metrics(prediction, rows["selection"])
            campaign.artifact("spatial-selection.pt", prediction)
            print(json.dumps({"spatial_reference": progress["controls"]["spatial"]}), flush=True)
        del spatial
        for config in configs():
            number = config["id"]
            if any(r["config"] == config for r in progress["records"]):
                continue
            model = campaign.fit(f"train-{number}", models.make_model(config).cuda(), "volume", cache["train"], rows["train"], lr=config["lr"])
            cutoff = campaign.phase(f"extract-{number}", PLAN["fit_cap_seconds"])
            aggregation = model.aggregation.eval().requires_grad_(False)
            aggregation_hash = d.base.encoder_state_sha256(aggregation)
            vector_path = f"vectors-{number}.pt"
            if vector_path in progress["artifacts"]:
                stored = campaign.load(vector_path)
                if stored["aggregation_hash"] != aggregation_hash:
                    raise ValueError("vector aggregation key mismatch")
            else:
                stored = {"aggregation_hash": aggregation_hash,
                          "probe": prior.vectors(aggregation, cache["probe"], cutoff),
                          "selection": prior.vectors(aggregation, cache["selection"], cutoff)}
                campaign.artifact(vector_path, stored)
            z, statistics = models.normalized_vectors(stored["probe"], config)
            selected = heads.normalize(stored["selection"], statistics)
            trained = prior.predict(model, "volume", cache["selection"], cutoff)
            head = campaign.fit(f"probe-{number}", models.make_head(config).cuda(), "vector", z, rows["probe"])
            cutoff = campaign.phase(f"assess-{number}", PLAN["fit_cap_seconds"])
            prediction = prior.predict(head, "vector", selected, cutoff)
            result = prior.metrics(prediction, rows["selection"])
            permutation = torch.randperm(len(selected), generator=torch.Generator().manual_seed(399))
            shuffled = prior.predict(head, "vector", selected[permutation], cutoff)
            zeroed = prior.predict(head, "vector", torch.zeros_like(selected), cutoff)
            if d.base.encoder_state_sha256(aggregation) != aggregation_hash:
                raise ValueError("independent fit mutated aggregation")
            singular = torch.linalg.svdvals(stored["probe"].double() - stored["probe"].double().mean(0))
            probability = singular.square() / singular.square().sum().clamp_min(1e-30)
            record = {"config": config, "steps": PLAN["steps"], "aggregation_state_sha256": aggregation_hash,
                      "aggregation_parameters": sum(p.numel() for p in aggregation.parameters()),
                      "head_parameters": sum(p.numel() for p in head.parameters()),
                      "variance_entropy_rank": float(torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum())),
                      "trained_decoder": prior.metrics(trained, rows["selection"]),
                      "shuffled": prior.diagnosis.compact_metrics(shuffled, rows["selection"], result["thresholds"]),
                      "zeroed": prior.diagnosis.compact_metrics(zeroed, rows["selection"], result["thresholds"]), **result}
            campaign.artifact(f"assessment-{number}.pt", {"record": record, "statistics": statistics,
                              "prediction": prediction, "shuffled": shuffled, "zeroed": zeroed, "trained_prediction": trained})
            progress["records"].append(record); progress["inflight"] = None; campaign.save()
            print(json.dumps({"candidate": config, "rank": corpus.rank(record), "scores": {k: v["score"] for k, v in result["selection"].items()}}), flush=True)
            del model, aggregation, head, stored, z, selected
        for number in (0, 8, 10):
            config = configs()[number]; name = f"null-{number}"
            zeros = torch.zeros(len(cache["probe"]), config["dimension"])
            null = campaign.fit(name, models.make_head(config).cuda(), "vector", zeros, rows["probe"])
            if name not in progress["controls"]:
                cutoff = campaign.phase(f"{name}-assessment", PLAN["fit_cap_seconds"])
                prediction = prior.predict(null, "vector", torch.zeros(len(cache["selection"]), config["dimension"]), cutoff)
                progress["controls"][name] = prior.metrics(prediction, rows["selection"])
                campaign.artifact(f"{name}-selection.pt", prediction)
            del null
        finalists = sorted(progress["records"], key=corpus.rank, reverse=True)[:2]
        lock = {"identity_sha256": env["identity_sha256"], "ids": [r["config"]["id"] for r in finalists]}
        lock_path = output / "selection-lock.json"
        if lock_path.exists():
            if json.loads(lock_path.read_text()) != lock:
                raise ValueError("selection lock changed")
        else:
            d.base.write_json(lock_path, lock)
        for record in finalists:
            config = record["config"]; number = config["id"]
            if any(r["id"] == number for r in progress["common"]):
                continue
            stored = campaign.load(f"vectors-{number}.pt")
            statistics = campaign.load(f"assessment-{number}.pt")["statistics"]
            z = heads.normalize(stored["probe"], statistics)
            torch.manual_seed(399)
            head = campaign.fit(f"common-{number}", heads.ConvolutionalReadout(config["dimension"]).cuda(), "vector", z, rows["probe"])
            cutoff = campaign.phase(f"common-{number}-assessment", PLAN["fit_cap_seconds"])
            prediction = prior.predict(head, "vector", heads.normalize(stored["selection"], statistics), cutoff)
            progress["common"].append({"id": number, **prior.metrics(prediction, rows["selection"])})
            campaign.artifact(f"common-{number}-selection.pt", prediction)
            del head, stored, z
        cutoff = campaign.phase("development", PLAN["preparation_cap_seconds"])
        development = data(["development"])
        if corpus.old.identity(development) != {"development": payload["data"]["development"]}:
            raise ValueError("development identity mismatch")
        dev_cache = campaign.cache("development", backbone, development["development"], cutoff)
        assessments = []
        for record in finalists:
            config = record["config"]; number = config["id"]
            model = restore(campaign, f"train-{number}", models.make_model(config).cuda())
            statistics = campaign.load(f"assessment-{number}.pt")["statistics"]
            z = heads.normalize(prior.vectors(model.aggregation, dev_cache, cutoff), statistics)
            common = next(r for r in progress["common"] if r["id"] == number)
            for family in ("native", "common"):
                stage = f"probe-{number}" if family == "native" else f"common-{number}"
                factory = models.make_head(config) if family == "native" else heads.ConvolutionalReadout(config["dimension"])
                head = restore(campaign, stage, factory.cuda())
                prediction = prior.predict(head, "vector", z, cutoff)
                thresholds = record["thresholds"] if family == "native" else common["thresholds"]
                assessments.append({"id": number, "head": family, "metrics": prior.diagnosis.compact_metrics(prediction, development["development"], thresholds)})
                campaign.artifact(f"development-{number}-{family}.pt", prediction)
        spatial = restore(campaign, "spatial", prior.models.SpatialReference().cuda())
        prediction = prior.predict(spatial, "spatial", dev_cache, cutoff)
        campaign.artifact("spatial-development.pt", prediction)
        spatial_dev = prior.diagnosis.compact_metrics(prediction, development["development"], progress["controls"]["spatial"]["thresholds"])
        d.check_deadline(cutoff)
        evidence = {"registration": env, "status": "completed", "records": progress["records"], "stages": progress["stages"],
                    "controls": progress["controls"], "common": progress["common"], "selected_id": lock["ids"][0], "development": assessments,
                    "spatial_development": spatial_dev, "artifacts": progress["artifacts"], "caches": progress["caches"],
                    "elapsed_seconds": campaign.elapsed_before + time.monotonic() - start, "promotion_eligible": False,
                    "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()}}
        evidence["report_payload_sha256"] = d.base.payload_sha256(evidence); d.base.write_json(output / "report.json", evidence)
        ledger["runs"][PLAN["run_id"]].update(charged_seconds=campaign.elapsed_before + time.monotonic() - start, settled=True)
        search.atomic_json(ledger_path, ledger)
        progress["inflight"] = None; campaign.save()
        print(json.dumps({"completed": True, "selected_id": lock["ids"][0], "development": assessments}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--registration", type=Path, required=True); parser.add_argument("--spec-sha")
    parser.add_argument("--prior-reports", type=Path, nargs=4); parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--budget-ledger", type=Path)
    args = parser.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "freeze":
        freeze(args.registration, args.spec_sha, args.prior_reports)
    else:
        run(args.registration, args.source, args.output, args.budget_ledger)


if __name__ == "__main__":
    main()
