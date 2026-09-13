"""Post-hoc compact search diagnosis, without development access or weight fitting."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from . import compact_search as search
from . import compact_search_data as corpus

d = search.d
PLAN = {"run_id": "compact-search-diagnosis-v1-run1", "cap_seconds": 900,
        "verification_reserve_seconds": 120, "rungs": [512, 2048, 8192],
        "search_report_sha256": "572e8910c6d1934f59662a9b79312f0a08e482fb91dfe2d832f08a6bca8048a6"}


def spectrum(vectors):
    x = vectors.double()
    centered = x - x.mean(0)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / len(x)).clamp_min(0)
    total = eigenvalues.sum()
    probabilities = eigenvalues / total.clamp_min(1e-30)
    entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum()
    return {"dimension": x.shape[1], "effective_rank": float(entropy.exp()) if total > 0 else 0.,
            "leading_fraction": float(probabilities[-1]), "eigenvalues": eigenvalues.tolist(),
            "standard_deviations": x.std(0, unbiased=False).tolist(),
            "mean_vector_norm": float(x.norm(dim=1).mean())}


def trained_prediction(state, vectors):
    logits = F.linear(vectors, state["weight"], state["bias"]).reshape(len(vectors), 3, 4913)
    return torch.cat((logits[:, :2].sigmoid(), logits[:, 2:].clamp(0, 1)), 1)


def compact_metrics(prediction, rows, thresholds):
    return {k: {"score": v["score"], "families": v["families"]}
            for k, v in corpus.evaluate(prediction, rows, thresholds).items()}


def read_search(directory):
    report = json.loads((directory / "report.json").read_text())
    digest = report.pop("report_payload_sha256")
    if digest != PLAN["search_report_sha256"] or d.base.payload_sha256(report) != digest:
        raise ValueError("search report mismatch")
    return report


def freeze(path, spec, directory):
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec):
        raise ValueError("full spec SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit source first")
    report = read_search(directory)
    ids = [r["config"]["id"] for r in report["records"] if r["steps"] == 8192]
    records = [r for r in report["records"] if r["config"]["id"] in ids]
    if len(ids) != 2 or len(records) != 6:
        raise ValueError("expected two finalists and three rungs")
    payload = {"plan": PLAN, "spec_commit": spec, "source_commit": d.base.git("rev-parse", "HEAD"),
               "spec_path": "projects/theseo-anysearch/python/perception-encoder-compact-diagnosis.md",
               "registration_sha256": report["registration"]["identity_sha256"],
               "checkpoints": {r["checkpoint"]: report["artifacts"][r["checkpoint"]] for r in records}}
    d.base.write_json(path, {"payload": payload, "identity_sha256": d.base.payload_sha256(payload)})


def ensure_unstarted(output):
    if (output / "started.json").exists() or (output / "report.json").exists():
        raise ValueError("analysis already started; retain reservation and audit before any new run")


def run(path, directory, source, output, ledger_path):
    start = time.monotonic()
    env = json.loads(path.read_text()); payload = env["payload"]
    if d.base.payload_sha256(payload) != env["identity_sha256"] or payload["plan"] != PLAN:
        raise ValueError("analysis registration mismatch")
    if d.base.git("diff", payload["source_commit"], "--", "theseo_anysearch/garden") or d.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("analysis source mismatch")
    report = read_search(directory)
    if report["registration"]["identity_sha256"] != payload["registration_sha256"]:
        raise ValueError("source registration mismatch")
    with search.exclusive_lock(ledger_path.with_suffix(".lock")):
        ensure_unstarted(output)
        ledger = search.reserve_budget(ledger_path, PLAN["run_id"], PLAN["cap_seconds"])
        d.base.write_json(output / "started.json", {"registration_sha256": env["identity_sha256"]})
        deadline = start + PLAN["cap_seconds"] - PLAN["verification_reserve_seconds"]
        torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
        rows = corpus.data(["probe", "selection"])
        registered = report["registration"]["payload"]
        if corpus.identity(rows) != {s: registered["data"][s] for s in rows}:
            raise ValueError("analysis data mismatch")
        if corpus.helpers.prior.sources(source) != registered["source_encoders"]:
            raise ValueError("initializer mismatch")
        corpus_mean = rows["probe"]["targets"].mean(0, keepdim=True).expand(len(rows["selection"]["ids"]), -1, -1)
        prior_thresholds = corpus.select_thresholds(corpus_mean, rows["selection"])
        results = []
        for record in report["records"]:
            name = record["checkpoint"]
            if name not in payload["checkpoints"]:
                continue
            d.check_deadline(deadline)
            with (directory / name).open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != payload["checkpoints"][name]:
                raise ValueError("checkpoint hash mismatch")
            saved = torch.load(directory / name, map_location="cuda", weights_only=True)
            if saved["record"] != record:
                raise ValueError("checkpoint record mismatch")
            model = corpus.make_encoder(record["config"], source, registered["source_encoders"][0])
            model.load_state_dict(saved["model"]); model.eval().requires_grad_(False)
            state_hash = d.base.encoder_state_sha256(model)
            if state_hash != record["encoder_state_sha256"]:
                raise ValueError("encoder state mismatch")
            probe = corpus.vectors(model, rows["probe"], deadline)
            z = corpus.vectors(model, rows["selection"], deadline).cuda()
            readout = saved["readout"]
            prediction = corpus.ridge_predict(readout, z)
            if not torch.allclose(prediction, saved["selection_predictions"], rtol=1e-5, atol=1e-6):
                raise ValueError("regenerated prediction mismatch")
            decoder = trained_prediction(saved["decoder"], z)
            decoder_thresholds = corpus.select_thresholds(decoder.cpu(), rows["selection"])
            permutation = torch.randperm(len(z), generator=torch.Generator().manual_seed(391)).cuda()
            variants = {"ridge": prediction, "trained_decoder": decoder,
                        "mean_vector": corpus.ridge_predict(readout, probe.mean(0).cuda().expand_as(z)),
                        "zero_vector": corpus.ridge_predict(readout, torch.zeros_like(z)),
                        "shuffled_vector": corpus.ridge_predict(readout, z[permutation])}
            result = {"checkpoint": name, "config": record["config"], "steps": record["steps"],
                      "probe_spectrum": spectrum(probe), "trained_decoder_thresholds": decoder_thresholds,
                      "metrics": {k: compact_metrics(v.cpu(), rows["selection"], decoder_thresholds if k == "trained_decoder" else record["thresholds"])
                                  for k, v in variants.items()}}
            if d.base.encoder_state_sha256(model) != state_hash:
                raise ValueError("analysis mutated encoder")
            results.append(result)
            print(json.dumps({"checkpoint": name, "rank": result["probe_spectrum"]["effective_rank"],
                              "boundary": {k: v["boundary_f1"]["score"] for k, v in result["metrics"].items()}}), flush=True)
            del saved, model, probe, z, readout, prediction, decoder, variants
        d.check_deadline(deadline)
        evidence = {"registration": env, "status": "post_hoc_diagnosis_completed", "promotion_eligible": False,
                    "results": results, "prior_thresholds": prior_thresholds,
                    "spatial_prior": compact_metrics(corpus_mean, rows["selection"], prior_thresholds),
                    "elapsed_seconds": time.monotonic() - start,
                    "verification_reserve_seconds": PLAN["verification_reserve_seconds"]}
        evidence["report_payload_sha256"] = d.base.payload_sha256(evidence)
        d.base.write_json(output / "report.json", evidence)
        ledger["runs"][PLAN["run_id"]].update(charged_seconds=time.monotonic()-start+PLAN["verification_reserve_seconds"], settled=True)
        search.atomic_json(ledger_path, ledger)
        print(json.dumps({"complete": True, "report_payload_sha256": evidence["report_payload_sha256"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--search", type=Path, required=True)
    parser.add_argument("--spec-sha"); parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--budget-ledger", type=Path)
    args = parser.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "freeze":
        freeze(args.registration, args.spec_sha, args.search)
    else:
        run(args.registration, args.search, args.source, args.output, args.budget_ledger)


if __name__ == "__main__":
    main()
