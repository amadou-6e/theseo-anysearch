"""Matched-trajectory final versus selection-best checkpoint diagnostic."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import diagnostic_execution as d

PLAN = {**d.PLAN, "study_id": "voxel-encoder-checkpoint-diagnostic-v1",
        "dataset_id": "voxel-encoder-checkpoint-diagnostic-v1-development-1",
        "query_id": "voxel-encoder-checkpoint-diagnostic-v1-queries-1",
        "streams": {"train": 36201, "selection": 36202, "assessment": 36203},
        "checkpoint_interval": 256, "bootstrap_seed": 362,
        "prior_registration": "docs/perception-encoder-local-geometry/diagnostics-preregistration.json"}


def fresh_data() -> dict:
    data = d.corpus(PLAN)
    prior = json.loads(Path(PLAN["prior_registration"]).read_text())["payload"]["data"]
    old_hashes = {h for split in prior.values() for h in split["occupancy_hashes"]}
    if any(h in old_hashes for split in data.values() for h in split["hashes"]):
        raise ValueError("prior diagnostic geometry reused")
    return data


def freeze(path: Path, spec: str, artifacts: Path, source_report: Path) -> None:
    started = time.monotonic()
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec):
        raise ValueError("full specification SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit source before registration")
    contract = d.lg3.artifact_contract(source_report)
    files = {k: v for k, v in contract["files"].items() if k.endswith("encoder.pt")}
    for name, digest in files.items():
        if hashlib.sha256((artifacts / name).read_bytes()).hexdigest() != digest:
            raise ValueError("checkpoint mismatch")
    payload = {"plan": PLAN, "source_commit": d.base.git("rev-parse", "HEAD"),
               "spec_commit": spec, "spec_path": "projects/theseo-anysearch/python/perception-encoder-checkpoint-diagnostic.md",
               "files": files, "states": contract["states"], "data": d.data_contract(fresh_data()),
               "prior_registration_sha256": hashlib.sha256(Path(PLAN["prior_registration"]).read_bytes()).hexdigest(),
               "preparation_seconds": time.monotonic()-started}
    d.base.write_json(path, {"payload": payload, "identity_sha256": d.base.payload_sha256(payload)})


def paired_gain(a: list, b: list) -> dict:
    """Positive means a has lower loss than b, paired across seeds/geometries."""
    delta = np.mean([np.asarray(y["geometry_log_loss"])-np.asarray(x["geometry_log_loss"])
                     for x, y in zip(a, b)], axis=0)
    rng = np.random.default_rng(362)
    strata = [np.arange(i, len(delta), 12) for i in range(12)]
    samples = [float(delta[np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])].mean())
               for _ in range(2000)]
    return {"mean": float(delta.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist()}


def assess(trials: list) -> dict:
    if sorted(t["seed"] for t in trials) != [0, 1, 2]:
        return {"outcome": "inconclusive", "next_action": "assess incomplete evidence"}
    best, final, probe = ([t[k] for t in trials] for k in ("best", "final", "probe"))
    gain, relative = paired_gain(best, final), paired_gain(best, probe)
    if gain["ci95"][0] <= 0:
        outcome, action = "selection_benefit_unresolved", "investigate data coverage and observation controls"
    elif relative["ci95"][0] > 0:
        outcome, action = "reference_exceeds_probe", "investigate encoder training or representation"
    elif relative["ci95"][1] < 0:
        outcome, action = "reference_improves_below_probe", "refine supervised reference before attributing encoder failure"
    else:
        outcome, action = "reference_probe_difference_unresolved", "retain paired uncertainty; no architecture conclusion"
    return {"outcome": outcome, "next_action": action, "best_minus_final_loss_gain": gain,
            "best_minus_probe_loss_gain": relative, "promotion_eligible": False}


def train(data: dict, seed: int, deadline: float) -> tuple:
    torch.manual_seed(36010+seed)
    model = d.Reference().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    inputs, labels = d.visible(data["train"]), data["train"]["labels"]
    rng = torch.Generator(device="cuda").manual_seed(36020+seed)
    best_loss, best_state, best_step, curve = float("inf"), None, None, []
    sy = data["selection"]["labels"].cpu().numpy()
    for step in range(2048):
        d.check_deadline(deadline)
        idx = torch.randint(len(inputs), (8,), generator=rng, device="cuda")
        logits = model(inputs[idx]).flatten(1).gather(1, data["train"]["indices"][idx])
        loss = F.binary_cross_entropy_with_logits(logits, labels[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite reference loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if (step+1) % 256 == 0:
            selection_loss = float(d.logloss_rows(d.ref_predict(model, data["selection"]), sy).mean())
            curve.append({"step": step+1, "training_loss": float(loss.detach()), "selection_loss": selection_loss})
            if selection_loss < best_loss:
                best_loss, best_step, best_state = selection_loss, step+1, copy.deepcopy(model.state_dict())
            print(json.dumps({"seed": seed, **curve[-1]}), flush=True)
    d.check_deadline(deadline)
    best = d.Reference().cuda()
    best.load_state_dict(best_state)
    return model.eval(), best.eval(), best_step, curve


def run(registration: Path, artifacts: Path, output: Path) -> dict:
    started = time.monotonic()
    env = json.loads(registration.read_text())
    p = env["payload"]
    if set(env) != {"payload", "identity_sha256"} or d.base.payload_sha256(p) != env["identity_sha256"] or p["plan"] != PLAN:
        raise ValueError("registration mismatch")
    if d.base.git("diff", p["source_commit"], "--", "theseo_anysearch/garden") or d.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("source mismatch")
    if hashlib.sha256(Path(PLAN["prior_registration"]).read_bytes()).hexdigest() != p["prior_registration_sha256"]:
        raise ValueError("prior identity mismatch")
    for name, sha in p["files"].items():
        if hashlib.sha256((artifacts / name).read_bytes()).hexdigest() != sha:
            raise ValueError("checkpoint mismatch")
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    deadline = started + 7200-p["preparation_seconds"]
    report = {"registration": env, "trials": [], "encoder_updates": 0}
    try:
        data = fresh_data()
        if d.data_contract(data) != p["data"]:
            raise ValueError("data identity mismatch")
        data = {s: {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in rows.items()} for s, rows in data.items()}
        report["D0"] = d.d0(min(deadline, time.monotonic()+300))
        if report["D0"]["outcome"] != "pass":
            raise ValueError("D0 sanity failed")
        y, sy = (data[s]["labels"].cpu().numpy() for s in ("assessment", "selection"))
        for seed in [0, 1, 2]:
            final, best, best_step, curve = train(data, seed, min(deadline, time.monotonic()+800))
            trial = {"seed": seed, "best_step": best_step, "curve": curve}
            for name, model in (("final", final), ("best", best)):
                sp = d.ref_predict(model, data["selection"])
                threshold = d.threshold(sp, sy)
                predictions = d.ref_predict(model, data["assessment"])
                trial[name] = d.evaluate(predictions, y, threshold)
                torch.save({"state": model.state_dict(), "predictions": predictions, "threshold": threshold}, output / f"{name}-{seed}.pt")
            encoder = d.base.make_encoder(seed, torch.device("cuda"))
            encoder.load_state_dict(torch.load(artifacts / f"joint-{seed}-encoder.pt", map_location="cuda", weights_only=True))
            encoder.eval().requires_grad_(False)
            before = d.base.encoder_state_sha256(encoder)
            if before != p["states"][str(seed)]["final_state_sha256"]:
                raise ValueError("encoder state mismatch")
            features = {}
            with torch.no_grad():
                for split, rows in data.items():
                    values = []
                    for start in range(0, len(rows["labels"]), 8):
                        occ, hidden = rows["occupancy"][start:start+8], rows["hidden"][start:start+8]
                        level = d.base.VoxelLevel.from_occupancy(occ, unknown_mask=hidden)
                        volume = encoder(level, hidden).local_feature_volume
                        values.append(d.base.gather_features(volume, rows["indices"][start:start+8]))
                    features[split] = torch.cat(values)
            candidates = []
            probe_deadline = min(deadline, time.monotonic()+800)
            for width in [0, 32, 128]:
                model, mean, scale, curve = d.fit_probe(features, data, seed, width, probe_deadline)
                with torch.no_grad():
                    sp = model((features["selection"]-mean)/scale)[..., 0].sigmoid().cpu().numpy()
                    prediction = model((features["assessment"]-mean)/scale)[..., 0].sigmoid().cpu().numpy()
                threshold = d.threshold(sp, sy)
                candidates.append({"width": width, "selection_loss": float(d.logloss_rows(sp, sy).mean()),
                                   "assessment": d.evaluate(prediction, y, threshold), "curve": curve})
                torch.save({"state": model.state_dict(), "mean": mean, "scale": scale,
                            "predictions": prediction}, output / f"probe-{seed}-{width}.pt")
            selected = min(candidates, key=lambda c: c["selection_loss"])
            trial.update({"probe": selected["assessment"], "selected_width": selected["width"],
                          "capacities": candidates, "encoder_state_before": before,
                          "encoder_state_after": d.base.encoder_state_sha256(encoder)})
            if trial["encoder_state_after"] != before:
                raise ValueError("encoder mutated")
            report["trials"].append(trial)
        report["assessment"] = assess(report["trials"])
        report["validity"] = "valid"
    except TimeoutError as exc:
        report.update({"validity": "valid", "assessment": {"outcome": "inconclusive", "reason": str(exc)}})
    except (ValueError, FloatingPointError) as exc:
        report.update({"validity": "invalid", "assessment": {"outcome": "repair_required", "reason": str(exc)}})
    report["elapsed_seconds"] = time.monotonic()-started+p["preparation_seconds"]
    report["environment"] = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__}
    report["artifacts"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in output.glob("*.pt")}
    report["report_payload_sha256"] = d.base.payload_sha256(report)
    d.base.write_json(output / "report.json", report)
    print(json.dumps(report["assessment"]), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--spec-sha")
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "freeze":
        freeze(args.registration, args.spec_sha, args.artifacts, args.source_report)
    else:
        run(args.registration, args.artifacts, args.output)


if __name__ == "__main__":
    main()
