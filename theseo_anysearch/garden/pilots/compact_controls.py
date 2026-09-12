"""Development diagnostic separating compact information from probe calibration."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..compact import CompactEncoder, query_coordinates, query_features
from ..evaluation.metrics import binary_ranking_metrics
from . import compact_smoke as previous

d = previous.d
PLAN = {"study_id": "compact-controls-v1", "counts": {"probe": 48, "selection": 24, "development": 48},
        "query_seed": 38100, "tasks": ["occupied_iou", "boundary_f1"],
        "recipes": {"plain128": [128, False], "plain1024": [1024, False], "balanced1024": [1024, True]},
        "cap_seconds": 1800, "quality_claim": False, "seed": 0}


def data():
    rows = previous.data(study_id=PLAN["study_id"], counts=PLAN["counts"], query_seed=PLAN["query_seed"])
    for values in rows.values():
        for task in PLAN["tasks"]:
            for family in range(2):
                y = values["targets"][task][[i for i in range(len(values["ids"])) if (i % 6)//3 == family]]
                if min(float(y.sum()), float((1-y).sum())) < 20:
                    raise ValueError("insufficient family class support")
    return rows


def sources(root):
    report = json.loads(Path("docs/perception-encoder-local-geometry/compact-smoke-v2-report.json").read_text())
    sha = report.pop("report_payload_sha256")
    if d.base.payload_sha256(report) != sha: raise ValueError("source report mismatch")
    for name, digest in report["artifacts"].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest: raise ValueError("source checkpoint mismatch")
    return {"report_sha256": sha, "artifacts": report["artifacts"],
            "states": {t["name"]: t["encoder_state_sha256"] for t in report["trials"]}}


def freeze(path, spec, root, encoder_root):
    started = time.monotonic()
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec): raise ValueError("full spec SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"): raise ValueError("commit source first")
    p = {"plan": PLAN, "source_commit": d.base.git("rev-parse", "HEAD"), "spec_commit": spec,
         "spec_path": "projects/theseo-anysearch/python/perception-encoder-compact-controls.md",
         "compact_sources": sources(root), "spatial_sources": previous.prior.sources(encoder_root),
         "data": previous.identity(data()), "preparation_seconds": time.monotonic()-started}
    d.base.write_json(path, {"payload": p, "identity_sha256": d.base.payload_sha256(p)})


def raw_features(occupancy, hidden, indices):
    channels = torch.stack((occupancy.bool() & ~hidden.bool(), hidden.bool()), 1).float()
    offsets = torch.tensor(list(itertools.product((-1, 0, 1), repeat=3)), device=indices.device)
    xyz = torch.stack((indices//289, indices//17 % 17, indices % 17), -1) + 8
    xyz = xyz[:, :, None] + offsets
    locations = xyz[..., 0]*33**2 + xyz[..., 1]*33 + xyz[..., 2]
    features = [c.flatten(1).gather(1, locations.flatten(1)).reshape(len(c), -1, 27) for c in channels.unbind(1)]
    return torch.cat((*features, query_coordinates(indices)), -1)


@torch.no_grad()
def features(name, model, rows, deadline):
    result = {}
    for split, v in rows.items():
        arrays = {task: [] for task in PLAN["tasks"]}
        for start in range(0, len(v["ids"]), 4):
            d.check_deadline(deadline)
            sl = slice(start, start+4)
            occ = v["occupancy"][sl].to("cuda", dtype=torch.float32)
            hidden = v["hidden"][sl].cuda()
            level = d.base.VoxelLevel.from_occupancy(occ, unknown_mask=hidden[:, None])
            encoded = model(level, hidden[:, None]) if model is not None else None
            for task in arrays:
                idx = v["indices"][task][sl].cuda()
                if name == "coordinates": x = query_coordinates(idx)
                elif name == "raw_neighborhood": x = raw_features(occ, hidden, idx)
                elif name == "spatial":
                    x = torch.cat((d.base.gather_features(encoded.local_feature_volume, previous.prior.mapped_indices(idx, 33)), query_coordinates(idx)), -1)
                else: x = query_features(encoded, idx)
                arrays[task].append(x.cpu())
        result[split] = {task: torch.cat(v) for task, v in arrays.items()}
    return result


def fit(x, y, steps, balanced, deadline):
    x, y = x.flatten(0, 1), y.flatten()
    positive = y.sum(); negative = (1-y).sum()
    if min(float(positive), float(negative)) <= 0: raise ValueError("both classes required")
    weight = negative / positive if balanced else torch.ones((), device=x.device)
    mean = x.mean(0); scale = x.std(0, unbiased=False).clamp_min(1e-5); x = (x-mean)/scale
    torch.manual_seed(41000)
    head = nn.Sequential(nn.Linear(x.shape[-1], 32), nn.SiLU(), nn.Linear(32, 1)).to(x.device)
    opt = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
    rng = torch.Generator(device=x.device).manual_seed(42000); curve = []
    for step in range(steps):
        d.check_deadline(deadline)
        idx = torch.randint(len(y), (1024,), generator=rng, device=x.device)
        loss = F.binary_cross_entropy_with_logits(head(x[idx])[:, 0], y[idx], pos_weight=weight)
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite probe loss")
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step+1 in (128, 512, 1024): curve.append({"step": step+1, "loss": float(loss.detach())})
    return (head.eval().requires_grad_(False), mean, scale), curve, float(weight)


def metrics(pred, target, threshold, task):
    p = np.asarray(pred, dtype=np.float64); y = np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or p.ndim != 2: raise ValueError("aligned scene-query arrays required")
    ranking = binary_ranking_metrics(p, y)
    safe = np.clip(p, 1e-7, 1-1e-7)
    def score(t):
        yes = p >= t; truth = y > .5
        stats = np.stack(((yes & truth).sum(1), (yes & ~truth).sum(1), (~yes & truth).sum(1)), 1)
        return d.base.score(stats, task)
    return {"fixed05_score": score(.5), "selected_score": score(threshold), "auprc": ranking.auprc,
            "auroc": ranking.auroc, "log_loss_nats": float(-(y*np.log(safe)+(1-y)*np.log(1-safe)).mean()),
            "brier": float(((p-y)**2).mean()), "prevalence": float(y.mean()),
            "positive_rate05": float((p >= .5).mean()), "positive_rate_selected": float((p >= threshold).mean()),
            "probability_quantiles": np.quantile(p, [0, .1, .5, .9, 1]).tolist()}


def run(path, root, encoder_root, output):
    started = time.monotonic(); env = json.loads(path.read_text()); p = env["payload"]
    if set(env) != {"payload", "identity_sha256"} or d.base.payload_sha256(p) != env["identity_sha256"] or p["plan"] != PLAN or sources(root) != p["compact_sources"] or previous.prior.sources(encoder_root) != p["spatial_sources"]: raise ValueError("registration mismatch")
    if d.base.git("diff", p["source_commit"], "--", "theseo_anysearch/garden") or d.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"): raise ValueError("source mismatch")
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    rows = data()
    if previous.identity(rows) != p["data"]: raise ValueError("data mismatch")
    output.mkdir(parents=True, exist_ok=False)
    deadline = started + PLAN["cap_seconds"] - p["preparation_seconds"]
    records = []; extraction = []
    names = list(p["compact_sources"]["states"]) + ["random_grid", "spatial", "raw_neighborhood", "coordinates"]
    for name in names:
        if name in p["compact_sources"]["states"]:
            mode, kind = name.split("-")
            model = CompactEncoder(d.base.make_encoder(0, torch.device("cuda")), 128, mode, joint=kind=="joint").cuda()
            model.load_state_dict(torch.load(root/(name+".pt"), map_location="cuda", weights_only=True))
            if d.base.encoder_state_sha256(model) != p["compact_sources"]["states"][name]: raise ValueError("state mismatch")
        elif name == "random_grid":
            model = CompactEncoder(d.base.make_encoder(381, torch.device("cuda")), 128, "grid").cuda()
        elif name == "spatial":
            model = d.base.make_encoder(0, torch.device("cuda"))
            model.load_state_dict(torch.load(encoder_root/p["spatial_sources"][0]["path"], map_location="cuda", weights_only=True))
            if d.base.encoder_state_sha256(model) != p["spatial_sources"][0]["state_hash"]: raise ValueError("spatial state mismatch")
        else: model = None
        if model is not None: model.eval().requires_grad_(False)
        state = d.base.encoder_state_sha256(model) if model is not None else None
        t = time.monotonic(); cached = features(name, model, rows, deadline)
        extraction.append({"name": name, "seconds": time.monotonic()-t, "encoder_state": state})
        for task in PLAN["tasks"]:
            trainx = cached["probe"][task].cuda(); trainy = rows["probe"]["targets"][task].cuda()
            for recipe, (steps, balanced) in PLAN["recipes"].items():
                t = time.monotonic(); fitted, curve, weight = fit(trainx, trainy, steps, balanced, deadline)
                sp = d.base.predict(fitted, cached["selection"][task].cuda(), task).cpu().numpy()
                threshold = d.threshold(sp, rows["selection"]["targets"][task].numpy())
                dp = d.base.predict(fitted, cached["development"][task].cuda(), task).cpu().numpy()
                trainp = d.base.predict(fitted, trainx, task).cpu().numpy()
                y = rows["development"]["targets"][task].numpy()
                record = {"name": name, "task": task, "recipe": recipe, "threshold": threshold,
                          "input_dim": trainx.shape[-1], "probe_parameters": sum(x.numel() for x in fitted[0].parameters()),
                          "positive_weight": weight, "curve": curve, "fit_and_evaluation_seconds": time.monotonic()-t,
                          "train": metrics(trainp, trainy.cpu().numpy(), threshold, task),
                          "development": metrics(dp, y, threshold, task),
                          "families": {family: metrics(dp[[i for i in range(48) if (i%6)//3==f]], y[[i for i in range(48) if (i%6)//3==f]], threshold, task) for f, family in enumerate(previous.prior.PLAN["families"])}}
                records.append(record)
                torch.save({"state": fitted[0].state_dict(), "mean": fitted[1].cpu(), "scale": fitted[2].cpu(),
                            "selection_predictions": torch.from_numpy(sp), "development_predictions": torch.from_numpy(dp),
                            "train_predictions": torch.from_numpy(trainp)}, output/f"{name}-{task}-{recipe}.pt")
        if model is not None and d.base.encoder_state_sha256(model) != state: raise ValueError("encoder mutated")
        print(json.dumps({"completed": name, "fits": 6}), flush=True)
        del model, cached, fitted, trainx, trainy
    report = {"registration": env, "status": "development_diagnostic_completed", "promotion_eligible": False,
              "records": records, "extraction": extraction, "elapsed_seconds": time.monotonic()-started+p["preparation_seconds"],
              "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
              "artifacts": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in output.glob("*.pt")}}
    report["report_payload_sha256"] = d.base.payload_sha256(report); d.base.write_json(output/"report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    for arg in ("registration", "source", "encoder-source"):
        parser.add_argument("--"+arg, type=Path, required=True)
    parser.add_argument("--spec-sha"); parser.add_argument("--output", type=Path)
    args = parser.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "freeze": freeze(args.registration, args.spec_sha, args.source, args.encoder_source)
    else: run(args.registration, args.source, args.encoder_source, args.output)


if __name__ == "__main__": main()
