"""Development-only compact bottleneck training and profiling, not HPO evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..compact import CompactEncoder, query_features
from . import context_scale as prior

d = prior.d
PLAN = {"study_id": "compact-c0c3-smoke-v2", "seed": 0, "side": 33,
        "splits": ["train", "probe", "development"], "scenes_per_split": 12,
        "queries": 256, "train_steps": 96, "probe_steps": 128, "batch": 4,
        "dimension_train": 128, "dimensions_profile": [64, 128, 192],
        "modes": ["grid", "strided", "attention"], "cap_seconds": 1200,
        "quality_claim": False, "final_assessment": False}


def data():
    result, seen = {}, set()
    for split in PLAN["splits"]:
        values = {k: [] for k in ("occupancy", "hidden")}
        indices, targets = {k: [] for k in d.base.TASKS}, {k: [] for k in d.base.TASKS}
        ids, hashes = [], []
        rngq = torch.Generator().manual_seed(37900 + PLAN["splits"].index(split))
        for i in range(12):
            gid = f"compact-c0c3-smoke-v2-{split}-{i:03d}"
            occ = prior.scene(gid, prior.PLAN["families"][(i % 6) // 3], [.08, .16, .28][i % 3])
            sha = hashlib.sha256(occ.tobytes()).hexdigest()
            if sha in seen:
                raise ValueError("duplicate parent")
            seen.add(sha); ids.append(gid); hashes.append(sha)
            target = d.base.compute_geometry_targets(occ, truncation=8.)
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256((gid + "-mask").encode()).digest()[:8], "little"))
            hidden = rng.random(occ.shape) < .2
            values["occupancy"].append(prior.crop(occ, 33).copy())
            values["hidden"].append(prior.crop(hidden, 33).copy())
            h = torch.tensor(prior.crop(hidden, 17).copy())
            free = ~torch.tensor(prior.crop(occ, 17).copy())
            for task, mask in zip(d.base.TASKS, (h, h, ~h & free, h & free)):
                cells = torch.where(mask.flatten())[0]
                if not len(cells):
                    raise ValueError("empty query pool")
                draw = torch.randperm(len(cells), generator=rngq)[:256] if len(cells) >= 256 else torch.randint(len(cells), (256,), generator=rngq)
                idx = cells[draw]
                truth = occ if task == "occupied_iou" else target.boundary if task == "boundary_f1" else np.maximum(target.signed_distance / 8, 0)
                indices[task].append(idx)
                targets[task].append(torch.tensor(prior.crop(truth, 17).copy(), dtype=torch.float32).flatten()[idx])
        result[split] = {**{k: torch.tensor(np.stack(v), dtype=torch.bool) for k, v in values.items()},
                         "ids": ids, "hashes": hashes, "indices": {k: torch.stack(v) for k, v in indices.items()},
                         "targets": {k: torch.stack(v) for k, v in targets.items()}}
    return result


def identity(rows):
    return {s: {"ids": v["ids"], "parents": v["hashes"],
                "inputs": {k: d.digest_tensor(v[k]) for k in ("occupancy", "hidden")},
                **{k: {t: d.digest_tensor(x) for t, x in v[k].items()} for k in ("indices", "targets")}}
            for s, v in rows.items()}


def freeze(path, spec, source):
    started = time.monotonic()
    if len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec):
        raise ValueError("full spec SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit source first")
    p = {"plan": PLAN, "source_commit": d.base.git("rev-parse", "HEAD"), "spec_commit": spec,
         "spec_path": "projects/theseo-anysearch/python/perception-encoder-compact-smoke.md",
         "sources": prior.sources(source), "data": identity(data()), "preparation_seconds": time.monotonic()-started}
    d.base.write_json(path, {"payload": p, "identity_sha256": d.base.payload_sha256(p)})


def make(source, record, mode, dimension, joint):
    torch.manual_seed(379)
    backbone = d.base.make_encoder(0, torch.device("cuda"))
    backbone.load_state_dict(torch.load(source / record["path"], map_location="cuda", weights_only=True))
    if d.base.encoder_state_sha256(backbone) != record["state_hash"]:
        raise ValueError("source state mismatch")
    return CompactEncoder(backbone, dimension, mode, joint=joint).cuda()


def encode(model, rows, idx):
    occ = rows["occupancy"][idx].to("cuda", dtype=torch.float32)
    hidden = rows["hidden"][idx].cuda().unsqueeze(1)
    return model(d.base.VoxelLevel.from_occupancy(occ, unknown_mask=hidden), hidden)


@torch.no_grad()
def vectors(model, rows):
    return torch.cat([encode(model, rows, slice(i, i+4)).cpu() for i in range(0, 12, 4)])


def run(path, source, output):
    started = time.monotonic()
    env = json.loads(path.read_text()); p = env["payload"]
    if set(env) != {"payload", "identity_sha256"} or d.base.payload_sha256(p) != env["identity_sha256"] or p["plan"] != PLAN or prior.sources(source) != p["sources"]:
        raise ValueError("registration mismatch")
    if d.base.git("diff", p["source_commit"], "--", "theseo_anysearch/garden") or d.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("source mismatch")
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    t = time.monotonic(); rows = data(); data_seconds = time.monotonic()-t
    if identity(rows) != p["data"]:
        raise ValueError("dataset mismatch")
    output.mkdir(parents=True, exist_ok=False)
    deadline = started + PLAN["cap_seconds"] - p["preparation_seconds"]
    profiles, trials = [], []
    for mode in PLAN["modes"]:
        for dim in PLAN["dimensions_profile"]:
            model = make(source, p["sources"][0], mode, dim, False).eval()
            for batch in (1, 4):
                d.check_deadline(deadline)
                with torch.no_grad():
                    for _ in range(3): encode(model, rows["train"], slice(0, batch))
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.monotonic()
                    for _ in range(10): encode(model, rows["train"], slice(0, batch))
                    torch.cuda.synchronize(); elapsed = time.monotonic()-t
                profiles.append({"mode": mode, "dimension": dim, "batch": batch,
                                 "ms_per_batch_including_transfer": elapsed*100,
                                 "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                 "aggregation_parameters": sum(x.numel() for x in model.aggregation.parameters())})
            del model
    print(json.dumps({"profiles_completed": len(profiles)}), flush=True)
    for mode in PLAN["modes"]:
        for joint in (False, True):
            model = make(source, p["sources"][0], mode, 128, joint).train()
            initial = d.base.encoder_state_sha256(model.backbone)
            heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(131, 32), nn.SiLU(), nn.Linear(32, 1)) for k in d.base.TASKS}).cuda()
            opt = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad] + list(heads.parameters()), lr=.001, weight_decay=.01)
            rng = torch.Generator().manual_seed(379); curve = []
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.monotonic()
            for step in range(96):
                d.check_deadline(deadline)
                idx = torch.randint(12, (4,), generator=rng)
                z = encode(model, rows["train"], idx); losses = []
                for task in d.base.TASKS:
                    pred = heads[task](query_features(z, rows["train"]["indices"][task][idx].cuda())).squeeze(-1)
                    y = rows["train"]["targets"][task][idx].cuda()
                    losses.append(F.binary_cross_entropy_with_logits(pred, y) if task in d.base.TASKS[:2] else 10 * F.smooth_l1_loss(pred, y))
                loss = torch.stack(losses).sum()
                if not torch.isfinite(loss): raise FloatingPointError("nonfinite loss")
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                if (step+1) % 24 == 0: curve.append({"step": step+1, "loss": float(loss.detach())})
            torch.cuda.synchronize()
            training = {"seconds": time.monotonic()-t, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "curve": curve}
            if not joint and d.base.encoder_state_sha256(model.backbone) != initial:
                raise ValueError("frozen backbone mutated")
            del opt, heads, loss, losses, z, pred
            model.eval().requires_grad_(False)
            frozen_hash = d.base.encoder_state_sha256(model)
            t = time.monotonic(); z = {s: vectors(model, rows[s]) for s in ("probe", "development")}; extraction_seconds = time.monotonic()-t
            scores = {}; t = time.monotonic()
            for task in d.base.TASKS:
                trainf = query_features(z["probe"].cuda(), rows["probe"]["indices"][task].cuda())
                fitted = d.base.fit_probe(trainf, rows["probe"]["targets"][task].cuda(), task, 0, steps=128)
                scores[task] = {}
                for control, vector in (("trained", z["development"]), ("zeroed", torch.zeros_like(z["development"])), ("shuffled", z["development"].roll(1, 0))):
                    pred = d.base.predict(fitted, query_features(vector.cuda(), rows["development"]["indices"][task].cuda()), task)
                    stats = d.base.sufficient_statistics(pred, rows["development"]["targets"][task].cuda(), task)
                    scores[task][control] = d.base.score(np.asarray(stats), task)
            torch.cuda.synchronize(); probe_seconds = time.monotonic()-t
            if d.base.encoder_state_sha256(model) != frozen_hash: raise ValueError("probe mutated encoder")
            name = f"{mode}-{'joint' if joint else 'frozen'}"
            torch.save(model.state_dict(), output / f"{name}.pt")
            trials.append({"name": name, "training": training, "extraction_seconds": extraction_seconds,
                           "probe_seconds": probe_seconds, "scores_development_only": scores,
                           "encoder_state_sha256": frozen_hash})
            print(json.dumps({"completed": name, "train_seconds": training["seconds"], "probe_seconds": probe_seconds}), flush=True)
            del model, fitted, trainf, z, pred
    report = {"registration": env, "status": "engineering_smoke_completed", "promotion_eligible": False,
              "profiles": profiles, "trials": trials, "data_seconds": data_seconds,
              "elapsed_seconds": time.monotonic()-started+p["preparation_seconds"],
              "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
              "artifacts": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in output.glob("*.pt")}}
    report["report_payload_sha256"] = d.base.payload_sha256(report)
    d.base.write_json(output / "report.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spec-sha")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(); os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "freeze": freeze(args.registration, args.spec_sha, args.source)
    else: run(args.registration, args.source, args.output)


if __name__ == "__main__": main()
