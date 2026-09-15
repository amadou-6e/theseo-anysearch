"""Frozen tiled33 versus dense65, wider spatial queries and fixed heads."""
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

from . import context_65 as previous

d = previous.d
PLAN = {"study_id": "voxel-context-tiles-v1", "run_id": "voxel-context-tiles-v1-run1",
        "scenes": 48, "queries": 256, "seeds": [0, 1, 2], "cap_seconds": 3600,
        "tile_batch": 4, "peak_cap_bytes": 6 * 1024**3,
        "arms": ["tiled33_head33", "dense65_head33", "dense65_head65"]}
STARTS = list(itertools.product((0, 16, 32), repeat=3))


def query_mapping(indices):
    xyz = torch.stack((indices // 49**2, indices // 49 % 49, indices % 49), -1)
    tile = (xyz // 16).clamp_max(2)
    local = xyz + 8 - tile * 16
    return tile[..., 0] * 9 + tile[..., 1] * 3 + tile[..., 2], local[..., 0] * 33**2 + local[..., 1] * 33 + local[..., 2]


def data():
    inputs = {k: [] for k in ("occupancy", "hidden")}
    indices, targets = {k: [] for k in d.base.TASKS}, {k: [] for k in d.base.TASKS}
    ids, hashes = [], []
    rngq = torch.Generator().manual_seed(37600)
    for i in range(48):
        gid = f"voxel-context-tiles-v1-data-assessment-{i:03d}"
        occ = previous.scene(gid, previous.PLAN["families"][(i % 6) // 3], [.08, .16, .28][i % 3])
        sha = hashlib.sha256(occ.tobytes()).hexdigest()
        if sha in hashes:
            raise ValueError("duplicate parent")
        ids.append(gid)
        hashes.append(sha)
        target = d.base.compute_geometry_targets(occ, truncation=8.)
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256((gid + "-mask").encode()).digest()[:8], "little"))
        hidden = rng.random(occ.shape) < .2
        inputs["occupancy"].append(previous.prior.crop(occ, 65).copy())
        inputs["hidden"].append(previous.prior.crop(hidden, 65).copy())
        h = torch.tensor(previous.prior.crop(hidden, 49).copy())
        free = ~torch.tensor(previous.prior.crop(occ, 49).copy())
        masks = dict(zip(d.base.TASKS, (h, h, ~h & free, h & free)))
        for task in d.base.TASKS:
            cells = torch.where(masks[task].flatten())[0]
            if not len(cells):
                raise ValueError("empty query pool")
            draw = torch.randperm(len(cells), generator=rngq)[:256] if len(cells) >= 256 else torch.randint(len(cells), (256,), generator=rngq)
            idx = cells[draw]
            truth = occ if task == "occupied_iou" else target.boundary if task == "boundary_f1" else np.maximum(target.signed_distance / 8, 0)
            indices[task].append(idx)
            targets[task].append(torch.tensor(previous.prior.crop(truth, 49).copy(), dtype=torch.float32).flatten()[idx])
    rows = {**{k: torch.tensor(np.stack(v), dtype=torch.bool) for k, v in inputs.items()},
            "ids": ids, "hashes": hashes, "indices": {k: torch.stack(v) for k, v in indices.items()},
            "targets": {k: torch.stack(v) for k, v in targets.items()}}
    for task in d.base.TASKS[:2]:
        for family in range(2):
            y = rows["targets"][task][[i for i in range(48) if (i % 6) // 3 == family]]
            if min(float(y.sum()), float((1 - y).sum())) < 20:
                raise ValueError("insufficient family classes")
    return rows


def sources(encoder_root, probe_root):
    report = json.loads(Path("docs/perception-encoder-local-geometry/context65-report.json").read_text())
    sha = report.pop("report_payload_sha256")
    if d.base.payload_sha256(report) != sha or report["status"] != "completed":
        raise ValueError("invalid source report")
    for name, digest in report["artifacts"].items():
        if hashlib.sha256((probe_root / name).read_bytes()).hexdigest() != digest:
            raise ValueError("probe artifact mismatch")
    return {"encoders": previous.prior.sources(encoder_root), "probe_report_sha256": sha,
            "probes": report["artifacts"], "thresholds": {f"{t['arm']}-{t['seed']}-{t['task']}": t["threshold"] for t in report["trials"]}}


def freeze(path, spec, encoder_root, probe_root):
    started = time.monotonic()
    if not isinstance(spec, str) or len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec):
        raise ValueError("full spec SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit source first")
    payload = {"plan": PLAN, "source_commit": d.base.git("rev-parse", "HEAD"), "spec_commit": spec,
               "spec_path": "projects/theseo-anysearch/python/perception-context-tiles.md",
               "sources": sources(encoder_root, probe_root), "identity": previous.identity({"assessment": data()}),
               "preparation_seconds": time.monotonic() - started}
    d.base.write_json(path, {"payload": payload, "identity_sha256": d.base.payload_sha256(payload)})


def tile_inputs(occ, hidden):
    return (torch.stack([x[a:a+33, b:b+33, c:c+33] for a, b, c in STARTS]) for x in (occ, hidden))


@torch.no_grad()
def forward(model, occ, hidden):
    mask = hidden.unsqueeze(1)
    return model(d.base.VoxelLevel.from_occupancy(occ, unknown_mask=mask), mask).local_feature_volume


@torch.no_grad()
def profile(model, rows, tiled):
    occ = rows["occupancy"][0].to("cuda", dtype=torch.float32)
    hidden = rows["hidden"][0].cuda()
    if tiled:
        occ, hidden = tile_inputs(occ, hidden)
        batch = 4
    else:
        occ, hidden, batch = occ.unsqueeze(0), hidden.unsqueeze(0), 1

    def cover():
        for start in range(0, len(occ), batch):
            forward(model, occ[start:start+batch], hidden[start:start+batch])

    for _ in range(3):
        cover()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    for _ in range(10):
        cover()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return {"ms_per_scene": (time.monotonic() - started) * 100,
            "peak_allocated_bytes": peak, "incremental_peak_bytes": peak - baseline}


@torch.no_grad()
def features(model, rows, index, tiled):
    occ = rows["occupancy"][index].to("cuda", dtype=torch.float32)
    hidden = rows["hidden"][index].cuda()
    if not tiled:
        volume = forward(model, occ.unsqueeze(0), hidden.unsqueeze(0))
        result = {}
        for task in d.base.TASKS:
            idx = rows["indices"][task][index].cuda()
            dense = (idx // 49**2 + 8) * 65**2 + (idx // 49 % 49 + 8) * 65 + idx % 49 + 8
            result[task] = d.base.gather_features(volume, dense.unsqueeze(0)).cpu()
        return result
    occ, hidden = tile_inputs(occ, hidden)
    result = {k: torch.empty(1, 256, 8) for k in d.base.TASKS}
    mappings = {k: query_mapping(rows["indices"][k][index]) for k in d.base.TASKS}
    for start in range(0, 27, 4):
        volume = forward(model, occ[start:start+4], hidden[start:start+4])
        for task, (tiles, local) in mappings.items():
            for tile in range(start, min(start + 4, 27)):
                chosen = tiles == tile
                if chosen.any():
                    values = d.base.gather_features(volume[tile-start:tile-start+1], local[chosen].unsqueeze(0).cuda())
                    result[task][:, chosen] = values.cpu()
    return result


def load_probe(root, seed, side, task):
    saved = torch.load(root / f"frozen{side}-{seed}-{task}.pt", map_location="cuda", weights_only=True)
    model = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 1)).cuda()
    model.load_state_dict(saved["state"])
    return model.eval().requires_grad_(False), saved["mean"], saved["scale"]


def run(path, encoder_root, probe_root, output):
    started = time.monotonic()
    env = json.loads(path.read_text())
    p = env["payload"]
    if set(env) != {"payload", "identity_sha256"} or d.base.payload_sha256(p) != env["identity_sha256"] or p["plan"] != PLAN or sources(encoder_root, probe_root) != p["sources"]:
        raise ValueError("registration mismatch")
    if d.base.git("diff", p["source_commit"], "--", "theseo_anysearch/garden") or d.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("source mismatch")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = data()
    if previous.identity({"assessment": rows}) != p["identity"]:
        raise ValueError("data mismatch")
    output.mkdir(parents=True, exist_ok=False)
    deadline = started + PLAN["cap_seconds"] - p["preparation_seconds"]
    resources, trials = [], []
    status = "completed"
    try:
        for seed in PLAN["seeds"]:
            model = previous.load_encoder(seed, encoder_root, p["sources"]["encoders"])
            for tiled in (False, True):
                d.check_deadline(deadline)
                record = {"seed": seed, "mode": "tiled33" if tiled else "dense65", **profile(model, rows, tiled)}
                resources.append(record)
                if record["peak_allocated_bytes"] > PLAN["peak_cap_bytes"]:
                    status = "resource_limited"
            del model
        print(json.dumps({"profiles": resources, "status": status}), flush=True)
        if status == "completed":
            for seed in PLAN["seeds"]:
                model = previous.load_encoder(seed, encoder_root, p["sources"]["encoders"])
                feats = {mode: {task: [] for task in d.base.TASKS} for mode in ("dense65", "tiled33")}
                for index in range(48):
                    d.check_deadline(deadline)
                    for mode in feats:
                        values = features(model, rows, index, mode == "tiled33")
                        for task, value in values.items():
                            feats[mode][task].append(value)
                for arm in PLAN["arms"]:
                    mode, head = arm.split("_head")
                    for task in d.base.TASKS:
                        fitted = load_probe(probe_root, seed, head, task)
                        pred = d.base.predict(fitted, torch.cat(feats[mode][task]).cuda(), task)
                        threshold = p["sources"]["thresholds"][f"frozen{head}-{seed}-{task}"]
                        scored = (pred >= threshold).float() if task in d.base.TASKS[:2] else pred
                        stats = d.base.sufficient_statistics(scored, rows["targets"][task].cuda(), task)
                        families = {family: d.base.score(np.asarray(stats)[[i for i in range(48) if (i % 6) // 3 == f]], task) for f, family in enumerate(previous.PLAN["families"])}
                        trials.append({"seed": seed, "arm": arm, "task": task, "threshold": threshold, "statistics": stats, "families": families})
                        torch.save(pred.cpu(), output / f"{arm}-{seed}-{task}.pt")
                if d.base.encoder_state_sha256(model) != p["sources"]["encoders"][seed]["state_hash"]:
                    raise ValueError("encoder mutated")
                del model, feats, fitted, pred, scored
                print(json.dumps({"completed_seed": seed}), flush=True)
    except torch.cuda.OutOfMemoryError:
        status = "resource_limited"
    comparisons = {a + "_vs_" + b: previous.prior.compare(trials, a, b) for a, b in
                   (("tiled33_head33", "dense65_head33"), ("tiled33_head33", "dense65_head65"), ("dense65_head33", "dense65_head65"))} if status == "completed" else {}
    report = {"registration": env, "status": status, "resources": resources, "trials": trials,
              "comparisons": comparisons, "elapsed_seconds": time.monotonic() - started + p["preparation_seconds"],
              "promotion_eligible": False, "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
              "artifacts": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in output.glob("*.pt")}}
    report["report_payload_sha256"] = d.base.payload_sha256(report)
    d.base.write_json(output / "report.json", report)
    print(json.dumps(comparisons), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    for name in ("registration", "encoder-root", "probe-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--spec-sha")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "freeze":
        freeze(args.registration, args.spec_sha, args.encoder_root, args.probe_root)
    else:
        run(args.registration, args.encoder_root, args.probe_root, args.output)


if __name__ == "__main__":
    main()
