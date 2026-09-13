"""Memory-first paired 33/65 context evaluation with frozen encoders."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from . import context_scale as prior

d = prior.d
PLAN = {
    "study_id": "voxel-context-65-v1", "run_id": "voxel-context-65-v1-run1",
    "sides": [33, 65], "parent_side": 81,
    "counts": {"train": 48, "selection": 24, "assessment": 48},
    "seeds": [0, 1, 2], "probe_steps": 1024, "queries": 256,
    "families": ["random_field", "oblique_sheets"], "cap_seconds": 7200,
    "batch_candidates": [4, 1], "peak_cap_bytes": 6 * 1024**3,
    "bootstrap_seed": 372,
}


def scene(gid, family, fraction):
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(gid.encode()).digest()[:8], "little"))
    if family == "random_field":
        field = gaussian_filter(rng.normal(size=(81, 81, 81)), sigma=1.5, mode="reflect")
    elif family == "oblique_sheets":
        xyz = np.stack(np.meshgrid(*([np.arange(-40, 41) / 8] * 3), indexing="ij"))
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        field = np.abs(np.sin(rng.uniform(4, 8) * (xyz * direction[:, None, None, None]).sum(0) + rng.uniform(-np.pi, np.pi)))
    else:
        raise ValueError("unknown family")
    return field <= np.quantile(field, fraction)


def data():
    result, seen = {}, set()
    for split, count in PLAN["counts"].items():
        inputs = {k: [] for k in ("occupancy", "hidden")}
        truths = {k: [] for k in ("occupancy", "boundary", "distance")}
        ids, hashes = [], []
        for i in range(count):
            gid = f"voxel-context-65-v1-data-{split}-{i:03d}"
            occ = scene(gid, PLAN["families"][(i % 6) // 3], [.08, .16, .28][i % 3])
            sha = hashlib.sha256(occ.tobytes()).hexdigest()
            if sha in seen:
                raise ValueError("duplicate parent")
            seen.add(sha)
            ids.append(gid)
            hashes.append(sha)
            target = d.base.compute_geometry_targets(occ, truncation=8.)
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256((gid + "-mask").encode()).digest()[:8], "little"))
            inputs["occupancy"].append(prior.crop(occ, 65).copy())
            inputs["hidden"].append(prior.crop(rng.random(occ.shape) < .2, 65).copy())
            for key, value in (("occupancy", occ), ("boundary", target.boundary), ("distance", target.signed_distance / 8)):
                truths[key].append(prior.crop(value, 17).copy())
        tensors = {k: torch.tensor(np.stack(v), dtype=torch.bool) for k, v in inputs.items()}
        truth = {k: torch.tensor(np.stack(v), dtype=torch.float32) for k, v in truths.items()}
        hidden = prior.crop(tensors["hidden"], 17)
        free = truth["occupancy"] < .5
        masks = dict(zip(d.base.TASKS, (hidden, hidden, ~hidden & free, hidden & free)))
        indices, targets = {}, {}
        rng = torch.Generator().manual_seed(37400 + list(PLAN["counts"]).index(split))
        for task in d.base.TASKS:
            choices = []
            for mask in masks[task]:
                cells = torch.where(mask.flatten())[0]
                if not len(cells):
                    raise ValueError("empty central query pool")
                draw = torch.randperm(len(cells), generator=rng)[:256] if len(cells) >= 256 else torch.randint(len(cells), (256,), generator=rng)
                choices.append(cells[draw])
            indices[task] = torch.stack(choices)
            y = truth["occupancy"] if task == "occupied_iou" else truth["boundary"] if task == "boundary_f1" else truth["distance"].clamp_min(0)
            targets[task] = y.flatten(1).gather(1, indices[task])
            if task in d.base.TASKS[:2]:
                for family in range(2):
                    labels = targets[task][[i for i in range(count) if (i % 6) // 3 == family]]
                    if min(float(labels.sum()), float((1 - labels).sum())) < 20:
                        raise ValueError("insufficient family labels")
        result[split] = {**tensors, "ids": ids, "hashes": hashes, "indices": indices, "targets": targets}
    return result


def identity(rows):
    return {s: {"ids": v["ids"], "hashes": v["hashes"],
                "arrays": {k: d.digest_tensor(v[k]) for k in ("occupancy", "hidden")},
                **{k: {t: d.digest_tensor(x) for t, x in v[k].items()} for k in ("indices", "targets")}}
            for s, v in rows.items()}


def freeze(path, spec, source):
    start = time.monotonic()
    if not isinstance(spec, str) or len(spec) != 40 or any(c not in "0123456789abcdef" for c in spec):
        raise ValueError("full spec SHA required")
    if d.base.git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit source first")
    payload = {"plan": PLAN, "source_commit": d.base.git("rev-parse", "HEAD"), "spec_commit": spec,
               "spec_path": "projects/theseo-anysearch/python/perception-context-65.md",
               "encoders": prior.sources(source), "identity": identity(data()),
               "preparation_seconds": time.monotonic() - start}
    d.base.write_json(path, {"payload": payload, "identity_sha256": d.base.payload_sha256(payload)})


def load_encoder(seed, source, records):
    model = d.base.make_encoder(seed, torch.device("cuda"))
    model.load_state_dict(torch.load(source / records[seed]["path"], map_location="cuda", weights_only=True))
    model.eval().requires_grad_(False)
    if d.base.encoder_state_sha256(model) != records[seed]["state_hash"]:
        raise ValueError("encoder state mismatch")
    return model


def batch_input(rows, side, start, batch):
    occ = prior.crop(rows["occupancy"][start:start + batch], side).to(device="cuda", dtype=torch.float32)
    mask = prior.crop(rows["hidden"][start:start + batch], side).cuda().unsqueeze(1)
    return d.base.VoxelLevel.from_occupancy(occ, unknown_mask=mask), mask


@torch.no_grad()
def profile(model, rows, side, batch):
    level, mask = batch_input(rows, side, 0, batch)
    for _ in range(3):
        model(level, mask)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    for _ in range(20):
        model(level, mask)
    torch.cuda.synchronize()
    seconds = time.monotonic() - start
    peak = torch.cuda.max_memory_allocated()
    return {"batch": batch, "iterations": 20, "ms_per_batch": seconds * 50,
            "crops_per_second": batch * 20 / seconds,
            "peak_allocated_bytes": peak, "incremental_peak_bytes": peak - baseline}


def select_batch(measure):
    attempts = []
    for batch in PLAN["batch_candidates"]:
        try:
            resources = measure(batch)
            accepted = all(r["inference"]["peak_allocated_bytes"] <= PLAN["peak_cap_bytes"] for r in resources)
            attempts.append({"batch": batch, "accepted": accepted, "resources": resources})
            if accepted:
                return batch, attempts
        except torch.cuda.OutOfMemoryError:
            attempts.append({"batch": batch, "accepted": False, "reason": "cuda_oom"})
        gc.collect()
        torch.cuda.empty_cache()
    return None, attempts


@torch.no_grad()
def features(model, rows, side, batch):
    result = {task: [] for task in d.base.TASKS}
    for start in range(0, len(rows["occupancy"]), batch):
        level, mask = batch_input(rows, side, start, batch)
        volume = model(level, mask).local_feature_volume
        for task in result:
            indices = prior.mapped_indices(rows["indices"][task][start:start + batch], side).cuda()
            result[task].append(d.base.gather_features(volume, indices).cpu())
    return {k: torch.cat(v) for k, v in result.items()}


def run(path, source, output):
    started = time.monotonic()
    env = json.loads(path.read_text())
    p = env["payload"]
    if set(env) != {"payload", "identity_sha256"} or d.base.payload_sha256(p) != env["identity_sha256"] or p["plan"] != PLAN or prior.sources(source) != p["encoders"]:
        raise ValueError("registration mismatch")
    if d.base.git("diff", p["source_commit"], "--", "theseo_anysearch/garden") or d.base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("source mismatch")
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = data()
    if identity(rows) != p["identity"]:
        raise ValueError("data mismatch")
    output.mkdir(parents=True, exist_ok=False)
    deadline = started + PLAN["cap_seconds"] - p["preparation_seconds"]

    def measure(batch):
        resources = []
        for seed in PLAN["seeds"]:
            model = load_encoder(seed, source, p["encoders"])
            try:
                for side in PLAN["sides"]:
                    d.check_deadline(deadline)
                    resources.append({"seed": seed, "arm": f"frozen{side}", "inference": profile(model, rows["assessment"], side, batch)})
            finally:
                del model
        return resources

    batch, attempts = select_batch(measure)
    print(json.dumps({"profile_complete": True, "batch": batch, "attempts": attempts}), flush=True)
    trials = []
    if batch is not None:
        for seed in PLAN["seeds"]:
            model = load_encoder(seed, source, p["encoders"])
            for side in PLAN["sides"]:
                d.check_deadline(deadline)
                arm = f"frozen{side}"
                feats = {s: features(model, v, side, batch) for s, v in rows.items()}
                for task in d.base.TASKS:
                    fitted = d.base.fit_probe(feats["train"][task].cuda(), rows["train"]["targets"][task].cuda(), task, seed, steps=PLAN["probe_steps"])
                    pred = d.base.predict(fitted, feats["assessment"][task].cuda(), task)
                    threshold = .5
                    if task in d.base.TASKS[:2]:
                        sp = d.base.predict(fitted, feats["selection"][task].cuda(), task).cpu().numpy()
                        threshold = d.threshold(sp, rows["selection"]["targets"][task].numpy())
                    scored = (pred >= threshold).float() if task in d.base.TASKS[:2] else pred
                    stats = d.base.sufficient_statistics(scored, rows["assessment"]["targets"][task].cuda(), task)
                    families = {family: d.base.score(np.asarray(stats)[[i for i in range(48) if (i % 6) // 3 == f]], task) for f, family in enumerate(PLAN["families"])}
                    trials.append({"seed": seed, "arm": arm, "task": task, "threshold": threshold, "statistics": stats, "families": families})
                    torch.save({"predictions": pred.cpu(), "state": fitted[0].state_dict(), "mean": fitted[1].cpu(), "scale": fitted[2].cpu()}, output / f"{arm}-{seed}-{task}.pt")
                    d.check_deadline(deadline)
                if d.base.encoder_state_sha256(model) != p["encoders"][seed]["state_hash"]:
                    raise ValueError("frozen encoder mutated")
                print(json.dumps({"completed": arm, "seed": seed}), flush=True)
                del feats, fitted, pred, scored
            del model
    comparisons = prior.compare(trials, "frozen65", "frozen33") if batch is not None else {}
    report = {"registration": env, "status": "completed" if batch is not None else "resource_limited",
              "trials": trials, "batch": batch, "profile_attempts": attempts, "comparisons": comparisons,
              "elapsed_seconds": time.monotonic() - started + p["preparation_seconds"], "promotion_eligible": False,
              "environment": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
              "artifacts": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in output.glob("*.pt")}}
    report["report_payload_sha256"] = d.base.payload_sha256(report)
    d.base.write_json(output / "report.json", report)
    print(json.dumps(comparisons), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["freeze", "run"])
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--spec-sha")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.command == "freeze":
        freeze(args.registration, args.spec_sha, args.source)
    else:
        run(args.registration, args.source, args.output)


if __name__ == "__main__":
    main()
