"""Frozen D0-D2 boundary diagnostics. No encoder updates or promotion."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from sklearn.metrics import average_precision_score
import torch
from torch import nn
from torch.nn import functional as F

from . import local_geometry_confirmation as lg3
from .diagnostic_routing import DiagnosticEvidence, Outcome, route

base = lg3.base
PLAN = {
    "study_id": "voxel-encoder-diagnostics-v1",
    "dataset_id": "voxel-encoder-diagnostics-v1-development-1",
    "query_id": "voxel-encoder-diagnostics-v1-queries-1",
    "splits": {"train": 48, "selection": 24, "assessment": 48},
    "streams": {"train": 36001, "selection": 36002, "assessment": 36003},
    "seeds": [0, 1, 2], "mask_probability": .2, "queries": 256,
    "reference_width": 16, "reference_dilations": [1, 2, 4, 1],
    "probe_widths": [0, 32, 128], "reference_steps": 2048,
    "probe_steps": 1024, "d0_steps": 512, "lr": .001,
    "weight_decay": .01, "reference_batch": 8, "probe_batch": 1024,
    "stage_seconds": {"D0": 300, "D1": 2400, "D2": 2400},
    "campaign_seconds": 7200, "thresholds": [i / 100 for i in range(1, 100)],
    "bootstrap_draws": 2000, "bootstrap_seed": 358, "boundary_bar": .7,
    "source_report_sha256": lg3.PLAN["source_report_sha256"],
    "ordering_exception": "User authorizes execution stacked on unmerged LG3 and diagnostic preparation; no integration merges.",
    "scope": "boundary diagnosis on fresh development instances, not confirmation",
}


def digest_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def corpus() -> dict:
    result, seen = {}, set()
    for split, count in PLAN["splits"].items():
        ids, occ, boundary, hashes = [], [], [], []
        for i in range(count):
            gid = f"{PLAN['dataset_id']}-{split}-{i:03d}"
            grid = lg3.generate(gid, lg3.FAMILIES[(i % 12)//3], [.08, .16, .28][i % 3])
            sha = hashlib.sha256(grid.tobytes()).hexdigest()
            if sha in seen:
                raise ValueError("duplicate geometry; no redraw")
            seen.add(sha)
            ids.append(gid)
            hashes.append(sha)
            occ.append(grid)
            boundary.append(base.compute_geometry_targets(grid, truncation=8.).boundary)
        occupancy = torch.tensor(np.stack(occ), dtype=torch.float32)
        truth = torch.tensor(np.stack(boundary), dtype=torch.float32)
        rng = torch.Generator().manual_seed(PLAN["streams"][split])
        hidden = torch.rand(occupancy.shape, generator=rng) < PLAN["mask_probability"]
        indices = []
        for mask in hidden:
            eligible = torch.where(mask.flatten())[0]
            if len(eligible) < PLAN["queries"]:
                raise ValueError("insufficient hidden query support")
            indices.append(eligible[torch.randperm(len(eligible), generator=rng)[:PLAN["queries"]]])
        indices = torch.stack(indices)
        labels = truth.flatten(1).gather(1, indices)
        for family in range(4):
            y = labels[[i for i in range(count) if (i % 12)//3 == family]]
            if min(float(y.sum()), float((1-y).sum())) < 20:
                raise ValueError("insufficient family class support")
        result[split] = {"ids": ids, "hashes": hashes, "occupancy": occupancy,
                         "hidden": hidden.unsqueeze(1), "indices": indices, "labels": labels}
    return result


def data_contract(data: dict) -> dict:
    return {s: {"ids": d["ids"], "occupancy_hashes": d["hashes"],
                **{k: digest_tensor(d[k]) for k in ("hidden", "indices", "labels")}}
            for s, d in data.items()}


def freeze(path: Path, spec_sha: str, artifacts: Path, report: Path) -> None:
    started = time.monotonic()
    if len(spec_sha) != 40 or any(c not in "0123456789abcdef" for c in spec_sha):
        raise ValueError("full spec SHA required")
    if base.git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit source before freezing")
    contract = lg3.artifact_contract(report)
    files = {k: v for k, v in contract["files"].items() if k.endswith("encoder.pt")}
    for name, sha in files.items():
        if hashlib.sha256((artifacts / name).read_bytes()).hexdigest() != sha:
            raise ValueError("checkpoint mismatch")
    payload = {"plan": PLAN, "source_commit": base.git("rev-parse", "HEAD"),
               "spec_commit": spec_sha,
               "spec_path": "projects/theseo-anysearch/python/perception-encoder-diagnostics.md",
               "files": files, "states": contract["states"], "data": data_contract(corpus()),
               "preparation_seconds": time.monotonic() - started}
    base.write_json(path, {"payload": payload, "identity_sha256": base.payload_sha256(payload)})


def read_registration(path: Path, artifacts: Path) -> dict:
    env = json.loads(path.read_text())
    p = env["payload"]
    if set(env) != {"payload", "identity_sha256"} or base.payload_sha256(p) != env["identity_sha256"] or p["plan"] != PLAN:
        raise ValueError("registration mismatch")
    if base.git("diff", p["source_commit"], "--", "theseo_anysearch/garden") or base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("executable source mismatch")
    for name, sha in p["files"].items():
        if hashlib.sha256((artifacts / name).read_bytes()).hexdigest() != sha:
            raise ValueError("checkpoint mismatch")
    return env


def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise TimeoutError("registered compute cap exhausted")


def f1(p: np.ndarray, y: np.ndarray, threshold: float) -> float:
    pred, truth = p >= threshold, y > .5
    tp = int((pred & truth).sum())
    denom = int(pred.sum() + truth.sum())
    return 2 * tp / denom if denom else 0.


def logloss_rows(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-7, 1-1e-7)
    return -(y*np.log(p) + (1-y)*np.log1p(-p)).mean(axis=1)


def threshold(p: np.ndarray, y: np.ndarray) -> float:
    # First maximum makes ties deterministic without inspecting assessment labels.
    return max(PLAN["thresholds"], key=lambda t: f1(p, y, t))


def metrics(p: np.ndarray, y: np.ndarray, t: float) -> dict:
    return {"log_loss": float(logloss_rows(p, y).mean()),
            "auprc": float(average_precision_score(y.ravel(), p.ravel())),
            "f1_05": f1(p, y, .5), "f1_calibrated": f1(p, y, t),
            "threshold": t, "positive_prevalence": float(y.mean()),
            "predicted_positive_count": int((p >= t).sum()),
            "target_positive_count": int(y.sum())}


def evaluate(p: np.ndarray, y: np.ndarray, t: float) -> dict:
    return {**metrics(p, y, t), "geometry_log_loss": logloss_rows(p, y).tolist(),
            "families": {family: metrics(p[idx], y[idx], t)
                         for f, family in enumerate(lg3.FAMILIES)
                         for idx in [[i for i in range(len(y)) if (i % 12)//3 == f]]}}


def classify(rows: list, null_rows: np.ndarray) -> dict:
    if sorted(r["seed"] for r in rows) != PLAN["seeds"]:
        return {"outcome": "inconclusive", "reason": "incomplete seeds"}
    delta = np.mean([null_rows - np.asarray(r["assessment"]["geometry_log_loss"]) for r in rows], axis=0)
    rng = np.random.default_rng(PLAN["bootstrap_seed"])
    strata = [np.arange(s, len(delta), 12) for s in range(12)]
    samples = [float(delta[np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])].mean())
               for _ in range(PLAN["bootstrap_draws"])]
    lo, hi = np.quantile(samples, [.025, .975]).tolist()
    scores = [r["assessment"]["f1_calibrated"] for r in rows]
    outcome = "pass" if min(scores) >= PLAN["boundary_bar"] and lo > 0 else (
        "fail" if max(scores) < PLAN["boundary_bar"] or hi <= 0 else "inconclusive")
    return {"outcome": outcome, "seed_f1": scores, "null_log_loss_gain_ci95": [lo, hi]}


class Reference(nn.Module):
    def __init__(self):
        super().__init__()
        layers, channels = [], 3
        for dilation in PLAN["reference_dilations"]:
            layers.extend([nn.Conv3d(channels, 16, 3, padding=dilation, dilation=dilation),
                           nn.GroupNorm(4, 16), nn.SiLU()])
            channels = 16
        self.net = nn.Sequential(*layers, nn.Conv3d(16, 1, 1))

    def forward(self, x):
        return self.net(x)


def visible(d: dict) -> torch.Tensor:
    # Neither hidden occupancy nor derived boundary/distance truth enters the model.
    return base.VoxelLevel.from_occupancy(d["occupancy"], unknown_mask=d["hidden"]).features[:, :3]


@torch.no_grad()
def ref_predict(model: nn.Module, d: dict) -> np.ndarray:
    rows = []
    for start in range(0, len(d["labels"]), 8):
        x = visible({k: v[start:start+8] for k, v in d.items() if isinstance(v, torch.Tensor)})
        rows.append(model(x).flatten(1).gather(1, d["indices"][start:start+8]).sigmoid())
    return torch.cat(rows).cpu().numpy()


def fit_reference(data: dict, seed: int, deadline: float) -> tuple:
    check_deadline(deadline)
    torch.manual_seed(36010 + seed)
    model = Reference().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=PLAN["lr"], weight_decay=PLAN["weight_decay"])
    train = data["train"]
    inputs = visible(train)
    rng = torch.Generator(device="cuda").manual_seed(36020 + seed)
    curve = []
    for step in range(PLAN["reference_steps"]):
        check_deadline(deadline)
        idx = torch.randint(len(inputs), (8,), generator=rng, device="cuda")
        logits = model(inputs[idx]).flatten(1).gather(1, train["indices"][idx])
        loss = F.binary_cross_entropy_with_logits(logits, train["labels"][idx])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite reference loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 256 == 0 or step + 1 == PLAN["reference_steps"]:
            p = ref_predict(model, data["selection"])
            curve.append({"step": step+1, "train_loss": float(loss.detach()),
                          "selection_loss": float(logloss_rows(p, data["selection"]["labels"].cpu().numpy()).mean())})
            print(json.dumps({"D1": seed, **curve[-1]}), flush=True)
    check_deadline(deadline)
    return model.eval(), curve


def fit_probe(features: dict, data: dict, seed: int, width: int, deadline: float) -> tuple:
    check_deadline(deadline)
    torch.manual_seed(36030 + seed)
    x = features["train"].flatten(0, 1)
    mean, scale = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-5)
    x = (x-mean)/scale
    y = data["train"]["labels"].flatten()
    model = (nn.Linear(8, 1) if width == 0 else nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, 1))).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=PLAN["lr"], weight_decay=PLAN["weight_decay"])
    rng = torch.Generator(device="cuda").manual_seed(36040 + seed)
    curve = []
    for step in range(PLAN["probe_steps"]):
        check_deadline(deadline)
        idx = torch.randint(len(y), (1024,), generator=rng, device="cuda")
        loss = F.binary_cross_entropy_with_logits(model(x[idx])[:, 0], y[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite probe loss")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 256 == 0 or step+1 == PLAN["probe_steps"]:
            with torch.no_grad():
                sel = model((features["selection"]-mean)/scale)[..., 0]
                curve.append({"step": step+1, "train_loss": float(loss.detach()),
                              "selection_loss": float(F.binary_cross_entropy_with_logits(sel, data["selection"]["labels"]))})
    check_deadline(deadline)
    return model.eval(), mean, scale, curve


def fixture_checks() -> None:
    empty, solid = np.zeros((17,)*3, bool), np.ones((17,)*3, bool)
    interface = empty.copy()
    interface[8:] = True
    for grid in (empty, solid, interface):
        truth = base.compute_geometry_targets(grid, truncation=8.).boundary
        expected = np.zeros_like(grid)
        for axis in range(3):
            a, b = [slice(None)]*3, [slice(None)]*3
            a[axis], b[axis] = slice(None, -1), slice(1, None)
            diff = grid[tuple(a)] != grid[tuple(b)]
            expected[tuple(a)] |= diff
            expected[tuple(b)] |= diff
        expected &= grid
        if not np.array_equal(truth, expected):
            raise ValueError("independent boundary fixture mismatch")
    assert f1(np.array([.9, .1]), np.array([1, 0]), .5) == 1
    assert f1(np.array([.1, .9]), np.array([1, 0]), .5) == 0


def d0(deadline: float) -> dict:
    fixture_checks()
    scores = []
    y = torch.tensor([0., 1.]*32, device="cuda")
    x = (2*y-1).unsqueeze(1)
    for seed in PLAN["seeds"]:
        torch.manual_seed(36050 + seed)
        model = nn.Linear(1, 1).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.01)
        for _ in range(PLAN["d0_steps"]):
            check_deadline(deadline)
            loss = F.binary_cross_entropy_with_logits(model(x)[:, 0], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        scores.append(f1(model(x)[:, 0].sigmoid().detach().cpu().numpy(), y.cpu().numpy(), .5))
    return {"outcome": "pass" if min(scores) >= .99 else "fail", "seed_f1": scores}


def run(registration: Path, artifacts: Path, output: Path) -> dict:
    started = time.monotonic()
    frozen = read_registration(registration, artifacts)
    output.mkdir(parents=True, exist_ok=False)
    deadline = started + PLAN["campaign_seconds"] - frozen["payload"]["preparation_seconds"]
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    report = {"registration": frozen, "D1": [], "D2": [], "encoder_updates": 0}
    statuses = {k: Outcome.PENDING for k in ("d0", "d1", "d2")}
    stage = "d0"
    try:
        data = corpus()
        if data_contract(data) != frozen["payload"]["data"]:
            raise ValueError("data/query identity mismatch")
        data = {s: {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in d.items()} for s, d in data.items()}
        report["D0"] = d0(min(deadline, time.monotonic()+300))
        statuses["d0"] = Outcome(report["D0"]["outcome"])
        if statuses["d0"] == Outcome.PASS:
            y = data["assessment"]["labels"].cpu().numpy()
            sy = data["selection"]["labels"].cpu().numpy()
            prevalence = float(data["train"]["labels"].mean())
            null_rows = logloss_rows(np.full(y.shape, prevalence), y)
            report["null"] = {"train_prevalence": prevalence, "geometry_log_loss": null_rows.tolist()}
            stage = "d1"
            stage_deadline = min(deadline, time.monotonic()+2400)
            for seed in PLAN["seeds"]:
                model, curve = fit_reference(data, seed, stage_deadline)
                sp = ref_predict(model, data["selection"])
                t = threshold(sp, sy)
                p = ref_predict(model, data["assessment"])
                report["D1"].append({"seed": seed, "curve": curve, "selection": metrics(sp, sy, t),
                                     "assessment": evaluate(p, y, t)})
                torch.save({"model": model.state_dict(), "predictions": p, "threshold": t}, output / f"d1-{seed}.pt")
            report["D1_assessment"] = classify(report["D1"], null_rows)
            statuses["d1"] = Outcome(report["D1_assessment"]["outcome"])
            stage = "d2"
            stage_deadline = min(deadline, time.monotonic()+2400)
            for seed in PLAN["seeds"]:
                encoder = base.make_encoder(seed, torch.device("cuda"))
                encoder.load_state_dict(torch.load(artifacts / f"joint-{seed}-encoder.pt", weights_only=True, map_location="cuda"))
                encoder.eval().requires_grad_(False)
                before = base.encoder_state_sha256(encoder)
                if before != frozen["payload"]["states"][str(seed)]["final_state_sha256"]:
                    raise ValueError("loaded encoder state mismatch")
                sample = {k: v[:2] for k, v in data["train"].items() if isinstance(v, torch.Tensor)}
                altered = dict(sample)
                altered["occupancy"] = torch.where(sample["hidden"][:, 0], 1-sample["occupancy"], sample["occupancy"])
                if not torch.equal(visible(sample), visible(altered)):
                    raise ValueError("hidden-target leakage in visible input")
                with torch.no_grad():
                    a = encoder(base.VoxelLevel.from_occupancy(sample["occupancy"], unknown_mask=sample["hidden"]), sample["hidden"]).local_feature_volume
                    b = encoder(base.VoxelLevel.from_occupancy(altered["occupancy"], unknown_mask=sample["hidden"]), sample["hidden"]).local_feature_volume
                    if not torch.equal(a, b):
                        raise ValueError("encoder mask isolation failure")
                features = {}
                with torch.no_grad():
                    for split, d in data.items():
                        rows = []
                        for start in range(0, len(d["labels"]), 8):
                            occ, hidden = d["occupancy"][start:start+8], d["hidden"][start:start+8]
                            volume = encoder(base.VoxelLevel.from_occupancy(occ, unknown_mask=hidden), hidden).local_feature_volume
                            rows.append(base.gather_features(volume, d["indices"][start:start+8]))
                        features[split] = torch.cat(rows)
                candidates = []
                for width in PLAN["probe_widths"]:
                    model, mean, scale, curve = fit_probe(features, data, seed, width, stage_deadline)
                    with torch.no_grad():
                        sp = model((features["selection"]-mean)/scale)[..., 0].sigmoid().cpu().numpy()
                        p = model((features["assessment"]-mean)/scale)[..., 0].sigmoid().cpu().numpy()
                    t = threshold(sp, sy)
                    candidates.append({"width": width, "selection": metrics(sp, sy, t), "assessment": evaluate(p, y, t), "curve": curve})
                    torch.save({"model": model.state_dict(), "mean": mean, "scale": scale,
                                "predictions": p, "threshold": t}, output / f"d2-{seed}-{width}.pt")
                selected = min(candidates, key=lambda r: r["selection"]["log_loss"])
                after = base.encoder_state_sha256(encoder)
                if before != after:
                    raise ValueError("frozen encoder mutated")
                report["D2"].append({"seed": seed, "selected_width": selected["width"],
                                     "assessment": selected["assessment"], "capacities": candidates,
                                     "initial_state_sha256": before, "final_state_sha256": after})
                print(json.dumps({"D2": seed, "width": selected["width"], "f1": selected["assessment"]["f1_calibrated"]}), flush=True)
            report["D2_assessment"] = classify(report["D2"], null_rows)
            statuses["d2"] = Outcome(report["D2_assessment"]["outcome"])
        errors = ()
    except TimeoutError as exc:
        statuses[stage] = Outcome.INCONCLUSIVE
        report["interruption"] = str(exc)
        errors = ()
    except (ValueError, FloatingPointError) as exc:
        errors = (str(exc),)
    elapsed = time.monotonic()-started + frozen["payload"]["preparation_seconds"]
    report["elapsed_seconds"] = elapsed
    report["decision"] = asdict(route(DiagnosticEvidence(validity_errors=errors, elapsed_hours=elapsed/3600, **statuses)))
    report["stage_outcomes"] = {k: v.value for k, v in statuses.items()}
    report["environment"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "precision": "float32"}
    report["artifacts"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.pt")}
    report["report_payload_sha256"] = base.payload_sha256(report)
    base.write_json(output / "report.json", report)
    print(json.dumps(report["decision"]), flush=True)
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
