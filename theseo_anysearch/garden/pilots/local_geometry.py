"""LG1: standalone, preregistered local-feature pretraining comparison.

This does not consume topology-study contracts or advance the P1-P8 chain.
Run ``python -m theseo_anysearch.garden.pilots.local_geometry --help``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from theseo_anysearch.garden.evaluation.probes import encoder_state_sha256
from theseo_anysearch.garden.masking import DenseMaskAwareEncoder, mask_isolation_max_abs
from theseo_anysearch.garden.models.outputs import VoxelLevel
from theseo_anysearch.garden.pilots.corpus import LOCAL_PROGRAM, make_pilot_observation
from theseo_anysearch.garden.pilots.io import payload_sha256
from theseo_anysearch.garden.splits import GeometryDescriptor
from theseo_anysearch.garden.targets import compute_geometry_targets


TASKS = ("occupied_iou", "boundary_f1", "clearance_nmae", "recovery_nmae")
FAMILIES = ("open", "thin_obstacle", "topology", "imported")
BANDS = ("low", "medium", "high")
PLAN = {
    "program": LOCAL_PROGRAM,
    "study_id": LOCAL_PROGRAM + "-lg1",
    "dataset_id": LOCAL_PROGRAM + "-dataset-1",
    "query_id": LOCAL_PROGRAM + "-queries-1",
    "schema_version": 1,
    "seeds": [0, 1], "objectives": ["occupancy", "esdf"],
    "geometries": {"train": 96, "probe": 48, "evaluation": 48},
    "radius": 8, "observation_index": 1, "mask_probability": 0.20,
    "stem_width": 8, "local_channels": 8, "truncation": 8.0,
    "encoder_steps": 1024, "encoder_batch": 8, "encoder_lr": 0.001,
    "weight_decay": 0.01, "probe_steps": 512, "probe_batch": 1024,
    "probe_hidden": 32, "probe_lr": 0.001, "queries_per_geometry": 256,
    "bootstrap_replicates": 2000, "bootstrap_seed": 341,
    "absolute_bars": {"occupied_iou": 0.60, "boundary_f1": 0.70,
                      "clearance_nmae": 0.15, "recovery_nmae": 0.20},
    "classification_improvement": 0.05, "regression_relative_reduction": 0.10,
    "rank_fraction_min": 0.25, "near_dead_fraction_max": 0.05,
    "near_dead_std": 0.00001, "hours_cap": 4,
    "scope": "local_features_only_no_topology_no_P1_P8",
    "ordering_exception": "User explicitly authorized independent execution before #352/specs#28 merge on 2026-09-08; no merges authorized.",
}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def geometry_records(split: str, *, development: bool = False) -> list[GeometryDescriptor]:
    count = PLAN["geometries"][split]
    prefix = "lg1-development" if development else PLAN["dataset_id"]
    return [GeometryDescriptor(
        geometry_id=f"{prefix}-{split}-{i:03d}",
        family=FAMILIES[(i % 12) // 3], occupancy_band=BANDS[i % 3],
        source="procedural-native-not-real-imports",
    ) for i in range(count)]


def git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def freeze(path: Path, spec_sha: str) -> dict:
    if len(spec_sha) != 40 or any(c not in "0123456789abcdef" for c in spec_sha):
        raise ValueError("spec SHA must be a full lowercase Git SHA")
    if git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit executable garden source before freezing")
    payload = {"plan": PLAN, "source_commit": git("rev-parse", "HEAD"),
               "spec_commit": spec_sha,
               "spec_path": "projects/theseo-anysearch/python/perception-encoder-local-geometry.md",
               "geometry_ids": {s: [r.geometry_id for r in geometry_records(s)]
                                for s in PLAN["geometries"]}}
    envelope = {"payload": payload, "identity_sha256": payload_sha256(payload)}
    write_json(path, envelope)
    return envelope


def read_frozen(path: Path) -> dict:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if set(envelope) != {"payload", "identity_sha256"}:
        raise ValueError("invalid registration envelope")
    payload = envelope["payload"]
    if payload_sha256(payload) != envelope["identity_sha256"] or payload["plan"] != PLAN:
        raise ValueError("registration identity or frozen plan mismatch")
    expected = {s: [r.geometry_id for r in geometry_records(s)] for s in PLAN["geometries"]}
    if payload["geometry_ids"] != expected:
        raise ValueError("registration split identities mismatch")
    if git("diff", payload["source_commit"], "--", "theseo_anysearch/garden"):
        raise ValueError("executable source differs from preregistration")
    if git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("unregistered garden source exists")
    return envelope


def materialize(split: str, device: torch.device) -> dict:
    records = geometry_records(split)
    occupancy, boundary, distance, identities = [], [], [], []
    for record in records:
        obs = make_pilot_observation(record, PLAN["observation_index"],
                                     radius=PLAN["radius"], program=LOCAL_PROGRAM)
        if obs.unknown_mask.any():
            raise ValueError("LG1 requires clean targets before independent masking")
        targets = compute_geometry_targets(obs.occupancy, truncation=PLAN["truncation"])
        occupancy.append(obs.occupancy)
        boundary.append(targets.boundary)
        distance.append(targets.signed_distance / PLAN["truncation"])
        identities.append(obs.identity_sha256)
    tensor = lambda arrays: torch.as_tensor(np.stack(arrays), device=device, dtype=torch.float32)
    return {"occupancy": tensor(occupancy), "boundary": tensor(boundary),
            "distance": tensor(distance), "records": records,
            "identities": identities}


def independent_mask(occupancy: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return (torch.rand(occupancy.shape, device=occupancy.device, generator=generator)
            < PLAN["mask_probability"]).unsqueeze(1)


def make_encoder(seed: int, device: torch.device) -> DenseMaskAwareEncoder:
    torch.manual_seed(seed)
    model = DenseMaskAwareEncoder(stem_width=PLAN["stem_width"],
                                  local_channels=PLAN["local_channels"]).to(device)
    # The local losses do not train the global projection; it is outside this study.
    model.projection.requires_grad_(False)
    return model


def pretraining_loss(logits: torch.Tensor, target: torch.Tensor,
                     hidden: torch.Tensor, objective: str) -> torch.Tensor:
    prediction, selected = logits[:, 0], hidden[:, 0]
    if not selected.any():
        raise ValueError("empty pretraining mask")
    if objective == "occupancy":
        return F.binary_cross_entropy_with_logits(prediction[selected], target[selected])
    if objective == "esdf":
        return F.smooth_l1_loss(prediction[selected], target[selected])
    raise ValueError("unknown pretraining objective")


def train_encoder(data: dict, objective: str, seed: int, deadline: float) -> tuple:
    device = data["occupancy"].device
    model = make_encoder(seed, device).train()
    initial_hash = encoder_state_sha256(model)
    head = nn.Conv3d(PLAN["local_channels"], 1, 1).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad]
                                 + list(head.parameters()), lr=PLAN["encoder_lr"],
                                 weight_decay=PLAN["weight_decay"])
    generator = torch.Generator(device=device).manual_seed(34100 + seed)
    curves = []
    started = time.monotonic()
    for step in range(PLAN["encoder_steps"]):
        if time.monotonic() > deadline:
            raise TimeoutError("preregistered LG1 compute cap exceeded")
        indices = torch.randint(len(data["occupancy"]), (PLAN["encoder_batch"],),
                                generator=generator, device=device)
        occupancy = data["occupancy"][indices]
        hidden = independent_mask(occupancy, generator)
        output = model(VoxelLevel.from_occupancy(occupancy), hidden)
        target = occupancy if objective == "occupancy" else data["distance"][indices]
        loss = pretraining_loss(head(output.local_feature_volume), target, hidden, objective)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite encoder loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 128 == 0 or step + 1 == PLAN["encoder_steps"]:
            row = {"step": step + 1, "loss": float(loss.detach()),
                   "seconds": time.monotonic() - started}
            curves.append(row)
            print(json.dumps({"objective": objective, "seed": seed, **row}), flush=True)
    model.eval().requires_grad_(False)
    return model, {"objective": objective, "seed": seed, "curve": curves,
                   "initial_state_sha256": initial_hash,
                   "final_state_sha256": encoder_state_sha256(model),
                   "updates": PLAN["encoder_steps"]}


def query_bank(data: dict, split: str) -> dict:
    device = data["occupancy"].device
    generator = torch.Generator(device=device).manual_seed(34101 if split == "probe" else 34102)
    hidden = independent_mask(data["occupancy"], generator)
    free = data["occupancy"] < 0.5
    eligible = {TASKS[0]: hidden[:, 0], TASKS[1]: hidden[:, 0],
                TASKS[2]: ~hidden[:, 0] & free, TASKS[3]: hidden[:, 0] & free}
    indices, targets = {}, {}
    for task in TASKS:
        rows = []
        for mask in eligible[task]:
            choices = torch.where(mask.flatten())[0]
            if not len(choices):
                raise ValueError(f"no eligible queries for {split}/{task}")
            # Fixed-size uniform queries, with replacement only in small eligible pools.
            if len(choices) >= PLAN["queries_per_geometry"]:
                draw = torch.randperm(len(choices), generator=generator, device=device)
                rows.append(choices[draw[:PLAN["queries_per_geometry"]]])
            else:
                draw = torch.randint(len(choices), (PLAN["queries_per_geometry"],),
                                     generator=generator, device=device)
                rows.append(choices[draw])
        indices[task] = torch.stack(rows)
        truth = (data["occupancy"] if task == TASKS[0] else data["boundary"]
                 if task == TASKS[1] else data["distance"].clamp_min(0))
        targets[task] = truth.flatten(1).gather(1, indices[task])
        if task in TASKS[:2] and min(int(targets[task].sum()),
                                    int((1 - targets[task]).sum())) < 100:
            raise ValueError(f"insufficient class support for {split}/{task}")
    digest = hashlib.sha256(hidden.cpu().numpy().tobytes())
    for task in TASKS:
        digest.update(indices[task].cpu().numpy().tobytes())
        digest.update(targets[task].cpu().numpy().tobytes())
    return {"hidden": hidden, "indices": indices, "targets": targets,
            "sha256": digest.hexdigest()}


def gather_features(volume: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    cells = volume.flatten(2).transpose(1, 2)
    return cells.gather(1, indices.unsqueeze(-1).expand(-1, -1, cells.shape[-1]))


@torch.no_grad()
def extract(model: nn.Module | None, data: dict, bank: dict) -> dict:
    outputs = {task: [] for task in TASKS}
    for start in range(0, len(data["occupancy"]), PLAN["encoder_batch"]):
        end = start + PLAN["encoder_batch"]
        occupancy, hidden = data["occupancy"][start:end], bank["hidden"][start:end]
        level = VoxelLevel.from_occupancy(occupancy, unknown_mask=hidden)
        if model is not None:
            volume = model(level, hidden).local_feature_volume
        else:
            # A visible-only 3x3x3 neighborhood; no clean/hidden truth is available.
            padded = F.pad(level.features[:, :3], (1, 1, 1, 1, 1, 1))
            patches = padded.unfold(2, 3, 1).unfold(3, 3, 1).unfold(4, 3, 1)
            volume = patches.permute(0, 1, 5, 6, 7, 2, 3, 4).reshape(
                len(occupancy), 81, *occupancy.shape[1:])
        for task in TASKS:
            outputs[task].append(gather_features(volume, bank["indices"][task][start:end]))
    return {task: torch.cat(values) for task, values in outputs.items()}


def fit_probe(features: torch.Tensor, targets: torch.Tensor, task: str, seed: int,
              *, shuffled_labels: bool = False, steps: int | None = None) -> tuple:
    torch.manual_seed(41000 + seed)
    inputs, labels = features.flatten(0, 1), targets.flatten()
    mean = inputs.mean(0)
    scale = inputs.std(0, unbiased=False).clamp_min(0.00001)
    normalized = (inputs - mean) / scale
    generator = torch.Generator(device=inputs.device).manual_seed(42000 + seed)
    if shuffled_labels:
        permutation = torch.randperm(len(labels), generator=generator, device=inputs.device)
        labels = labels[permutation]
        # Reset minibatch stream to match the true-label control's budget and draws.
        generator.manual_seed(42000 + seed)
    probe = nn.Sequential(nn.Linear(inputs.shape[-1], PLAN["probe_hidden"]), nn.SiLU(),
                          nn.Linear(PLAN["probe_hidden"], 1)).to(inputs.device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=PLAN["probe_lr"],
                                 weight_decay=PLAN["weight_decay"])
    for _ in range(PLAN["probe_steps"] if steps is None else steps):
        idx = torch.randint(len(inputs), (PLAN["probe_batch"],),
                            generator=generator, device=inputs.device)
        prediction = probe(normalized[idx])[:, 0]
        loss = (F.binary_cross_entropy_with_logits(prediction, labels[idx])
                if task in TASKS[:2] else F.smooth_l1_loss(prediction, labels[idx]))
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite probe loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return probe.eval().requires_grad_(False), mean, scale


@torch.no_grad()
def predict(fitted: tuple, features: torch.Tensor, task: str) -> torch.Tensor:
    probe, mean, scale = fitted
    flat = features.flatten(0, 1)
    prediction = probe((flat - mean) / scale)[:, 0].reshape(features.shape[:2])
    return prediction.sigmoid() if task in TASKS[:2] else prediction.clamp(0, 1)


def sufficient_statistics(prediction: torch.Tensor, target: torch.Tensor,
                          task: str) -> list:
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("nonfinite evaluation predictions")
    if task in TASKS[:2]:
        p, t = prediction >= 0.5, target >= 0.5
        stats = torch.stack(((p & t).sum(1), (p & ~t).sum(1), (~p & t).sum(1)), 1)
    else:
        stats = torch.stack(((prediction - target).abs().sum(1),
                             torch.full_like(target[:, 0], target.shape[1])), 1)
    return stats.cpu().tolist()


def score(statistics: np.ndarray, task: str) -> float:
    totals = np.asarray(statistics, dtype=float).sum(axis=0)
    if task == TASKS[0]:
        denominator = totals.sum()
        return float(totals[0] / denominator) if denominator else 1.0
    if task == TASKS[1]:
        denominator = 2 * totals[0] + totals[1] + totals[2]
        return float(2 * totals[0] / denominator) if denominator else 1.0
    return float(totals[0] / totals[1])


def rank_diagnostics(features: torch.Tensor) -> dict:
    flat = features.flatten(0, 1).double()
    centered = flat - flat.mean(0)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / len(centered)).clamp_min(0)
    total = eigenvalues.sum()
    probabilities = eigenvalues / total.clamp_min(1e-30)
    rank = float(torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum())) if total > 0 else 0.0
    dead = float((flat.std(0, unbiased=False) < PLAN["near_dead_std"]).double().mean())
    return {"effective_rank_fraction": rank / flat.shape[-1], "near_dead_fraction": dead}


def assess(trials: list) -> dict:
    assessment = {}
    for objective in PLAN["objectives"]:
        selected = [trial for trial in trials if trial["objective"] == objective]
        if len(selected) != len(PLAN["seeds"]) or sorted(t["seed"] for t in selected) != PLAN["seeds"]:
            raise ValueError("assessment requires every preregistered seed")
        gates, components = [], {}
        for trial in selected:
            gates.append(trial["integrity_ok"] and trial["mask_isolation_max_abs"] == 0.0)
            gates.append(trial["rank"]["effective_rank_fraction"] >= PLAN["rank_fraction_min"])
            gates.append(trial["rank"]["near_dead_fraction"] <= PLAN["near_dead_fraction_max"])
        for task in TASKS:
            higher = task in TASKS[:2]
            candidate_scores = [score(t["statistics"][task]["trained"], task) for t in selected]
            absolute = all(v >= PLAN["absolute_bars"][task] if higher else
                           v <= PLAN["absolute_bars"][task] for v in candidate_scores)
            component = {"seed_scores": candidate_scores, "absolute_pass": absolute, "controls": {}}
            gates.append(absolute)
            for control in ("random", "zero", "shuffle", "shuffled_labels"):
                c = [np.asarray(t["statistics"][task]["trained"]) for t in selected]
                b = [np.asarray(t["statistics"][task][control]) for t in selected]
                if any(len(rows) != PLAN["geometries"]["evaluation"] for rows in c + b):
                    raise ValueError("evaluation geometry count mismatch")
                baseline_scores = [score(rows, task) for rows in b]
                delta = [(v - u if higher else u - v) for v, u in zip(candidate_scores, baseline_scores)]
                point_pass = all(d >= PLAN["classification_improvement"] if higher else
                                 d >= PLAN["regression_relative_reduction"] * u and d > 0
                                 for d, u in zip(delta, baseline_scores))
                rng = np.random.default_rng(PLAN["bootstrap_seed"])
                boot = []
                for _ in range(PLAN["bootstrap_replicates"]):
                    # Same geometry draws across candidates, controls and seeds; keep all 12 strata.
                    idx = np.concatenate([rng.choice(np.arange(s, 48, 12), 4, replace=True)
                                          for s in range(12)])
                    boot.append(np.mean([(score(x[idx], task) - score(y[idx], task)) * (1 if higher else -1)
                                         for x, y in zip(c, b)]))
                interval = np.quantile(boot, [0.025, 0.975]).tolist()
                passed = point_pass and interval[0] > 0
                component["controls"][control] = {"seed_scores": baseline_scores,
                    "seed_improvements": delta, "paired_geometry_ci95": interval, "pass": passed}
                gates.append(passed)
            components[task] = component
        assessment[objective] = {"decision": "qualifies_local_followup" if all(gates) else "not_qualified",
                                 "components": components}
    return assessment


def run(registration: Path, output: Path) -> dict:
    frozen = read_frozen(registration)
    if not torch.cuda.is_available():
        raise RuntimeError("LG1 study requires CUDA; CPU tests are development-only")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + PLAN["hours_cap"] * 3600
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    train = materialize("train", device)
    models, training = {}, []
    for objective in PLAN["objectives"]:
        for seed in PLAN["seeds"]:
            model, result = train_encoder(train, objective, seed, deadline)
            models[(objective, seed)] = model
            training.append(result)
            torch.save(model.state_dict(), output / f"{objective}-{seed}-encoder.pt")
    del train["occupancy"], train["boundary"], train["distance"]
    probe_data = materialize("probe", device)
    evaluation = materialize("evaluation", device)
    probe_bank = query_bank(probe_data, "probe")
    eval_bank = query_bank(evaluation, "evaluation")
    raw_train, raw_eval = extract(None, probe_data, probe_bank), extract(None, evaluation, eval_bank)
    raw_stats = {}
    for seed in PLAN["seeds"]:
        raw_stats[seed] = {task: sufficient_statistics(
            predict(fit_probe(raw_train[task], probe_bank["targets"][task], task, seed), raw_eval[task], task),
            eval_bank["targets"][task], task) for task in TASKS}
    trials = []
    for result in training:
        if time.monotonic() > deadline:
            raise TimeoutError("preregistered LG1 compute cap exceeded")
        objective, seed = result["objective"], result["seed"]
        model = models[(objective, seed)]
        before = encoder_state_sha256(model)
        random_model = make_encoder(seed, device).eval().requires_grad_(False)
        if encoder_state_sha256(random_model) != result["initial_state_sha256"]:
            raise ValueError("random encoder does not match paired initialization")
        train_features, eval_features = extract(model, probe_data, probe_bank), extract(model, evaluation, eval_bank)
        random_train, random_eval = extract(random_model, probe_data, probe_bank), extract(random_model, evaluation, eval_bank)
        statistics, prediction_payload, probe_states = {}, {}, {}
        for task in TASKS:
            fitted = fit_probe(train_features[task], probe_bank["targets"][task], task, seed)
            label_control = fit_probe(train_features[task], probe_bank["targets"][task], task, seed, shuffled_labels=True)
            random_probe = fit_probe(random_train[task], probe_bank["targets"][task], task, seed)
            features = eval_features[task]
            generator = torch.Generator(device=device).manual_seed(43000 + seed)
            flattened = features.flatten(0, 1)
            shuffled = flattened[torch.randperm(len(flattened), generator=generator, device=device)].reshape_as(features)
            predictions = {"trained": predict(fitted, features, task),
                           "zero": predict(fitted, torch.zeros_like(features), task),
                           "shuffle": predict(fitted, shuffled, task),
                           "shuffled_labels": predict(label_control, features, task),
                           "random": predict(random_probe, random_eval[task], task)}
            statistics[task] = {name: sufficient_statistics(values, eval_bank["targets"][task], task)
                                for name, values in predictions.items()}
            statistics[task]["raw_neighborhood_diagnostic"] = raw_stats[seed][task]
            statistics[task]["full_information_oracle_diagnostic"] = sufficient_statistics(
                eval_bank["targets"][task], eval_bank["targets"][task], task)
            prediction_payload[task] = {k: v.cpu() for k, v in predictions.items()}
            probe_states[task] = {"state": fitted[0].state_dict(), "mean": fitted[1], "scale": fitted[2]}
        torch.save({"predictions": prediction_payload, "targets": eval_bank["targets"],
                    "probes": probe_states}, output / f"{objective}-{seed}-evaluation.pt")
        isolation = mask_isolation_max_abs(model, VoxelLevel.from_occupancy(evaluation["occupancy"][:1]),
                                           eval_bank["hidden"][:1])
        trial = {**result, "statistics": statistics,
                 "rank": rank_diagnostics(eval_features[TASKS[0]]),
                 "mask_isolation_max_abs": isolation,
                 "integrity_ok": before == encoder_state_sha256(model),
                 "run_id": f"{PLAN['study_id']}-{objective}-seed-{seed}"}
        trials.append(trial)
        write_json(output / f"{objective}-{seed}-trial.json", trial)
        print(json.dumps({"completed_probe_trial": trial["run_id"], "rank": trial["rank"]}), flush=True)
    assessment = assess(trials)
    if time.monotonic() > deadline:
        raise TimeoutError("preregistered LG1 compute cap exceeded during evaluation")
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.pt")}
    report = {"program": LOCAL_PROGRAM, "registration": frozen,
              "scope": PLAN["scope"], "completed": True, "training_trials": trials,
              "assessment": assessment, "elapsed_seconds": time.monotonic() - started,
              "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
                              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                              "deterministic_algorithms": True, "precision": "float32"},
              "corpus_hashes": {"train": train["identities"], "probe": probe_data["identities"],
                                "evaluation": evaluation["identities"]},
              "query_hashes": {"probe": probe_bank["sha256"], "evaluation": eval_bank["sha256"]},
              "evaluation_geometry_ids": [r.geometry_id for r in evaluation["records"]],
              "untracked_artifact_sha256": files,
              "limitations": ["Two seeds; intervals resample geometries, not a seed population.",
                              "Synthetic within-generator holdout only; no external-domain evidence.",
                              "One radius, one architecture; no architecture winner or scaling claim.",
                              "Local features only; no global embedding, topology or P1-P8 qualification."]}
    report["report_payload_sha256"] = payload_sha256(report)
    write_json(output / "report.json", report)
    print(json.dumps({"completed": True, "assessment": assessment,
                      "seconds": report["elapsed_seconds"]}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze_parser = sub.add_parser("freeze")
    freeze_parser.add_argument("--registration", type=Path, required=True)
    freeze_parser.add_argument("--spec-sha", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--registration", type=Path, required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(args.registration, args.spec_sha)), flush=True)
    else:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        run(args.registration, args.output)


if __name__ == "__main__":
    main()
