"""LG2: isolated joint-objective study; preserves the historical LG1 protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from theseo_anysearch.garden.pilots import local_geometry as base


PLAN = {
    "study_id": "voxel-encoder-local-geometry-v1-lg2",
    "dataset_id": "voxel-encoder-local-geometry-v1-dataset-2",
    "query_id": "voxel-encoder-local-geometry-v1-queries-2",
    "schema_version": 2, "seeds": [0, 1, 2],
    "objectives": ["occupancy", "esdf", "joint"],
    "geometries": {"train": 96, "probe": 48, "evaluation": 48},
    "joint_esdf_weight": 10.0, "encoder_steps": 1024,
    "bootstrap_replicates": 2000, "bootstrap_seed": 354,
    "classification_noninferiority_margin": 0.03,
    "regression_noninferiority_margin": 0.02,
    "covariance_rank_role": "diagnostic_only",
    "near_dead_fraction_max": 0.05,
    "shared_settings": {k: base.PLAN[k] for k in (
        "radius", "observation_index", "mask_probability", "stem_width", "local_channels",
        "truncation", "encoder_batch", "encoder_lr", "weight_decay", "probe_steps", "probe_batch",
        "probe_hidden", "probe_lr", "queries_per_geometry", "absolute_bars", "classification_improvement",
        "regression_relative_reduction", "near_dead_std", "hours_cap")},
    "probe_initialization_seed_base": 41000, "probe_sampling_seed_base": 42000,
    "ordering_exception": "User explicitly authorized stacking on unmerged #353/specs#30 and executing before review; no merges authorized.",
}
CONTROLS = ("random", "zero", "shuffle", "shuffled_labels")


def records(split: str) -> list:
    return [base.GeometryDescriptor(
        f"{PLAN['dataset_id']}-{split}-{i:03d}", base.FAMILIES[(i % 12) // 3],
        base.BANDS[i % 3], "procedural-native-not-real-imports")
        for i in range(PLAN["geometries"][split])]


def rank_fixture_report() -> dict:
    """No study data: invertible scaling can change covariance rank, not information."""
    generator = torch.Generator().manual_seed(354)
    x = torch.randn(2048, 8, generator=generator, dtype=torch.float64)
    y = x.sum(1)
    fixtures = {"independent": x, "invertibly_scaled": x * torch.tensor([1.] + [.01] * 7),
                "useful_rank_one": y[:, None].expand(-1, 8), "constant": torch.zeros_like(x)}
    results = {}
    for name, features in fixtures.items():
        train, test = features[:1024], features[1024:]
        weights = np.linalg.lstsq(train.numpy(), y[:1024].numpy(), rcond=None)[0]
        prediction = test.numpy() @ weights
        normalized = (features - features.mean(0)) / features.std(0, unbiased=False).clamp_min(1e-12)
        results[name] = {**base.rank_diagnostics(features.unsqueeze(0)),
                         "correlation_rank_fraction": base.rank_diagnostics(normalized.unsqueeze(0))["effective_rank_fraction"],
                         "heldout_linear_mse": float(np.mean((prediction - y[1024:].numpy()) ** 2))}
    return {"development_only": True, "identity": "lg2-rank-fixtures-1", "fixtures": results}


def freeze(path: Path, spec_sha: str) -> dict:
    if len(spec_sha) != 40 or any(c not in "0123456789abcdef" for c in spec_sha):
        raise ValueError("full spec SHA required")
    if base.git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit all executable source before freezing")
    payload = {"plan": PLAN, "source_commit": base.git("rev-parse", "HEAD"),
               "spec_commit": spec_sha,
               "spec_path": "projects/theseo-anysearch/python/perception-encoder-local-geometry-lg2.md",
               "rank_fixture_report": rank_fixture_report(),
               "geometry_ids": {s: [r.geometry_id for r in records(s)] for s in PLAN["geometries"]}}
    result = {"payload": payload, "identity_sha256": base.payload_sha256(payload)}
    base.write_json(path, result)
    return result


def read_frozen(path: Path) -> dict:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if set(envelope) != {"payload", "identity_sha256"}:
        raise ValueError("invalid envelope")
    p = envelope["payload"]
    if base.payload_sha256(p) != envelope["identity_sha256"] or p["plan"] != PLAN:
        raise ValueError("registration mismatch")
    if p["geometry_ids"] != {s: [r.geometry_id for r in records(s)] for s in PLAN["geometries"]}:
        raise ValueError("split identity mismatch")
    if base.git("diff", p["source_commit"], "--", "theseo_anysearch/garden") or base.git(
            "ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("source differs from registration")
    return envelope


def materialize(split: str, device: torch.device) -> dict:
    result = {k: [] for k in ("occupancy", "boundary", "distance", "identities")}
    result["records"] = records(split)
    for record in result["records"]:
        obs = base.make_pilot_observation(record, 1, radius=8, program=base.LOCAL_PROGRAM)
        if obs.unknown_mask.any():
            raise ValueError("clean targets required")
        target = base.compute_geometry_targets(obs.occupancy, truncation=8.)
        result["occupancy"].append(obs.occupancy)
        result["boundary"].append(target.boundary)
        result["distance"].append(target.signed_distance / 8.)
        result["identities"].append(obs.identity_sha256)
    for key in ("occupancy", "boundary", "distance"):
        result[key] = torch.as_tensor(np.stack(result[key]), device=device, dtype=torch.float32)
    return result


def query_bank(data: dict, split: str) -> dict:
    device = data["occupancy"].device
    generator = torch.Generator(device=device).manual_seed(35401 if split == "probe" else 35402)
    hidden = base.independent_mask(data["occupancy"], generator)
    free = data["occupancy"] < .5
    masks = (hidden[:, 0], hidden[:, 0], ~hidden[:, 0] & free, hidden[:, 0] & free)
    indices, targets = {}, {}
    for task, mask in zip(base.TASKS, masks):
        rows = []
        for geometry in mask:
            choices = torch.where(geometry.flatten())[0]
            if not len(choices):
                raise ValueError("empty eligible query pool; no redraw permitted")
            draw = (torch.randperm(len(choices), device=device, generator=generator)[:256]
                    if len(choices) >= 256 else
                    torch.randint(len(choices), (256,), device=device, generator=generator))
            rows.append(choices[draw])
        indices[task] = torch.stack(rows)
        truth = (data["occupancy"] if task == base.TASKS[0] else data["boundary"]
                 if task == base.TASKS[1] else data["distance"].clamp_min(0))
        targets[task] = truth.flatten(1).gather(1, indices[task])
        if task in base.TASKS[:2] and min(int(targets[task].sum()), int((1 - targets[task]).sum())) < 100:
            raise ValueError("insufficient class support; no redraw permitted")
    digest = hashlib.sha256(hidden.cpu().numpy().tobytes())
    for task in base.TASKS:
        digest.update(indices[task].cpu().numpy().tobytes())
        digest.update(targets[task].cpu().numpy().tobytes())
    return {"hidden": hidden, "indices": indices, "targets": targets, "sha256": digest.hexdigest()}


def objective_loss(predictions: torch.Tensor, occupancy: torch.Tensor,
                   distance: torch.Tensor, hidden: torch.Tensor, objective: str) -> tuple:
    occupancy_loss = base.pretraining_loss(predictions[:, :1], occupancy, hidden, "occupancy")
    esdf_loss = base.pretraining_loss(predictions[:, 1:], distance, hidden, "esdf")
    if objective not in PLAN["objectives"]:
        raise ValueError("unknown objective")
    loss = (occupancy_loss if objective == "occupancy" else esdf_loss if objective == "esdf"
            else occupancy_loss + PLAN["joint_esdf_weight"] * esdf_loss)
    return loss, occupancy_loss, esdf_loss


def train(data: dict, objective: str, seed: int, deadline: float, *, steps: int | None = None) -> tuple:
    device = data["occupancy"].device
    model = base.make_encoder(seed, device).train()
    initial = base.encoder_state_sha256(model)
    # Instantiate both outputs in every arm to keep head initialization paired.
    head = nn.Conv3d(8, 2, 1).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad] + list(head.parameters()),
                                 lr=.001, weight_decay=.01)
    rng = torch.Generator(device=device).manual_seed(35400 + seed)
    started, curve = time.monotonic(), []
    count = PLAN["encoder_steps"] if steps is None else steps
    for step in range(count):
        if time.monotonic() > deadline:
            raise TimeoutError("LG2 compute cap exceeded")
        idx = torch.randint(len(data["occupancy"]), (8,), generator=rng, device=device)
        occupancy = data["occupancy"][idx]
        hidden = base.independent_mask(occupancy, rng)
        features = model(base.VoxelLevel.from_occupancy(occupancy), hidden).local_feature_volume
        loss, occ, sdf = objective_loss(head(features), occupancy, data["distance"][idx], hidden, objective)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 256 == 0 or step + 1 == count:
            row = {"step": step + 1, "loss": float(loss.detach()), "occupancy_loss": float(occ.detach()),
                   "esdf_loss": float(sdf.detach()), "seconds": time.monotonic() - started}
            curve.append(row)
            print(json.dumps({"objective": objective, "seed": seed, **row}), flush=True)
    return model.eval().requires_grad_(False), {"objective": objective, "seed": seed,
        "initial_state_sha256": initial, "final_state_sha256": base.encoder_state_sha256(model),
        "curve": curve, "updates": count}


def paired_comparison(candidates: list, controls: list, task: str) -> dict:
    if len(candidates) != 3 or len(controls) != 3 or any(len(x) != 48 for x in candidates + controls):
        raise ValueError("comparison requires 3 paired seeds and 48 geometries")
    c, b = [np.asarray(x) for x in candidates], [np.asarray(x) for x in controls]
    sign = 1 if task in base.TASKS[:2] else -1
    c_scores, b_scores = [base.score(x, task) for x in c], [base.score(x, task) for x in b]
    delta = [(x - y) * sign for x, y in zip(c_scores, b_scores)]
    rng = np.random.default_rng(PLAN["bootstrap_seed"])
    draws = []
    for _ in range(PLAN["bootstrap_replicates"]):
        idx = np.concatenate([rng.choice(np.arange(s, 48, 12), 4, replace=True) for s in range(12)])
        draws.append(np.mean([(base.score(x[idx], task) - base.score(y[idx], task)) * sign for x, y in zip(c, b)]))
    return {"candidate_seed_scores": c_scores, "control_seed_scores": b_scores,
            "seed_improvements": delta, "paired_geometry_ci95": np.quantile(draws, [.025, .975]).tolist()}


def assess(trials: list) -> dict:
    if len(trials) != 9:
        raise ValueError("all nine trials required")
    groups, recipes = {}, {}
    for objective in PLAN["objectives"]:
        selected = sorted([t for t in trials if t["objective"] == objective], key=lambda t: t["seed"])
        if [t["seed"] for t in selected] != PLAN["seeds"]:
            raise ValueError("missing/duplicate seed")
        groups[objective] = selected
        integrity = all(t["integrity_ok"] and t["mask_isolation_max_abs"] == 0 and
                        t["rank"]["near_dead_fraction"] <= .05 for t in selected)
        components, gates = {}, [integrity]
        for task in base.TASKS:
            candidate = [t["statistics"][task]["trained"] for t in selected]
            scores = [base.score(x, task) for x in candidate]
            absolute = all(v >= base.PLAN["absolute_bars"][task] if task in base.TASKS[:2]
                           else v <= base.PLAN["absolute_bars"][task] for v in scores)
            details = {"seed_scores": scores, "absolute_pass": absolute, "controls": {}}
            gates.append(absolute)
            for control in CONTROLS:
                comparison = paired_comparison(candidate, [t["statistics"][task][control] for t in selected], task)
                margin_pass = all(d >= .05 if task in base.TASKS[:2] else d >= .1 * b and d > 0
                                  for d, b in zip(comparison["seed_improvements"], comparison["control_seed_scores"]))
                comparison["pass"] = margin_pass and comparison["paired_geometry_ci95"][0] > 0
                gates.append(comparison["pass"])
                details["controls"][control] = comparison
            components[task] = details
        recipes[objective] = {"decision": "qualifies_local_followup" if all(gates) else "not_qualified",
                              "integrity_pass": integrity, "components": components}
    preservation = {}
    for task in base.TASKS:
        specialist = "occupancy" if task in base.TASKS[:2] else "esdf"
        result = paired_comparison([t["statistics"][task]["trained"] for t in groups["joint"]],
                                   [t["statistics"][task]["trained"] for t in groups[specialist]], task)
        margin = PLAN["classification_noninferiority_margin"] if task in base.TASKS[:2] else PLAN["regression_noninferiority_margin"]
        result.update({"specialist": specialist, "margin": margin,
                       "pass": min(result["seed_improvements"]) >= -margin and result["paired_geometry_ci95"][0] >= -margin})
        preservation[task] = result
    success = recipes["joint"]["decision"] == "qualifies_local_followup" and all(r["pass"] for r in preservation.values())
    return {"recipes": recipes, "joint_preservation": preservation,
            "decision": "joint_preserves_specialists" if success else "joint_hypothesis_not_validated"}


def evaluate(model: nn.Module, result: dict, probe_data: dict, evaluation: dict,
             probe_bank: dict, eval_bank: dict, raw_stats: dict, output: Path) -> dict:
    seed, objective = result["seed"], result["objective"]
    before = base.encoder_state_sha256(model)
    random = base.make_encoder(seed, evaluation["occupancy"].device).eval().requires_grad_(False)
    if base.encoder_state_sha256(random) != result["initial_state_sha256"]:
        raise ValueError("paired initialization mismatch")
    fitted_features, features = base.extract(model, probe_data, probe_bank), base.extract(model, evaluation, eval_bank)
    random_fit, random_eval = base.extract(random, probe_data, probe_bank), base.extract(random, evaluation, eval_bank)
    statistics, saved, probes = {}, {}, {}
    for task in base.TASKS:
        fit = base.fit_probe(fitted_features[task], probe_bank["targets"][task], task, seed)
        label_fit = base.fit_probe(fitted_features[task], probe_bank["targets"][task], task, seed, shuffled_labels=True)
        random_probe = base.fit_probe(random_fit[task], probe_bank["targets"][task], task, seed)
        values = features[task]
        flat = values.flatten(0, 1)
        rng = torch.Generator(device=values.device).manual_seed(35430 + seed)
        shuffled = flat[torch.randperm(len(flat), generator=rng, device=values.device)].reshape_as(values)
        predictions = {"trained": base.predict(fit, values, task),
                       "zero": base.predict(fit, torch.zeros_like(values), task),
                       "shuffle": base.predict(fit, shuffled, task),
                       "shuffled_labels": base.predict(label_fit, values, task),
                       "random": base.predict(random_probe, random_eval[task], task)}
        statistics[task] = {name: base.sufficient_statistics(p, eval_bank["targets"][task], task) for name, p in predictions.items()}
        statistics[task]["raw_neighborhood_diagnostic"] = raw_stats[seed][task]
        statistics[task]["full_information_oracle_diagnostic"] = base.sufficient_statistics(eval_bank["targets"][task], eval_bank["targets"][task], task)
        saved[task] = {name: p.cpu() for name, p in predictions.items()}
        probes[task] = {"state": fit[0].state_dict(), "mean": fit[1], "scale": fit[2]}
    torch.save({"predictions": saved, "targets": eval_bank["targets"], "probes": probes}, output / f"{objective}-{seed}-evaluation.pt")
    rank = base.rank_diagnostics(features[base.TASKS[0]])
    x = features[base.TASKS[0]]
    standardized = (x - x.flatten(0, 1).mean(0)) / x.flatten(0, 1).std(0, unbiased=False).clamp_min(1e-12)
    rank["correlation_rank_fraction"] = base.rank_diagnostics(standardized)["effective_rank_fraction"]
    rank["legacy_lg1_rank_floor_pass"] = rank["effective_rank_fraction"] >= .25
    isolation = base.mask_isolation_max_abs(model, base.VoxelLevel.from_occupancy(evaluation["occupancy"][:1]), eval_bank["hidden"][:1])
    return {**result, "statistics": statistics, "rank": rank, "mask_isolation_max_abs": isolation,
            "integrity_ok": before == base.encoder_state_sha256(model),
            "run_id": f"{PLAN['study_id']}-{objective}-seed-{seed}"}


def run(registration: Path, output: Path) -> dict:
    frozen = read_frozen(registration)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; development fixtures are CPU-only")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + 4 * 3600
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    device = torch.device("cuda")
    data = materialize("train", device)
    models, results = {}, []
    for objective in PLAN["objectives"]:
        for seed in PLAN["seeds"]:
            model, result = train(data, objective, seed, deadline)
            models[(objective, seed)] = model
            results.append(result)
            torch.save(model.state_dict(), output / f"{objective}-{seed}-encoder.pt")
    probe, evaluation = materialize("probe", device), materialize("evaluation", device)
    pb, eb = query_bank(probe, "probe"), query_bank(evaluation, "evaluation")
    raw_fit, raw_eval = base.extract(None, probe, pb), base.extract(None, evaluation, eb)
    raw_stats = {seed: {task: base.sufficient_statistics(base.predict(
        base.fit_probe(raw_fit[task], pb["targets"][task], task, seed), raw_eval[task], task), eb["targets"][task], task)
        for task in base.TASKS} for seed in PLAN["seeds"]}
    trials = []
    for result in results:
        if time.monotonic() > deadline:
            raise TimeoutError("LG2 compute cap exceeded")
        trial = evaluate(models[(result["objective"], result["seed"])], result, probe, evaluation, pb, eb, raw_stats, output)
        trials.append(trial)
        base.write_json(output / f"{trial['objective']}-{trial['seed']}-trial.json", trial)
        print(json.dumps({"completed_probe_trial": trial["run_id"]}), flush=True)
    assessment = assess(trials)
    if time.monotonic() > deadline:
        raise TimeoutError("LG2 compute cap exceeded")
    report = {"completed": True, "registration": frozen, "assessment": assessment, "training_trials": trials,
              "elapsed_seconds": time.monotonic() - started,
              "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "precision": "float32",
                              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()},
              "geometry_ids": {s: [r.geometry_id for r in records(s)] for s in PLAN["geometries"]},
              "corpus_hashes": {"train": data["identities"], "probe": probe["identities"], "evaluation": evaluation["identities"]},
              "query_hashes": {"probe": pb["sha256"], "evaluation": eb["sha256"]},
              "untracked_artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.pt")},
              "limitations": ["Three seeds; intervals conditional on seeds, not seed-population inference.",
                              "Equal updates and batches, not exact equal FLOPs; joint head has two supervised outputs.",
                              "Local synthetic radius-eight evidence only; no topology, P1-P8 or global projection qualification."]}
    report["report_payload_sha256"] = base.payload_sha256(report)
    base.write_json(output / "report.json", report)
    print(json.dumps({"completed": True, "decision": assessment["decision"], "seconds": report["elapsed_seconds"]}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fixtures", "freeze", "run"))
    parser.add_argument("--registration", type=Path)
    parser.add_argument("--spec-sha")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "fixtures":
        report = rank_fixture_report()
        if args.output:
            base.write_json(args.output, report)
        print(json.dumps(report), flush=True)
    elif args.command == "freeze":
        print(json.dumps(freeze(args.registration, args.spec_sha)), flush=True)
    else:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        run(args.registration, args.output)


if __name__ == "__main__":
    main()
