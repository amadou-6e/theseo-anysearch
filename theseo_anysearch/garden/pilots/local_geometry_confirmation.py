"""LG3: independent-generator confirmation without encoder retraining."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
from torch import nn

from theseo_anysearch.garden.pilots import local_geometry_joint as prior

base = prior.base
FAMILIES = ("random_field", "oblique_sheets", "height_field", "curved_tubes")
PLAN = {
    "study_id": "voxel-encoder-local-geometry-v1-lg3",
    "dataset_id": "voxel-encoder-local-geometry-v1-dataset-3",
    "query_id": "voxel-encoder-local-geometry-v1-queries-3",
    "seeds": [0, 1, 2], "split_counts": {"probe": 48, "evaluation": 48},
    "families": list(FAMILIES), "occupied_fractions": [.08, .16, .28],
    "encoder_updates": 0, "hours_cap": 2,
    "query_mask_streams": {"probe": 35401, "evaluation": 35402},
    "shared_settings": prior.PLAN["shared_settings"],
    "bootstrap_seed": 354, "bootstrap_replicates": 2000,
    "family_absolute_gates": True,
    "source_report_sha256": "a1c99cf09adc3301dbd36307575e723a094c55a1335d996761a6cc5a1f3f135a",
    "ordering_exception": "User explicitly authorizes stacking on unmerged LG2 and execution before review; no merges.",
}


def identity(split: str, index: int, development: bool = False) -> str:
    prefix = "lg3-development" if development else PLAN["dataset_id"]
    return f"{prefix}-{split}-{index:03d}"


def generate(geometry_id: str, family: str, fraction: float) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(geometry_id.encode("ascii")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    x, y, z = np.meshgrid(*([np.linspace(-1, 1, 17)] * 3), indexing="ij")
    phase = rng.uniform(-np.pi, np.pi, 3)
    if family == "random_field":
        field = gaussian_filter(rng.normal(size=x.shape), sigma=1.5, mode="reflect")
    elif family == "oblique_sheets":
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        field = np.abs(np.sin(rng.uniform(4, 8) * (direction[0]*x + direction[1]*y + direction[2]*z) + phase[0]))
    elif family == "height_field":
        field = z - .35*np.sin(3*x + phase[0]) - .25*np.cos(4*y + phase[1]) - rng.uniform(-.3, .3)*x
    elif family == "curved_tubes":
        a = (y - .45*np.sin(3*x + phase[0]))**2 + (z - .45*np.cos(2*x + phase[1]))**2
        b = (x - .45*np.cos(3*z + phase[2]))**2 + (y - .45*np.sin(2*z + phase[0]))**2
        field = np.minimum(a, b)
    else:
        raise ValueError("unknown independent generator")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between zero and one")
    return field <= np.quantile(field, fraction)


def materialize(split: str, device: torch.device) -> dict:
    data = {k: [] for k in ("occupancy", "boundary", "distance", "identities", "geometry_ids", "families")}
    for i in range(PLAN["split_counts"][split]):
        gid, family = identity(split, i), FAMILIES[(i % 12)//3]
        occupancy = generate(gid, family, PLAN["occupied_fractions"][i % 3])
        target = base.compute_geometry_targets(occupancy, truncation=8.)
        data["occupancy"].append(occupancy)
        data["boundary"].append(target.boundary)
        data["distance"].append(target.signed_distance / 8.)
        data["geometry_ids"].append(gid)
        data["families"].append(family)
        data["identities"].append(hashlib.sha256(occupancy.tobytes()).hexdigest())
    for key in ("occupancy", "boundary", "distance"):
        data[key] = torch.as_tensor(np.stack(data[key]), dtype=torch.float32, device=device)
    return data


def artifact_contract(source_report: Path) -> dict:
    report = json.loads(source_report.read_text(encoding="utf-8"))
    digest = report.pop("report_payload_sha256")
    if digest != PLAN["source_report_sha256"] or base.payload_sha256(report) != digest:
        raise ValueError("LG2 report identity mismatch")
    return {"files": {f"joint-{s}-{kind}.pt": report["untracked_artifact_sha256"][f"joint-{s}-{kind}.pt"]
                      for s in PLAN["seeds"] for kind in ("encoder", "evaluation")},
            "states": {str(t["seed"]): {k: t[k] for k in ("initial_state_sha256", "final_state_sha256", "run_id")}
                       for t in report["training_trials"] if t["objective"] == "joint"}}


def freeze(path: Path, spec_sha: str, source_report: Path) -> dict:
    if len(spec_sha) != 40 or any(c not in "0123456789abcdef" for c in spec_sha):
        raise ValueError("full spec SHA required")
    if base.git("status", "--porcelain", "--", "theseo_anysearch/garden"):
        raise ValueError("commit executable before registration")
    payload = {"plan": PLAN, "source_commit": base.git("rev-parse", "HEAD"), "spec_commit": spec_sha,
               "spec_path": "projects/theseo-anysearch/python/perception-encoder-local-geometry-lg3.md",
               "artifacts": artifact_contract(source_report),
               "geometry_ids": {s: [identity(s, i) for i in range(n)] for s, n in PLAN["split_counts"].items()}}
    envelope = {"payload": payload, "identity_sha256": base.payload_sha256(payload)}
    base.write_json(path, envelope)
    return envelope


def read_frozen(path: Path, artifacts: Path) -> dict:
    envelope = json.loads(path.read_text(encoding="utf-8"))
    p = envelope["payload"]
    if set(envelope) != {"payload", "identity_sha256"} or base.payload_sha256(p) != envelope["identity_sha256"] or p["plan"] != PLAN:
        raise ValueError("registration mismatch")
    if p["geometry_ids"] != {s: [identity(s, i) for i in range(n)] for s, n in PLAN["split_counts"].items()}:
        raise ValueError("geometry identity mismatch")
    if base.git("diff", p["source_commit"], "--", "theseo_anysearch/garden") or base.git("ls-files", "--others", "--exclude-standard", "theseo_anysearch/garden"):
        raise ValueError("source differs from registration")
    for name, digest in p["artifacts"]["files"].items():
        if hashlib.sha256((artifacts / name).read_bytes()).hexdigest() != digest:
            raise ValueError("checkpoint artifact mismatch")
    return envelope


def absolute_pass(value: float, task: str) -> bool:
    return value >= base.PLAN["absolute_bars"][task] if task in base.TASKS[:2] else value <= base.PLAN["absolute_bars"][task]


def assess(trials: list) -> dict:
    trials = sorted(trials, key=lambda t: t["seed"])
    if [t["seed"] for t in trials] != PLAN["seeds"]:
        raise ValueError("three distinct frozen seeds required")
    integrity = all(t["integrity_ok"] and t["mask_isolation_max_abs"] == 0 and t["rank"]["near_dead_fraction"] <= .05 for t in trials)
    gates, components = [integrity], {}
    for task in base.TASKS:
        candidate = [t["statistics"][task]["trained"] for t in trials]
        scores = [base.score(x, task) for x in candidate]
        families = {family: [base.score(np.asarray(rows)[[i for i in range(48) if (i % 12)//3 == f]], task)
                             for rows in candidate] for f, family in enumerate(FAMILIES)}
        absolute = all(absolute_pass(v, task) for v in scores)
        family_pass = all(absolute_pass(v, task) for values in families.values() for v in values)
        component = {"seed_scores": scores, "absolute_pass": absolute, "family_seed_scores": families,
                     "family_absolute_pass": family_pass, "controls": {}}
        gates.extend((absolute, family_pass))
        for control in prior.CONTROLS:
            result = prior.paired_comparison(candidate, [t["statistics"][task][control] for t in trials], task)
            point_pass = all(d >= .05 if task in base.TASKS[:2] else d >= .1*b and d > 0
                             for d, b in zip(result["seed_improvements"], result["control_seed_scores"]))
            result["pass"] = point_pass and result["paired_geometry_ci95"][0] > 0
            gates.append(result["pass"])
            component["controls"][control] = result
        components[task] = component
    return {"decision": "independent_generators_confirmed" if all(gates) else "not_confirmed",
            "integrity_pass": integrity, "components": components}


def zero_shot(artifact: Path, model: nn.Module, data: dict, bank: dict) -> tuple:
    saved = torch.load(artifact, map_location=data["occupancy"].device, weights_only=True)
    features = base.extract(model, data, bank)
    stats, predictions = {}, {}
    for task in base.TASKS:
        probe = nn.Sequential(nn.Linear(8, 32), nn.SiLU(), nn.Linear(32, 1)).to(data["occupancy"].device)
        state = saved["probes"][task]
        probe.load_state_dict(state["state"])
        fitted = (probe.eval().requires_grad_(False), state["mean"], state["scale"])
        prediction = base.predict(fitted, features[task], task)
        stats[task] = base.sufficient_statistics(prediction, bank["targets"][task], task)
        predictions[task] = prediction.cpu()
    return stats, predictions


def run(registration: Path, artifacts: Path, output: Path) -> dict:
    frozen = read_frozen(registration, artifacts)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for registered confirmation")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    models = {}
    for seed in PLAN["seeds"]:
        model = base.make_encoder(seed, device)
        model.load_state_dict(torch.load(artifacts / f"joint-{seed}-encoder.pt", map_location=device, weights_only=True))
        model.eval().requires_grad_(False)
        if base.encoder_state_sha256(model) != frozen["payload"]["artifacts"]["states"][str(seed)]["final_state_sha256"]:
            raise ValueError("encoder state hash mismatch")
        models[seed] = model
    probe, evaluation = materialize("probe", device), materialize("evaluation", device)
    pb, eb = prior.query_bank(probe, "probe"), prior.query_bank(evaluation, "evaluation")
    for bank in (pb, eb):
        for family in range(4):
            indices = [i for i in range(48) if (i % 12)//3 == family]
            for task in base.TASKS[:2]:
                labels = bank["targets"][task][indices]
                if min(int(labels.sum()), int((1 - labels).sum())) < 20:
                    raise ValueError("insufficient family class support; no redraw permitted")
    raw_fit, raw_eval = base.extract(None, probe, pb), base.extract(None, evaluation, eb)
    raw = {seed: {task: base.sufficient_statistics(base.predict(base.fit_probe(raw_fit[task], pb["targets"][task], task, seed), raw_eval[task], task), eb["targets"][task], task)
                  for task in base.TASKS} for seed in PLAN["seeds"]}
    trials = []
    for seed in PLAN["seeds"]:
        state = frozen["payload"]["artifacts"]["states"][str(seed)]
        result = {"seed": seed, "objective": "joint", "updates": 0,
                  "initial_state_sha256": state["initial_state_sha256"], "pretraining_run_id": state["run_id"]}
        trial = prior.evaluate(models[seed], result, probe, evaluation, pb, eb, raw, output)
        trial["run_id"] = f"{PLAN['study_id']}-joint-seed-{seed}"
        zs_stats, predictions = zero_shot(artifacts / f"joint-{seed}-evaluation.pt", models[seed], evaluation, eb)
        for task in base.TASKS:
            trial["statistics"][task]["zero_shot_diagnostic"] = zs_stats[task]
        trial["integrity_ok"] &= base.encoder_state_sha256(models[seed]) == state["final_state_sha256"]
        torch.save({"predictions": predictions, "targets": eb["targets"]}, output / f"joint-{seed}-zero-shot.pt")
        trials.append(trial)
        print(json.dumps({"completed": trial["run_id"]}), flush=True)
    assessment = assess(trials)
    if time.monotonic() - start > PLAN["hours_cap"] * 3600:
        raise TimeoutError("LG3 compute cap exceeded")
    report = {"completed": True, "registration": frozen, "encoder_updates": 0, "trials": trials,
              "assessment": assessment, "elapsed_seconds": time.monotonic() - start,
              "corpus_hashes": {"probe": probe["identities"], "evaluation": evaluation["identities"]},
              "query_hashes": {"probe": pb["sha256"], "evaluation": eb["sha256"]},
              "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "precision": "float32"},
              "artifact_hashes": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*.pt")},
              "scope": "Frozen-feature transfer with target-domain probe fitting; zero-shot diagnostics separate; synthetic radius-eight only."}
    report["report_payload_sha256"] = base.payload_sha256(report)
    base.write_json(output / "report.json", report)
    print(json.dumps({"decision": assessment["decision"], "seconds": report["elapsed_seconds"]}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "run"))
    parser.add_argument("--registration", required=True, type=Path)
    parser.add_argument("--spec-sha")
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.command == "freeze":
        print(json.dumps(freeze(args.registration, args.spec_sha, args.source_report)))
    else:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        run(args.registration, args.artifacts, args.output)


if __name__ == "__main__":
    main()
