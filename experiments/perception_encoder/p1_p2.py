"""Execute the frozen P1/P2 perception-encoder trial matrices."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import yaml

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from theseo_anysearch.garden.evaluation.statistics import (
    paired_stratified_geometry_bootstrap,
)
from theseo_anysearch.garden.masking import DenseMaskAwareEncoder
from theseo_anysearch.garden.models.outputs import VoxelLevel
from theseo_anysearch.garden.pilots.benchmark import (
    COMPONENTS,
    ProbeProtocol,
    evaluate_frozen_representation,
)
from theseo_anysearch.garden.pilots.comparative import (
    ComparativeTrialConfig,
    build_bundle_modules,
    train_comparative_trial,
)
from theseo_anysearch.garden.pilots.contracts import FrozenPreregistration, ScoreAnchor
from theseo_anysearch.garden.pilots.corpus import make_pilot_observation
from theseo_anysearch.garden.pilots.io import read_contract
from theseo_anysearch.garden.pilots.runner import pilot_geometry_records
from theseo_anysearch.garden.splits import GeometryDescriptor


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "uri": path.as_posix(),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "media_type": "application/json",
    }


def _load_context(
    config_path: Path,
) -> tuple[
    dict[str, Any],
    FrozenPreregistration,
    list[GeometryDescriptor],
    list[GeometryDescriptor],
    ProbeProtocol,
]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    preregistration = read_contract(
        Path(config["preregistration"]), FrozenPreregistration
    )
    by_id = {record.geometry_id: record for record in pilot_geometry_records()}
    train = [
        by_id[geometry_id]
        for geometry_id in preregistration.pools["pilot_train"].geometry_ids
    ]
    dev = [
        by_id[geometry_id]
        for geometry_id in preregistration.pools["pilot_dev_early"].geometry_ids
    ]
    protocol = ProbeProtocol(**config["probe"])
    return config, preregistration, train, dev, protocol


def _latency_profile(
    encoder: torch.nn.Module,
    descriptor: GeometryDescriptor,
    *,
    device: torch.device,
) -> dict[str, float]:
    observation = make_pilot_observation(descriptor, 3, radius=16)
    level = VoxelLevel.from_occupancy(
        torch.from_numpy(observation.occupancy[None]).to(
            device=device, dtype=torch.float32
        ),
        unknown_mask=torch.from_numpy(observation.unknown_mask[None]).to(device),
    )
    hidden = torch.zeros_like(level.validity_mask)

    def forward() -> object:
        return (
            encoder(level, hidden)
            if isinstance(encoder, DenseMaskAwareEncoder)
            else encoder(level)
        )

    encoder.eval()
    with torch.no_grad():
        for _ in range(20):
            forward()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(100)]
        for start, end in zip(starts, ends):
            start.record()
            forward()
            end.record()
        torch.cuda.synchronize(device)
    values = np.asarray([start.elapsed_time(end) for start, end in zip(starts, ends)])
    return {
        "latency_p50_ms": float(np.quantile(values, 0.50)),
        "latency_p95_ms": float(np.quantile(values, 0.95)),
    }


def _normalized_component(
    name: str, value: float, anchor: ScoreAnchor
) -> float:
    if anchor.higher_is_better:
        return (value - anchor.floor) / (anchor.ceiling - anchor.floor)
    return (anchor.floor - value) / (anchor.floor - anchor.ceiling)


def _geometry_scores(
    rows: Sequence[dict[str, object]], anchors: dict[str, ScoreAnchor]
) -> tuple[list[float], list[str], list[str], list[str]]:
    scores: list[float] = []
    geometry_ids: list[str] = []
    families: list[str] = []
    bands: list[str] = []
    for row in rows:
        components = row["components"]
        scores.append(
            float(
                np.mean(
                    [
                        _normalized_component(name, components[name], anchors[name])
                        for name in COMPONENTS
                    ]
                )
            )
        )
        geometry_ids.append(str(row["geometry_id"]))
        families.append(str(row["family"]))
        bands.append(str(row["occupancy_band"]))
    return scores, geometry_ids, families, bands


def _bootstrap_summary(
    candidate_rows: Sequence[dict[str, object]],
    reference_rows: Sequence[dict[str, object]],
    anchors: dict[str, ScoreAnchor],
    *,
    seed: int,
) -> dict[str, object]:
    candidate = _geometry_scores(candidate_rows, anchors)
    reference = _geometry_scores(reference_rows, anchors)
    if candidate[1:] != reference[1:]:
        raise ValueError("paired bootstrap geometry identities or strata differ")
    result = paired_stratified_geometry_bootstrap(
        candidate[0],
        reference[0],
        geometry_ids=candidate[1],
        geometry_families=candidate[2],
        occupancy_bands=candidate[3],
        resamples=10_000,
        seed=seed,
    )
    return {
        "mean_difference": result.mean_difference,
        "lower_95": result.lower_95,
        "upper_95": result.upper_95,
        "probability_of_improvement": result.probability_of_improvement,
        "resamples": result.resamples,
        "seed": result.seed,
    }


def _final_evaluation(trial: dict[str, object]) -> dict[str, object]:
    checkpoints = trial["training"]["checkpoint_results"]
    if not checkpoints:
        raise ValueError("trial has no frozen-checkpoint evaluations")
    return checkpoints[-1]


def _random_key(bundle: str, seed: int) -> str:
    family = "mask_aware" if bundle in {"T1", "T3"} else "dense_residual"
    return f"random-{family}-seed-{seed}"


def _random_baseline(
    bundle: str,
    seed: int,
    train: Sequence[GeometryDescriptor],
    dev: Sequence[GeometryDescriptor],
    anchors: dict[str, ScoreAnchor],
    protocol: ProbeProtocol,
    raw_dir: Path,
    device: torch.device,
) -> tuple[dict[str, object], Path]:
    path = raw_dir / f"{_random_key(bundle, seed)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), path
    trial_config = ComparativeTrialConfig(bundle, 1e-4, seed, 1)
    encoder, _, _ = build_bundle_modules(trial_config, device=device)
    evaluation = evaluate_frozen_representation(
        encoder,
        train,
        dev,
        anchors,
        protocol=protocol,
        seed=seed,
        device=device,
        final=True,
        include_controls=False,
    )
    payload = {"key": _random_key(bundle, seed), "evaluation": evaluation}
    _write_json(path, payload)
    del encoder
    torch.cuda.empty_cache()
    return payload, path


def _execute_trial(
    trial_id: str,
    trial_config: ComparativeTrialConfig,
    train: Sequence[GeometryDescriptor],
    dev: Sequence[GeometryDescriptor],
    anchors: dict[str, ScoreAnchor],
    protocol: ProbeProtocol,
    raw_dir: Path,
    device: torch.device,
) -> tuple[dict[str, object], Path]:
    path = raw_dir / f"{trial_id}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), path

    def evaluate(encoder: torch.nn.Module, update: int) -> dict[str, object]:
        return evaluate_frozen_representation(
            encoder,
            train,
            dev,
            anchors,
            protocol=protocol,
            seed=trial_config.seed,
            device=device,
            final=update == trial_config.updates,
        )

    encoder, training = train_comparative_trial(
        train, trial_config, device=device, evaluate=evaluate
    )
    payload: dict[str, object] = {
        "trial_id": trial_id,
        "config": asdict(trial_config),
        "training": asdict(training),
    }
    if training.status == "completed":
        payload["inference"] = _latency_profile(encoder, dev[0], device=device)
    _write_json(path, payload)
    del encoder
    gc.collect()
    torch.cuda.empty_cache()
    return payload, path


def _component_improvements(
    candidate: dict[str, float], reference: dict[str, float]
) -> dict[str, float]:
    return {
        name: (
            candidate[name] - reference[name]
            if name not in {"clearance_nmae", "geodesic_nmae"}
            else reference[name] - candidate[name]
        )
        for name in COMPONENTS
    }


def _p1_trial_summary(
    trial: dict[str, object],
    baseline: dict[str, object],
    anchors: dict[str, ScoreAnchor],
) -> dict[str, object]:
    final = _final_evaluation(trial)
    random = baseline["evaluation"]
    real = final["real"]
    random_real = random["real"]
    improvements = _component_improvements(
        real["components"], random_real["components"]
    )
    collapse = final["collapse"]
    controls = final["controls"]
    vetoes: list[str] = []
    if collapse["effective_rank_fraction"] < 0.25:
        vetoes.append("effective_rank_fraction_below_0_25")
    if collapse["near_dead_dimensions_fraction"] > 0.05:
        vetoes.append("near_dead_dimensions_fraction_above_0_05")
    if not controls["passes_selectivity"]:
        vetoes.append("control_target_selectivity_below_0_05")
    if not controls["passes_embedding_necessity"]:
        vetoes.append("embedding_necessity_below_0_05")
    if real["false_open_rate"] > 0.05:
        vetoes.append("false_open_rate_above_0_05")
    if sum(value > 0 for value in improvements.values()) < 3:
        vetoes.append("fewer_than_three_components_improve_over_random")
    bootstrap = _bootstrap_summary(
        real["per_geometry"], random_real["per_geometry"], anchors, seed=0
    )
    return {
        "pilot_score": real["pilot_score"],
        "components": real["components"],
        "component_improvements_over_random": improvements,
        "random_pilot_score": random_real["pilot_score"],
        "bootstrap_vs_random": bootstrap,
        "collapse": collapse,
        "controls": controls,
        "false_open_rate": real["false_open_rate"],
        "vetoes": vetoes,
        "eligible": not vetoes,
    }


def run_p1(config_path: Path, raw_dir: Path, report_path: Path) -> dict[str, object]:
    config, preregistration, train, dev, protocol = _load_context(config_path)
    device = torch.device("cuda:0")
    started = time.perf_counter()
    raw_dir.mkdir(parents=True, exist_ok=True)
    random_payloads: dict[str, dict[str, object]] = {}
    random_artifacts: dict[str, dict[str, object]] = {}
    for bundle in config["p1"]["bundles"]:
        for seed in config["p1"]["seeds"]:
            key = _random_key(bundle, seed)
            if key in random_payloads:
                continue
            payload, path = _random_baseline(
                bundle,
                seed,
                train,
                dev,
                preregistration.score_anchors,
                protocol,
                raw_dir,
                device,
            )
            random_payloads[key] = payload
            random_artifacts[key] = _artifact(path)

    summaries: dict[str, dict[str, object]] = {}
    artifacts: dict[str, dict[str, object]] = {}
    for bundle in config["p1"]["bundles"]:
        for learning_rate in config["p1"]["learning_rates"]:
            for seed in config["p1"]["seeds"]:
                trial_id = f"p1-{bundle}-lr-{learning_rate:g}-seed-{seed}"
                trial_config = ComparativeTrialConfig(
                    bundle,
                    float(learning_rate),
                    int(seed),
                    int(config["p1"]["updates"]),
                    batch_size=int(config["batch_size"]),
                )
                trial, path = _execute_trial(
                    trial_id,
                    trial_config,
                    train,
                    dev,
                    preregistration.score_anchors,
                    protocol,
                    raw_dir,
                    device,
                )
                artifacts[trial_id] = _artifact(path)
                if trial["training"]["status"] != "completed":
                    summaries[trial_id] = {
                        "eligible": False,
                        "vetoes": ["trial_failed"],
                        "error": trial["training"]["error"],
                    }
                    continue
                summaries[trial_id] = _p1_trial_summary(
                    trial,
                    random_payloads[_random_key(bundle, seed)],
                    preregistration.score_anchors,
                )

    eligible_bundles = []
    for bundle in config["p1"]["bundles"]:
        bundle_trials = [
            value for key, value in summaries.items() if key.startswith(f"p1-{bundle}-")
        ]
        if bundle_trials and all(value["eligible"] for value in bundle_trials):
            eligible_bundles.append(bundle)
    bundle_scores = {
        bundle: float(
            np.mean(
                [
                    value["pilot_score"]
                    for key, value in summaries.items()
                    if key.startswith(f"p1-{bundle}-") and "pilot_score" in value
                ]
            )
        )
        for bundle in config["p1"]["bundles"]
    }
    retained = sorted(eligible_bundles, key=lambda name: bundle_scores[name], reverse=True)[:2]
    report = {
        "issue": 278,
        "pilot": "P1",
        "status": "completed",
        "integration_base_sha": _git("merge-base", "HEAD", "origin/exp/perception-encoder"),
        "code_sha": _git("rev-parse", "HEAD"),
        "spec_commit": config["spec_commit"],
        "preregistration_sha256": hashlib.sha256(
            Path(config["preregistration"]).read_bytes()
        ).hexdigest(),
        "resolved_config_sha256": _sha256(config_path),
        "dataset_query_hashes": {
            name: {
                "assignment_sha256": pool.assignment_sha256,
                "query_sha256": pool.query_sha256,
            }
            for name, pool in preregistration.pools.items()
            if name in {"pilot_train", "pilot_dev_early"}
        },
        "hardware": {
            "device": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "trial_counts": {
            "completed": sum("pilot_score" in value for value in summaries.values()),
            "failed": sum("pilot_score" not in value for value in summaries.values()),
            "skipped": 0,
            "oom": sum(
                "out of memory" in str(value.get("error", "")).lower()
                for value in summaries.values()
            ),
            "cap": config["p1"]["trial_cap"],
        },
        "trial_summaries": summaries,
        "artifact_references": {**random_artifacts, **artifacts},
        "bundle_mean_scores": bundle_scores,
        "decision_record": {
            "decision": "tie" if retained else "no_viable_direction",
            "retained": retained,
            "rejected": [name for name in config["p1"]["bundles"] if name not in retained],
            "rejection_rules": {
                name: sorted(
                    {
                        veto
                        for key, value in summaries.items()
                        if key.startswith(f"p1-{name}-")
                        for veto in value["vetoes"]
                    }
                )
                for name in config["p1"]["bundles"]
                if name not in retained
            },
            "next_pilot": "P2" if retained else None,
            "disposition": "retain" if retained else "reject",
        },
        "validity_limits": config["validity_limits"],
        "accelerator_hours": (time.perf_counter() - started) / 3600,
    }
    _write_json(report_path, report)
    return report


def _variant_config(
    bundle: str,
    learning_rate: float,
    seed: int,
    updates: int,
    batch_size: int,
    values: dict[str, object],
) -> ComparativeTrialConfig:
    return ComparativeTrialConfig(
        bundle,
        learning_rate,
        seed,
        updates,
        batch_size=batch_size,
        cube_rotations=True,
        **values,
    )


def run_p2(
    config_path: Path,
    p1_report_path: Path,
    raw_dir: Path,
    report_path: Path,
) -> dict[str, object]:
    config, preregistration, train, dev, protocol = _load_context(config_path)
    p1 = json.loads(p1_report_path.read_text(encoding="utf-8"))
    retained_bundles = p1["decision_record"]["retained"]
    if not retained_bundles:
        report = {
            "issue": 278,
            "pilot": "P2",
            "status": "blocked",
            "decision_record": {
                "decision": "no_viable_direction",
                "retained": [],
                "rejected": [],
                "rejection_rules": {"P2": ["P1_retained_no_bundles"]},
                "next_pilot": None,
                "disposition": "reject",
            },
            "validity_limits": config["validity_limits"],
        }
        _write_json(report_path, report)
        return report

    device = torch.device("cuda:0")
    started = time.perf_counter()
    summaries: dict[str, dict[str, object]] = {}
    artifacts: dict[str, dict[str, object]] = {}
    trials: dict[str, dict[str, object]] = {}
    for bundle in retained_bundles:
        variants = config["p2"]["variants"][bundle]
        for variant, values in variants.items():
            for learning_rate in config["p2"]["learning_rates"]:
                trial_id = f"p2-{bundle}-{variant}-lr-{learning_rate:g}"
                trial_config = _variant_config(
                    bundle,
                    float(learning_rate),
                    int(config["p2"]["seed"]),
                    int(config["p2"]["updates"]),
                    int(config["batch_size"]),
                    values,
                )
                trial, path = _execute_trial(
                    trial_id,
                    trial_config,
                    train,
                    dev,
                    preregistration.score_anchors,
                    protocol,
                    raw_dir,
                    device,
                )
                trials[trial_id] = trial
                artifacts[trial_id] = _artifact(path)
                if trial["training"]["status"] == "completed":
                    final = _final_evaluation(trial)
                    summaries[trial_id] = {
                        "pilot_score": final["real"]["pilot_score"],
                        "components": final["real"]["components"],
                        "collapse": final["collapse"],
                        "controls": final["controls"],
                        "false_open_rate": final["real"]["false_open_rate"],
                    }
                else:
                    summaries[trial_id] = {"error": trial["training"]["error"]}

    selected: dict[str, str] = {}
    variant_decisions: dict[str, object] = {}
    for bundle in retained_bundles:
        variants = config["p2"]["variants"][bundle]
        baseline_ids = {
            float(rate): f"p2-{bundle}-baseline-lr-{rate:g}"
            for rate in config["p2"]["learning_rates"]
        }
        promoted: list[str] = []
        comparisons: dict[str, object] = {}
        for variant in variants:
            if variant == "baseline":
                continue
            by_rate: dict[str, object] = {}
            for rate in config["p2"]["learning_rates"]:
                candidate_id = f"p2-{bundle}-{variant}-lr-{rate:g}"
                baseline_id = baseline_ids[float(rate)]
                candidate = _final_evaluation(trials[candidate_id])["real"]
                baseline = _final_evaluation(trials[baseline_id])["real"]
                bootstrap = _bootstrap_summary(
                    candidate["per_geometry"],
                    baseline["per_geometry"],
                    preregistration.score_anchors,
                    seed=0,
                )
                improvements = _component_improvements(
                    candidate["components"], baseline["components"]
                )
                by_rate[str(rate)] = {
                    "bootstrap": bootstrap,
                    "component_improvements": improvements,
                }
            effects = [
                value["bootstrap"]["mean_difference"] for value in by_rate.values()
            ]
            decisive = any(
                value["bootstrap"]["lower_95"] > 0
                and value["bootstrap"]["mean_difference"] >= 0.03
                for value in by_rate.values()
            ) and all(value > 0 for value in effects)
            broad_gain = all(value >= 0.05 for value in effects) and all(
                min(value["component_improvements"].values()) >= -0.02
                for value in by_rate.values()
            )
            if decisive or broad_gain:
                promoted.append(variant)
            comparisons[variant] = {
                "by_learning_rate": by_rate,
                "decisive_consistent": decisive,
                "broad_gain_consistent": broad_gain,
            }
        choice = (
            max(
                promoted,
                key=lambda name: np.mean(
                    [
                        summaries[f"p2-{bundle}-{name}-lr-{rate:g}"]["pilot_score"]
                        for rate in config["p2"]["learning_rates"]
                    ]
                ),
            )
            if promoted
            else "baseline"
        )
        selected[bundle] = choice
        variant_decisions[bundle] = {
            "selected": choice,
            "comparisons": comparisons,
            "unresolved_learning_rate_interaction": not promoted,
        }

    bundle_scores = {
        bundle: float(
            np.mean(
                [
                    summaries[f"p2-{bundle}-{variant}-lr-{rate:g}"]["pilot_score"]
                    for rate in config["p2"]["learning_rates"]
                ]
            )
        )
        for bundle, variant in selected.items()
    }
    ordered = sorted(bundle_scores, key=bundle_scores.get, reverse=True)
    report = {
        "issue": 278,
        "pilot": "P2",
        "status": "completed",
        "integration_base_sha": _git("merge-base", "HEAD", "origin/exp/perception-encoder"),
        "code_sha": _git("rev-parse", "HEAD"),
        "spec_commit": config["spec_commit"],
        "resolved_config_sha256": _sha256(config_path),
        "p1_report_sha256": _sha256(p1_report_path),
        "trial_counts": {
            "completed": sum("pilot_score" in value for value in summaries.values()),
            "failed": sum("pilot_score" not in value for value in summaries.values()),
            "skipped": config["p2"]["trial_cap"] - len(summaries),
            "oom": sum(
                "out of memory" in str(value.get("error", "")).lower()
                for value in summaries.values()
            ),
            "cap": config["p2"]["trial_cap"],
        },
        "trial_summaries": summaries,
        "artifact_references": artifacts,
        "variant_decisions": variant_decisions,
        "bundle_mean_scores": bundle_scores,
        "decision_record": {
            "decision": "winner" if ordered else "no_viable_direction",
            "retained": [f"{bundle}:{selected[bundle]}" for bundle in ordered],
            "rejected": [],
            "rejection_rules": {},
            "next_pilot": "P4" if ordered else None,
            "disposition": "retain" if ordered else "reject",
            "primary": f"{ordered[0]}:{selected[ordered[0]]}" if ordered else None,
            "fallback": (
                f"{ordered[1]}:{selected[ordered[1]]}" if len(ordered) > 1 else None
            ),
        },
        "validity_limits": config["validity_limits"],
        "accelerator_hours": (time.perf_counter() - started) / 3600,
    }
    _write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("P1", "P2", "all"), default="all")
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("runtime/perception_encoder/p1_p2")
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("experiments/perception_encoder/results"),
    )
    arguments = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("P1/P2 execution requires CUDA")
    p1_path = arguments.results_dir / "p1-report.json"
    if arguments.stage in {"P1", "all"}:
        report = run_p1(arguments.config, arguments.raw_dir, p1_path)
        print(json.dumps(report["decision_record"], sort_keys=True))
    if arguments.stage in {"P2", "all"}:
        report = run_p2(
            arguments.config,
            p1_path,
            arguments.raw_dir,
            arguments.results_dir / "p2-report.json",
        )
        print(json.dumps(report["decision_record"], sort_keys=True))


if __name__ == "__main__":
    main()
