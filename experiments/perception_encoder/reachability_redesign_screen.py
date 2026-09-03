"""CPU proof-of-mechanism screen for the reachability denominator (#339).

NON-EVIDENTIAL. The floor here is a *linear* ridge probe on raw occupancy;
connectivity is nonlinear, so that floor is too weak and every reported gap is
optimistic. This screen establishes only that occlusion *can* create headroom in
principle. Do not adopt any variant on these numbers. The real comparison needs
the v2r2 R0 feasibility audit and the R2 control ladder (identical probe over
raw, capacity-matched nonlinear raw, over-capacity reference) on the real corpus.
See docs/perception-encoder-calibration-revision-work-plan.md.

Measures, on a deterministic graded-occlusion fixture corpus, the floor/ceiling
gap each redesign option would give the ``reachability`` pilot-score component:

- V0  current: pairwise AUPRC over all pairs
- A   occlusion-aware: pairwise AUPRC on the hard strata (occlusion span >= 1)
- B1  excess over raw on the hard strata (same gate as A, framed as a denominator)
- B2  PVI gain (embedding vs coordinates-only null) for the reachability label
- C   connectivity-structure: 1 - normalized Variation of Information of the
      predicted free-space component labeling

Emits ``reachability-redesign-screen.json`` with a per-variant table and a
``recommendation`` of ``non_evidential:<mechanism>:defer_to_v2r2_R0`` - it cannot
recommend adoption. Disposition is ``retain`` (infrastructure kept, result not a
finding).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from scipy import ndimage

from theseo_anysearch.garden.evaluation.ceilings import (
    DEFAULT_K,
    _loo_neighbour_indices,
    classification_metric_ceiling,
)
from theseo_anysearch.garden.evaluation.metrics import (
    binary_ranking_metrics,
    variation_of_information,
)
from theseo_anysearch.garden.evaluation.reachability_variants import (
    coordinates_null,
    pair_matrix,
    raw_observed_feature,
    rich_completed_feature,
    sample_span_annotated_pairs,
)
from theseo_anysearch.garden.evaluation.triviality import assess_triviality
from theseo_anysearch.garden.pilots.reachability_fixtures import occlusion_corpus

GATE = 0.10
_SIX = ndimage.generate_binary_structure(3, 1)


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - screen still runs outside a repo
        return "unknown"


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    ).hexdigest()


def _ridge_scores(train_x, train_y, eval_x) -> np.ndarray:
    train_x = np.column_stack((np.ones(len(train_x)), train_x))
    eval_x = np.column_stack((np.ones(len(eval_x)), eval_x))
    penalty = 1e-3 * np.eye(train_x.shape[1])
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(train_x.T @ train_x + penalty, train_x.T @ train_y.astype(float))
    return 1.0 / (1.0 + np.exp(-np.clip(eval_x @ weights, -30, 30)))


def _stratum(span: int) -> str:
    if span < 0:
        return "unreachable"
    if span == 0:
        return "visible"
    if span <= 2:
        return "occluded_1_2"
    if span <= 5:
        return "occluded_3_5"
    return "occluded_6plus"


def _collect(corpus, feature_fn, *, seed):
    mats, labels, spans, gids = [], [], [], []
    for geometry in corpus:
        # uniform draw; the screen stratifies in analysis, not in sampling
        sample = sample_span_annotated_pairs(geometry, count=32, seed=seed)
        mats.append(pair_matrix(feature_fn, geometry, sample))
        labels.append(sample.reachable)
        spans.append(sample.occlusion_span)
        gids.extend(sample.geometry_ids)
    return (
        np.vstack(mats),
        np.concatenate(labels),
        np.concatenate(spans),
        np.asarray(gids),
    )


def _auprc_gap(train, evaluation):
    raw_scores = _ridge_scores(train["raw"], train["y"], evaluation["raw"])
    floor = binary_ranking_metrics(raw_scores, evaluation["y"]).auprc
    ceiling = classification_metric_ceiling(
        evaluation["rich"], evaluation["y"], metric="reachability_auprc"
    ).value
    return float(floor), float(ceiling), float(ceiling - floor)


def _structure_gap(corpus_eval) -> tuple[float, float, float]:
    raw_vis, ceil_vis = [], []
    for geometry in corpus_eval:
        mask = geometry.observed_free
        truth = geometry.free_component_labels[mask]
        observed_labels, _ = ndimage.label(geometry.observed_free, structure=_SIX)
        raw_pred = observed_labels[mask]
        raw_vis.append(1.0 - variation_of_information(raw_pred, truth))
        coords = np.argwhere(mask)
        rich = np.stack([rich_completed_feature(geometry, c) for c in coords])
        if len(rich) <= DEFAULT_K + 1:
            continue
        neighbours = _loo_neighbour_indices(rich, DEFAULT_K)
        knn_pred = np.array(
            [np.bincount(truth[row]).argmax() for row in neighbours]
        )
        ceil_vis.append(1.0 - variation_of_information(knn_pred, truth))
    floor = float(np.mean(raw_vis))
    ceiling = float(np.mean(ceil_vis))
    return floor, ceiling, ceiling - floor


def run_screen(*, n: int = 48, seed: int = 20260903) -> dict:
    corpus = occlusion_corpus(n=n, seed=seed)
    split = int(0.66 * n)
    train_geoms, eval_geoms = corpus[:split], corpus[split:]

    raw_tr = _collect(train_geoms, raw_observed_feature, seed=seed)
    raw_ev = _collect(eval_geoms, raw_observed_feature, seed=seed + 1)
    rich_ev = _collect(eval_geoms, rich_completed_feature, seed=seed + 1)
    null_ev = _collect(eval_geoms, coordinates_null, seed=seed + 1)

    train = {"raw": raw_tr[0], "y": raw_tr[1]}
    evaluation = {"raw": raw_ev[0], "rich": rich_ev[0], "null": null_ev[0], "y": raw_ev[1]}
    spans = raw_ev[2]

    variants: dict[str, dict] = {}

    floor, ceiling, gap = _auprc_gap(train, evaluation)
    variants["V0_pairwise_auprc"] = {
        "metric": "pairwise AUPRC (all pairs)",
        "floor": floor,
        "ceiling": ceiling,
        "gap": gap,
        "opens_gate": gap >= GATE,
    }

    hard = spans >= 1
    hard_or_neg = hard | ~evaluation["y"]
    raw_scores = _ridge_scores(train["raw"], train["y"], evaluation["raw"][hard_or_neg])
    a_floor = binary_ranking_metrics(raw_scores, evaluation["y"][hard_or_neg]).auprc
    a_ceiling = classification_metric_ceiling(
        evaluation["rich"][hard_or_neg], evaluation["y"][hard_or_neg], metric="reachability_auprc"
    ).value
    per_stratum = {}
    for name in ("visible", "occluded_1_2", "occluded_3_5", "occluded_6plus"):
        sel = np.array([_stratum(s) == name for s in spans]) | ~evaluation["y"]
        if evaluation["y"][sel].any() and (~evaluation["y"][sel]).any() and sel.sum() > DEFAULT_K + 1:
            f = binary_ranking_metrics(
                _ridge_scores(train["raw"], train["y"], evaluation["raw"][sel]),
                evaluation["y"][sel],
            ).auprc
            c = classification_metric_ceiling(
                evaluation["rich"][sel], evaluation["y"][sel], metric="reachability_auprc"
            ).value
            per_stratum[name] = {"floor": float(f), "ceiling": float(c), "gap": float(c - f)}
    variants["A_occlusion_hard_strata"] = {
        "metric": "pairwise AUPRC (occlusion span >= 1 + negatives)",
        "floor": float(a_floor),
        "ceiling": float(a_ceiling),
        "gap": float(a_ceiling - a_floor),
        "opens_gate": (a_ceiling - a_floor) >= GATE,
        "per_stratum": per_stratum,
    }
    variants["B1_excess_over_raw"] = {
        "metric": "denominator = ceiling - raw on the hard strata (gate == A)",
        "floor": float(a_floor),
        "ceiling": float(a_ceiling),
        "gap": float(a_ceiling - a_floor),
        "opens_gate": (a_ceiling - a_floor) >= GATE,
        "note": "opens iff A opens; frames the component as gain-over-raw",
    }

    pvi = assess_triviality(
        evaluation["rich"],
        evaluation["null"],
        evaluation["y"],
        task_type="binary",
        null_input="coordinates_only",
        min_pvi_gain=GATE,
        seed=seed,
    )
    variants["B2_pvi_gain"] = {
        "metric": "PVI gain (bits/sample), rich vs coordinates-only null",
        "floor": float(pvi.pvi_null),
        "ceiling": float(pvi.pvi_embedding),
        "gap": float(pvi.pvi_gain),
        "opens_gate": bool(pvi.pvi_gain >= GATE),
        "mdl_embedding_bits": pvi.mdl_embedding_bits,
        "mdl_null_bits": pvi.mdl_null_bits,
    }

    s_floor, s_ceiling, s_gap = _structure_gap(eval_geoms)
    variants["C_component_structure"] = {
        "metric": "1 - normalized Variation of Information of the component labeling",
        "floor": s_floor,
        "ceiling": s_ceiling,
        "gap": s_gap,
        "opens_gate": s_gap >= GATE,
    }

    # V0 is the control; B1 is A framed as a denominator. Rank the genuine
    # redesigns (A, B2, C) by gap among those that open the gate.
    candidates = {
        name: variants[name]["gap"]
        for name in ("A_occlusion_hard_strata", "B2_pvi_gain", "C_component_structure")
        if variants[name]["opens_gate"]
    }
    hard_fraction = float(np.mean(spans >= 1))
    # This screen is non-evidential: it cannot recommend adoption. It only
    # records whether the occlusion mechanism produced any headroom at all.
    mechanism = "occlusion_creates_headroom" if candidates else "no_headroom_observed"
    recommendation = f"non_evidential:{mechanism}:defer_to_v2r2_R0"

    report = {
        "issue": 339,
        "non_evidential": True,
        "non_evidential_reason": (
            "floor is a linear ridge probe on raw occupancy; connectivity is "
            "nonlinear so the floor is too weak and every gap is optimistic. "
            "Proof-of-mechanism only. Real comparison requires the v2r2 R0 audit "
            "and the R2 control ladder on the real corpus."
        ),
        "disposition": "retain",
        "code_sha": _git_sha(),
        "gate": GATE,
        "corpus": {"geometries": n, "seed": seed, "train": split, "evaluation": n - split},
        "evaluation_pairs": int(len(evaluation["y"])),
        "reachable_fraction": float(np.mean(evaluation["y"])),
        "hard_pair_fraction": hard_fraction,
        "variants": variants,
        "recommendation": recommendation,
        "caveats": [
            "Fixture corpus is occlusion-rich by construction; V0 opening here does "
            "not mean V0 opens on the real v2r1 corpus.",
            "rich_completed_feature is an oracle for completion, so ceilings are an "
            "upper bound on what a real encoder achieves (appropriate for a "
            "model-free ceiling, optimistic for a candidate).",
            "A and B2 both require the real corpus to contain a material fraction of "
            "occluded-path pairs (hard_pair_fraction >= 0.10); otherwise defer.",
            "C is inconclusive on this corpus (VI range too small), not rejected.",
        ],
        "hardware": {"python": platform.python_version()},
    }
    report["report_payload_sha256"] = _canonical_sha(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/perception_encoder/results/v2r1_reachability/reachability-redesign-screen.json"),
    )
    args = parser.parse_args()
    report = run_screen()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"recommendation": report["recommendation"], "sha256": report["report_payload_sha256"]}))
    for name, v in report["variants"].items():
        print(f"  {name:<28} floor={v['floor']:.3f} ceiling={v['ceiling']:.3f} gap={v['gap']:+.3f} opens={v['opens_gate']}")


if __name__ == "__main__":
    main()
