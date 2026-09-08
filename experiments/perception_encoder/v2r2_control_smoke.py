"""Exercise R0 control plumbing on a toy corpus; never an evidential R0 run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from theseo_anysearch.garden.pilots.io import payload_sha256
from theseo_anysearch.garden.pilots.v2r2_controls import (
    R0ControlRecipe, example_from_completions, fit_r0_controls,
)


def development_examples(count: int, *, offset: int, configuration: str):
    """Full-3D toy arrays for device/data-path validation, not identifiability."""

    rows = []
    for index in range(offset, offset + count):
        rng = np.random.default_rng(340_000 + index)
        observed = rng.random((17, 17, 17)) < 0.05
        unknown = np.zeros_like(observed)
        unknown[7:10] = True
        observed[unknown] = False
        observed[0, 0, 0] = observed[16, 16, 16] = False
        completions = np.stack([observed] * 16)
        # A deliberately trivial balanced label construction tests connectivity
        # extraction and fitting; it does not model real conditional completion.
        completions[8:, 8] = True
        rows.append(example_from_completions(
            context_id=f"development-r0-control-{index}", geometry_id=f"development-r0-control-{index}",
            generator_configuration=configuration, bootstrap_stratum="toy:low", stratum="3-5",
            observed_occupancy=observed, unknown=unknown,
            start=(0, 0, 0), goal=(16, 16, 16), completions=completions,
            forbidden_features=(float(index), float(configuration == "B")),
        ))
    return tuple(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite a development smoke report")
    root = Path(__file__).resolve().parents[2]
    code_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    recipe = R0ControlRecipe(
        updates=args.updates, learning_rate=0.001, weight_decay=0.01,
        visible_hidden_width=64, forbidden_hidden_width=16, seed=340,
        maximum_wall_seconds=120, optimizer="adamw_full_batch",
        checkpoint="last_update_no_evaluation_selection", weighting="equal_geometry_then_equal_context",
        geometric_prior="linear_observed_density_unknown_density_l1_separation",
    )
    training = development_examples(32, offset=0, configuration="A")
    domains = {}
    for domain, offset, configuration in (("in_domain", 32, "A"), ("heldout_generator", 44, "B")):
        fitted = fit_r0_controls(
            training, development_examples(12, offset=offset, configuration=configuration),
            fold_id=f"development-{domain}", domain=domain, recipe=recipe, device=args.device,
        )
        domains[domain] = {
            "fold": fitted.fold.model_dump(mode="json"), "fit_artifact": fitted.artifact,
            "resources": fitted.resources, "evaluation_contexts": len(fitted.contexts),
            "prediction_payload_sha256": payload_sha256([row.model_dump(mode="json") for row in fitted.contexts]),
        }
    source_files = (
        "theseo_anysearch/garden/pilots/v2r2_audit.py",
        "theseo_anysearch/garden/pilots/v2r2_controls.py",
        "experiments/perception_encoder/v2r2_control_smoke.py",
    )
    report = {
        "kind": "development_control_smoke", "non_evidential": True,
        "authorizes_comparative_run": False, "r0_executed": False,
        "candidate_training_updates": 0, "status": "passed",
        "code_sha": code_sha, "dirty_worktree": dirty,
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files},
        "shape": [17, 17, 17], "training_geometries": 32,
        "completions_per_context": 16, "domains": domains,
        "limitations": "Toy balanced completions and nominal generator configurations exercise plumbing only; no identifiability, transfer-quality or encoder-quality claim.",
    }
    report["report_payload_sha256"] = payload_sha256(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"report": str(args.output), "status": "passed", "non_evidential": True,
        "resources": {name: value["resources"] for name, value in domains.items()}}, indent=2))


if __name__ == "__main__":
    main()
