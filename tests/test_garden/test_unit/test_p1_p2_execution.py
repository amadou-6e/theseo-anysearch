"""Contract tests for P1/P2 execution configuration and decision helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


SCRIPT = (
    Path(__file__).parents[3]
    / "experiments"
    / "perception_encoder"
    / "p1_p2.py"
)
CONFIG = SCRIPT.with_name("p1-p2-config.yaml")
SPEC = importlib.util.spec_from_file_location("p1_p2_execution", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_p1_matrix_is_exactly_sixteen_trials() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    p1 = config["p1"]
    assert len(p1["bundles"]) * len(p1["learning_rates"]) * len(p1["seeds"]) == 16
    assert p1["updates"] == 2_000
    assert p1["trial_cap"] == 16


def test_p2_matrix_caps_two_retained_bundles_at_twelve_trials() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert all(len(variants) == 3 for variants in config["p2"]["variants"].values())
    assert 2 * 3 * len(config["p2"]["learning_rates"]) == config["p2"]["trial_cap"]
    assert config["p2"]["updates"] == 3_000


def test_component_improvements_reverse_error_direction() -> None:
    candidate = {
        "occupied_iou": 0.8,
        "boundary_f1": 0.7,
        "clearance_nmae": 0.2,
        "reachability_auprc": 0.9,
        "geodesic_nmae": 0.1,
    }
    reference = {
        "occupied_iou": 0.6,
        "boundary_f1": 0.5,
        "clearance_nmae": 0.3,
        "reachability_auprc": 0.7,
        "geodesic_nmae": 0.4,
    }
    improvements = MODULE._component_improvements(candidate, reference)
    assert improvements == pytest.approx(
        {
            "occupied_iou": 0.2,
            "boundary_f1": 0.2,
            "clearance_nmae": 0.1,
            "reachability_auprc": 0.2,
            "geodesic_nmae": 0.3,
        }
    )


def test_p2_variants_preserve_locked_rotation_and_optimizer_inputs() -> None:
    trial = MODULE._variant_config("T3", 3e-4, 0, 3_000, 2, {"ema_decay": 0.99})
    assert trial.bundle == "T3"
    assert trial.cube_rotations
    assert trial.ema_decay == 0.99
    assert trial.peak_learning_rate == 3e-4
