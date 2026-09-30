"""Regression tests for gating the native explain bridge to DQN runs.

`anysearch replay` used to unconditionally launch the native explain bridge
alongside every replay, which raises inside PolicyExplanationService for any
non-DQN algorithm -- breaking plain replay for most of the algorithm surface.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from theseo_anysearch.cli.commands.replay import _supports_native_explain


def _write_experiment(run_dir: Path, algorithm: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "experiment.yaml").write_text(
        yaml.safe_dump({"training": {"algorithm": algorithm}}),
        encoding="utf-8",
    )


def test_dqn_run_supports_native_explain(tmp_path: Path) -> None:
    run_dir = tmp_path / "dqn-run"
    _write_experiment(run_dir, "dqn")

    assert _supports_native_explain(run_dir) is True


def test_non_dqn_run_does_not_support_native_explain(tmp_path: Path) -> None:
    run_dir = tmp_path / "ppo-run"
    _write_experiment(run_dir, "ppo")

    assert _supports_native_explain(run_dir) is False


def test_missing_experiment_yaml_does_not_support_native_explain(tmp_path: Path) -> None:
    run_dir = tmp_path / "no-config-run"
    run_dir.mkdir(parents=True)

    assert _supports_native_explain(run_dir) is False


def test_malformed_experiment_yaml_does_not_support_native_explain(tmp_path: Path) -> None:
    run_dir = tmp_path / "malformed-run"
    run_dir.mkdir(parents=True)
    (run_dir / "experiment.yaml").write_text("{not: valid: yaml", encoding="utf-8")

    assert _supports_native_explain(run_dir) is False
