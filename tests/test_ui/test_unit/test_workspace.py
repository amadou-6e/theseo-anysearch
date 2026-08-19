"""Tests for native UI workspace discovery."""

from __future__ import annotations

import json
from pathlib import Path

from theseo_anysearch.ui.workspace import scan_workspace


def test_scan_keeps_ordinary_yaml_and_indexes_run_manifest(tmp_path: Path) -> None:
    tmp_path.joinpath("notes.yaml").write_text("title: ordinary\n", encoding="utf-8")
    run_dir = tmp_path.joinpath("runtime", "run-a")
    run_dir.mkdir(parents=True)
    run_dir.joinpath("run.json").write_text(
        json.dumps({"run_id": "run-a", "status": "completed"}), encoding="utf-8"
    )

    index = scan_workspace(tmp_path)

    by_path = {item.path: item for item in index.files}
    assert by_path["notes.yaml"].kind == "yaml"
    assert index.runs[0].run_id == "run-a"
    assert index.runs[0].status == "completed"


def test_invalid_anysearch_yaml_remains_visible_with_diagnostics(tmp_path: Path) -> None:
    tmp_path.joinpath("broken.yaml").write_text("experiment: {}\nenv: {}\n", encoding="utf-8")

    index = scan_workspace(tmp_path)

    candidate = next(item for item in index.files if item.path == "broken.yaml")
    assert candidate.kind == "invalid_anysearch"
    assert candidate.diagnostics


def test_run_artifacts_are_loaded_on_demand_not_held_in_workspace_index(tmp_path: Path) -> None:
    run_dir = tmp_path.joinpath("runtime", "run-a")
    trajectory_dir = run_dir.joinpath("trajectories")
    trajectory_dir.mkdir(parents=True)
    run_dir.joinpath("run.json").write_text(
        json.dumps({"run_id": "run-a", "status": "completed"}), encoding="utf-8"
    )
    trajectory_dir.joinpath("best.json").write_text("{}", encoding="utf-8")

    index = scan_workspace(tmp_path)

    paths = {item.path for item in index.files}
    assert "runtime/run-a/run.json" in paths
    assert "runtime/run-a/trajectories/best.json" not in paths
    assert index.runs[0].path == "runtime/run-a"
