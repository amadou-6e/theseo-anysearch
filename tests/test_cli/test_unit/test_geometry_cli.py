import json
from pathlib import Path

from typer.testing import CliRunner

from theseo_anysearch.cli.commands import geometry
from theseo_anysearch.worlds.compiler import BoxSource, compile_world
from theseo_anysearch.worlds.manifest import WorldExtent


runner = CliRunner()


def report(suitable: bool = True):
    return {
        "seed": 42,
        "extent": [8, 8, 8],
        "occupancy_count": 12,
        "geometry_identity": "a" * 64,
        "geometry_validity": {"valid": suitable, "rejection_reason": None if suitable else "out_of_bounds"},
        "task_feasibility": {"feasible": suitable},
        "training_suitability": {"suitable": suitable, "reason": None if suitable else "out_of_bounds"},
        "evaluation_suitability": {"suitable": suitable, "stable_identity": "a" * 64},
        "proposal": None,
        "bounded_large_world_read": False,
    }


def test_inspect_supports_machine_readable_json(monkeypatch) -> None:
    monkeypatch.setattr(geometry, "geometry_report", lambda *_args, **_kwargs: report())
    result = runner.invoke(geometry.app, ["inspect", "experiment.yaml", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["training_suitability"]["suitable"] is True


def test_validate_text_failure_returns_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(geometry, "geometry_report", lambda *_args, **_kwargs: report(False))
    result = runner.invoke(geometry.app, ["validate", "experiment.yaml"])
    assert result.exit_code == 1
    assert "geometry: INVALID" in result.stdout
    assert "out_of_bounds" not in result.stdout or "INVALID" in result.stdout


def test_compiled_world_validation_detects_an_occupied_start(
    tmp_path: Path, monkeypatch
) -> None:
    world = compile_world(
        [BoxSource((0, 0, 0), (0, 0, 0))],
        WorldExtent(x=8, y=8, z=8),
        tmp_path,
    )
    runtime = {
        "compiled_world_path": str(world.root),
        "extent": (8, 8, 8),
        "waypoints": {"start": [1, 1, 1], "goal": [4, 4, 4]},
        "action_mode": "discrete_6",
        "max_steps": 64,
        "geometry_validation": {"maximum_search_nodes": 1_000},
    }
    monkeypatch.setattr(
        geometry,
        "_resolve",
        lambda *_args, **_kwargs: (object(), runtime, ()),
    )

    result = geometry.geometry_report(Path("experiment.yaml"))

    assert result["bounded_large_world_read"] is True
    assert result["task_feasibility"]["feasible"] is False
    assert result["task_feasibility"]["rejection_reason"] == "occupied_start"
    assert result["training_suitability"] == {
        "suitable": False,
        "reason": "occupied_start",
    }
