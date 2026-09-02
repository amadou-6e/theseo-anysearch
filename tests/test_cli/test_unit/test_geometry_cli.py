import json

from typer.testing import CliRunner

from theseo_anysearch.cli.commands import geometry


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
