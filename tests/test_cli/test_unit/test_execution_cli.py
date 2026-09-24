"""Clone/apply command usability and fail-closed option validation."""

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from theseo_anysearch.cli.main import app


def _checkpoint(tmp_path: Path) -> Path:
    checkpoint = tmp_path / "run" / "checkpoints" / "iter_000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.json").write_text(json.dumps({
        "iteration": 1, "rllib_path": str(checkpoint),
    }))
    (checkpoint.parent.parent / "experiment.yaml").write_text(yaml.safe_dump({
        "experiment": {"name": "cli-test"},
        "env": {"agent_count": 1, "observation": {"mode": "box", "box_radius": 1},
                "geometry": {"grid_size": 32}},
        "training": {"algorithm": "ppo"},
    }))
    return checkpoint


def test_cli_clones_moves_and_dry_runs_portable_recipe(tmp_path: Path):
    runner = CliRunner()
    bundle = tmp_path / "bundle"
    cloned = runner.invoke(app, ["clone", "--checkpoint", str(_checkpoint(tmp_path)),
                                 "--scope", "evaluation", "--bundle-dir", str(bundle),
                                 "--output", "recipe.yaml"])
    assert cloned.exit_code == 0, cloned.output
    bundle.rename(tmp_path / "moved")
    applied = runner.invoke(app, ["apply", str(tmp_path / "moved" / "recipe.yaml"), "--dry-run"])
    assert applied.exit_code == 0, applied.output
    assert '"execution_supported": true' in applied.output.lower()


def test_cli_rejects_irrelevant_or_incomplete_override_options(tmp_path: Path):
    runner = CliRunner()
    recipe = tmp_path / "recipe.yaml"
    cloned = runner.invoke(app, ["clone", "--checkpoint", str(_checkpoint(tmp_path)),
                                 "--scope", "evaluation", "--output", str(recipe)])
    assert cloned.exit_code == 0, cloned.output
    replacement = tmp_path / "task.yaml"
    replacement.write_text("goal: {}")
    cases = [
        (["--task-config", str(replacement)], "task-config"),
        (["--task", "replace"], "task-config"),
        (["--routes-config", str(replacement)], "routes-config"),
        (["--iterations", "1"], "iterations"),
    ]
    for flags, message in cases:
        result = runner.invoke(app, ["apply", str(recipe), "--dry-run", *flags])
        assert result.exit_code != 0
        assert message in result.output
