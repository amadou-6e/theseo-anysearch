"""Unit tests for experiment CLI help surfaces."""

from __future__ import annotations

from typer.testing import CliRunner

from theseo_anysearch.cli.main import app

class DeterministicCliRunner(CliRunner):
    """Render CLI help at a stable width on local and CI terminals."""

    def invoke(self, *args, **kwargs):
        kwargs.setdefault("terminal_width", 240)
        return super().invoke(*args, **kwargs)

runner = DeterministicCliRunner()


class TestExperimentCLI:
    """Verify experiment CLI help output remains discoverable."""

    def test_experiment_help(self):
        result = runner.invoke(app, ["experiment", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output

    def test_experiment_run_help(self):
        result = runner.invoke(app, ["experiment", "run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_experiment_resume_help(self):
        result = runner.invoke(app, ["experiment", "resume", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output

    def test_experiment_inspect_help(self):
        result = runner.invoke(app, ["experiment", "inspect", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output

    def test_experiment_list_help(self):
        result = runner.invoke(app, ["experiment", "list", "--help"])
        assert result.exit_code == 0

    def test_experiment_repeat_help(self):
        result = runner.invoke(app, ["experiment", "repeat", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output

    def test_root_help_includes_experiment(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "experiment" in result.output
