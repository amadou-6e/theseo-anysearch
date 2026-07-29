"""CLI tests for adaptive resource benchmarking."""

from typer.testing import CliRunner

from theseo_anysearch.cli.main import app


def test_resource_benchmark_help_is_discoverable() -> None:
    result = CliRunner().invoke(app, ["benchmark", "resources", "--help"])

    assert result.exit_code == 0
    assert "--decline-patience" in result.output
    assert "--max-envs-per-worker" in result.output
    assert "--max-workers" in result.output
    assert "--max-gpu-utilization" in result.output
    assert "--max-duration-minutes" in result.output
    assert "--debug" in result.output
    assert "--open" in result.output
