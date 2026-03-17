from typer.testing import CliRunner

from theseo_anysearch.cli.main import app

runner = CliRunner()


class TestCliHelp:
    def test_root_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "train" in result.output
        assert "tune" in result.output

    def test_train_help(self):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "--stl" in result.output
        assert "--iterations" in result.output
        assert "--agents" in result.output

    def test_tune_help(self):
        result = runner.invoke(app, ["tune", "--help"])
        assert result.exit_code == 0
        assert "--stl" in result.output
        assert "--scheduler" in result.output

    def test_train_missing_stl_exits_nonzero(self):
        result = runner.invoke(app, ["train"])
        assert result.exit_code != 0

    def test_root_help_mentions_experiment(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "experiment" in result.output


class TestCliExperimentHelp:
    def test_experiment_help(self):
        result = runner.invoke(app, ["experiment", "--help"])
        assert result.exit_code == 0

    def test_experiment_run_help_mentions_config(self):
        result = runner.invoke(app, ["experiment", "run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_experiment_resume_help_mentions_run_id(self):
        result = runner.invoke(app, ["experiment", "resume", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output

    def test_experiment_repeat_help_mentions_run_id(self):
        result = runner.invoke(app, ["experiment", "repeat", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output

    def test_experiment_inspect_help_mentions_run_id(self):
        result = runner.invoke(app, ["experiment", "inspect", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output

    def test_experiment_list_help(self):
        result = runner.invoke(app, ["experiment", "list", "--help"])
        assert result.exit_code == 0
