"""
Unit tests for the anysearch CLI.

Covers:
- New top-level commands: run, add, list, inspect, resume, repeat
- Deprecated groups still present: experiment, train, tune
- Deprecation notices emitted on stderr
- Registry helpers: add_experiment, resolve_ref, find_config_in_dir
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from theseo_anysearch.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Root help
# ---------------------------------------------------------------------------

class TestRootHelp:
    """Tests RootHelp."""
    def test_root_help_exits_zero(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_root_help_shows_new_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "list" in result.output
        assert "add" in result.output
        assert "inspect" in result.output
        assert "resume" in result.output
        assert "repeat" in result.output

    def test_root_help_shows_replay_mlflow_ray(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "replay" in result.output
        assert "mlflow" in result.output
        assert "ray" in result.output


# ---------------------------------------------------------------------------
# anysearch run
# ---------------------------------------------------------------------------

class TestRunCommand:
    """Tests RunCommand."""
    def test_run_help(self):
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--tag" in result.output

    def test_run_missing_arg_exits_nonzero(self):
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0

    def test_run_nonexistent_dir_exits_nonzero(self):
        result = runner.invoke(app, ["run", "/nonexistent/dir/that/does/not/exist"])
        assert result.exit_code != 0

    def test_run_dir_with_multiple_yamls_exits_nonzero(self, tmp_path: Path):
        tmp_path.joinpath("a.yaml").write_text("experiment:\n  name: a\n")
        tmp_path.joinpath("b.yaml").write_text("experiment:\n  name: b\n")
        result = runner.invoke(app, ["run", str(tmp_path)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# anysearch add
# ---------------------------------------------------------------------------

class TestAddCommand:
    """Tests AddCommand."""
    def test_add_help(self):
        result = runner.invoke(app, ["add", "--help"])
        assert result.exit_code == 0

    def test_add_nonexistent_dir_exits_nonzero(self):
        result = runner.invoke(app, ["add", "/nonexistent/path"])
        assert result.exit_code != 0

    def test_add_registers_experiment(self, tmp_path, isolated_registry):
        exp_dir = tmp_path.joinpath("my_experiment")
        exp_dir.mkdir()
        exp_dir.joinpath("config.yaml").write_text(
            "experiment:\n  name: my-exp\n  seed: 42\n"
        )
        result = runner.invoke(app, ["add", str(exp_dir)])
        assert result.exit_code == 0
        # Name comes from directory basename, not experiment.name
        assert "my_experiment" in result.output

        assert isolated_registry.exists()
        data = yaml.safe_load(isolated_registry.read_text())
        assert "my_experiment" in data["experiments"]

    def test_add_with_explicit_name(self, tmp_path, isolated_registry):
        exp_dir = tmp_path.joinpath("some_dir")
        exp_dir.mkdir()
        exp_dir.joinpath("config.yaml").write_text(
            "experiment:\n  name: ignored\n  seed: 1\n"
        )
        result = runner.invoke(app, ["add", str(exp_dir), "custom-name"])
        assert result.exit_code == 0
        assert "custom-name" in result.output

        data = yaml.safe_load(isolated_registry.read_text())
        assert "custom-name" in data["experiments"]


# ---------------------------------------------------------------------------
# anysearch list
# ---------------------------------------------------------------------------

class TestListCommand:
    """Tests ListCommand."""
    def test_list_help(self):
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0

    def test_list_no_registry_shows_message(self, isolated_registry):
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "registered" in result.output.lower() or "No experiments" in result.output

    def test_list_with_explicit_dir(self, tmp_path):
        # Should not error even if directory has no runs
        result = runner.invoke(app, ["list", str(tmp_path)])
        assert result.exit_code == 0
        assert "no runs" in result.output.lower()


# ---------------------------------------------------------------------------
# anysearch inspect
# ---------------------------------------------------------------------------

class TestInspectCommand:
    """Tests InspectCommand."""
    def test_inspect_help(self):
        result = runner.invoke(app, ["inspect", "--help"])
        assert result.exit_code == 0

    def test_inspect_requires_run_id_in_ref(self):
        # Bare name with no colon → should exit nonzero with helpful message
        result = runner.invoke(app, ["inspect", "ppo-baseline"])
        assert result.exit_code != 0
        assert "run_id" in result.output or "run_id" in (result.stderr or "")

    def test_inspect_nonexistent_run_exits_nonzero(self, tmp_path):
        result = runner.invoke(
            app, ["inspect", f"{tmp_path}:deadbeef", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# anysearch resume / repeat
# ---------------------------------------------------------------------------

class TestResumeRepeatCommands:
    """Tests ResumeRepeatCommands."""
    def test_resume_help(self):
        result = runner.invoke(app, ["resume", "--help"])
        assert result.exit_code == 0

    def test_repeat_help(self):
        result = runner.invoke(app, ["repeat", "--help"])
        assert result.exit_code == 0

    def test_resume_requires_colon_ref(self):
        result = runner.invoke(app, ["resume", "barenamewithoutcolon"])
        assert result.exit_code != 0

    def test_repeat_requires_colon_ref(self):
        result = runner.invoke(app, ["repeat", "barenamewithoutcolon"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Deprecated commands still present + emit deprecation notice
# ---------------------------------------------------------------------------

class TestDeprecatedCommands:
    """Tests DeprecatedCommands."""
    def test_experiment_group_still_present(self):
        result = runner.invoke(app, ["experiment", "--help"])
        assert result.exit_code == 0

    def test_experiment_run_still_present(self):
        result = runner.invoke(app, ["experiment", "run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_experiment_run_emits_deprecation(self):
        # Invoke with a bad config so it fails fast, but we still see the notice
        result = runner.invoke(
            app, ["experiment", "run", "--config", "/nonexistent.yaml"]
        )
        assert "deprecated" in (result.output).lower()

    def test_experiment_resume_still_present(self):
        result = runner.invoke(app, ["experiment", "resume", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output

    def test_experiment_repeat_still_present(self):
        result = runner.invoke(app, ["experiment", "repeat", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output

    def test_experiment_inspect_still_present(self):
        result = runner.invoke(app, ["experiment", "inspect", "--help"])
        assert result.exit_code == 0
        assert "--run-id" in result.output

    def test_train_help_shows_deprecation(self):
        result = runner.invoke(app, ["train", "--help"])
        assert result.exit_code == 0
        assert "deprecated" in result.output.lower()

    def test_tune_help_shows_deprecation(self):
        result = runner.invoke(app, ["tune", "--help"])
        assert result.exit_code == 0
        assert "deprecated" in result.output.lower()


# ---------------------------------------------------------------------------
# Registry unit tests
# ---------------------------------------------------------------------------

class TestRegistry:
    """Tests Registry."""
    def test_load_empty_registry(self, isolated_registry):
        from theseo_anysearch.cli.registry import load_registry
        assert load_registry() == {}

    def test_add_and_load(self, tmp_path, isolated_registry):
        exp_dir = tmp_path / "ppo_baseline"
        exp_dir.mkdir()
        (exp_dir / "config.yaml").write_text(
            "experiment:\n  name: ppo-baseline\n  seed: 42\n"
        )
        from theseo_anysearch.cli.registry import add_experiment, load_registry
        # Name comes from directory basename
        name = add_experiment(exp_dir)
        assert name == "ppo_baseline"
        reg = load_registry()
        assert "ppo_baseline" in reg

    def test_add_yaml_uses_stem_as_name(self, tmp_path, isolated_registry):
        yaml_file = tmp_path / "multi_agent_ppo_asha.yaml"
        yaml_file.write_text("experiment:\n  name: ignored\n  seed: 1\n")
        from theseo_anysearch.cli.registry import add_experiment, load_registry
        name = add_experiment(yaml_file)
        assert name == "multi_agent_ppo_asha"
        assert "multi_agent_ppo_asha" in load_registry()

    def test_resolve_ref_name_colon_id(self, tmp_path, isolated_registry):
        exp_dir = tmp_path / "ppo_baseline"
        exp_dir.mkdir()
        (exp_dir / "config.yaml").write_text(
            "experiment:\n  name: ppo-baseline\n  seed: 1\n"
        )
        from theseo_anysearch.cli.registry import add_experiment, resolve_ref
        add_experiment(exp_dir)
        resolved_dir, identifier = resolve_ref("ppo_baseline:a1b2c3d4")
        assert identifier == "a1b2c3d4"
        assert resolved_dir == exp_dir.resolve()

    def test_resolve_ref_dir_colon_id(self, tmp_path, monkeypatch, isolated_registry):
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "my_exp"
        exp_dir.mkdir()
        from theseo_anysearch.cli.registry import resolve_ref
        # Use a relative path so there is no Windows drive-letter colon
        ref = "my_exp:v12"
        resolved_dir, identifier = resolve_ref(ref)
        assert identifier == "v12"

    def test_resolve_ref_no_colon_returns_none_id(self, tmp_path, monkeypatch, isolated_registry):
        monkeypatch.chdir(tmp_path)
        exp_dir = tmp_path / "my_exp"
        exp_dir.mkdir()
        from theseo_anysearch.cli.registry import resolve_ref
        resolved_dir, identifier = resolve_ref("my_exp")
        assert identifier is None

    def test_find_config_prefers_config_yaml(self, tmp_path):
        (tmp_path / "config.yaml").write_text("experiment:\n  name: x\n")
        (tmp_path / "other.yaml").write_text("experiment:\n  name: y\n")
        from theseo_anysearch.cli.registry import find_config_in_dir
        found = find_config_in_dir(tmp_path)
        assert found.name == "config.yaml"

    def test_repo_local_registry_preferred_over_home(self, tmp_path, monkeypatch):
        repo_root = tmp_path / "repo"
        repo_root.mkdir()
        (repo_root / ".git").mkdir()
        monkeypatch.chdir(repo_root)
        monkeypatch.delenv("ANYSEARCH_REGISTRY", raising=False)
        monkeypatch.delenv("ANYSEARCH_REPO_ROOT", raising=False)

        from theseo_anysearch.cli.registry import _registry_file

        assert _registry_file() == (repo_root / ".anysearch" / "registry.yaml")

    def test_home_registry_permission_error_is_legible(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("ANYSEARCH_REGISTRY", raising=False)
        monkeypatch.delenv("ANYSEARCH_REPO_ROOT", raising=False)

        from theseo_anysearch.cli import registry as registry_mod

        fake_home = tmp_path / "fake_home"
        fake_home.mkdir()
        monkeypatch.setattr(registry_mod.Path, "home", staticmethod(lambda: fake_home))
        monkeypatch.setattr(registry_mod, "_repo_registry_file", lambda: None)

        def _raise_permission(*args, **kwargs):
            raise PermissionError("denied")

        monkeypatch.setattr(registry_mod.Path, "write_text", _raise_permission)

        with pytest.raises(registry_mod.RegistryAccessError, match="home directory"):
            registry_mod.save_registry({"demo": "C:/tmp/demo.yaml"})

    def test_find_config_falls_back_to_experiment_yaml(self, tmp_path):
        (tmp_path / "experiment.yaml").write_text("experiment:\n  name: x\n")
        from theseo_anysearch.cli.registry import find_config_in_dir
        found = find_config_in_dir(tmp_path)
        assert found.name == "experiment.yaml"

    def test_find_config_single_yaml(self, tmp_path):
        (tmp_path / "my_exp.yaml").write_text("experiment:\n  name: x\n")
        from theseo_anysearch.cli.registry import find_config_in_dir
        found = find_config_in_dir(tmp_path)
        assert found.name == "my_exp.yaml"

    def test_find_config_multiple_yamls_raises(self, tmp_path):
        (tmp_path / "a.yaml").write_text("x: 1\n")
        (tmp_path / "b.yaml").write_text("x: 2\n")
        from theseo_anysearch.cli.registry import find_config_in_dir
        with pytest.raises(ValueError, match="Multiple YAML"):
            find_config_in_dir(tmp_path)

    def test_find_config_no_yaml_raises(self, tmp_path):
        from theseo_anysearch.cli.registry import find_config_in_dir
        with pytest.raises(FileNotFoundError):
            find_config_in_dir(tmp_path)
