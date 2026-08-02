"""Integration tests for CLI flows that exercise real Ray-backed training and tune entrypoints."""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path
from uuid import uuid4

import pytest

ray = pytest.importorskip("ray", reason="ray not installed")
pytestmark = pytest.mark.ray

REPO_ROOT = Path(__file__).resolve().parents[3]
ANYSEARCH = str(Path(REPO_ROOT, ".venv", "Scripts", "anysearch.exe"))

RAY_CLI_YAML = textwrap.dedent("""    experiment:
      name: cli-ray-test
      output_dir: {output_dir}

    env:
      stl_path: /tmp/toy.stl
      scale: 1.0
      agent_count: 1
      max_steps: 10
      seed: 0

    training:
      algorithm: ppo
      runner: local
      iterations: 2
      checkpoint_interval: 1
      video_every: 999

    anyscale:
      cluster_env: x
      compute_config: y
      project: z

    algorithm_config:
      lr: 1.0e-3
      train_batch_size: 200
      clip_param: 0.2
      num_sgd_iter: 1
      lambda_: 0.95
      kl_coeff: 0.2

    model_config:
      hidden_sizes: [32]
      activation: relu
""")


def _run_cli(
    args: list[str],
    tmp_path: Path,
    timeout_s: int = 180,
    extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [ANYSEARCH] + args,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _short_run_dir(name: str) -> Path:
    root = Path(REPO_ROOT, "pytest_tmp_root", "rt")
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root.joinpath(f"{name}_{uuid4().hex[:8]}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _last_json(text: str) -> dict:
    """Parse the last top-level JSON object from stdout.
    Scans from the end looking for a complete {...} that parses successfully."""
    pos = len(text)
    while True:
        start = text.rfind("{", 0, pos)
        if start < 0:
            return {}
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            pos = start  # try the next earlier {


@pytest.fixture()
def cli_yaml(tmp_path: Path) -> Path:
    """Provide cli yaml."""
    p = tmp_path.joinpath("cli_experiment.yaml")
    p.write_text(RAY_CLI_YAML.format(output_dir=str(tmp_path.joinpath("runs"))))
    return p


class TestCliExperimentRun:
    """Tests CliExperimentRun."""
    def test_exits_zero(self, cli_yaml, tmp_path):
        result = _run_cli(["experiment", "run", "--config", str(cli_yaml)], tmp_path)
        assert result.returncode == 0, result.stderr

    def test_output_status_completed(self, cli_yaml, tmp_path):
        result = _run_cli(["experiment", "run", "--config", str(cli_yaml)], tmp_path)
        payload = _last_json(result.stdout)
        assert payload.get("status") == "COMPLETED"

    def test_run_dir_created(self, cli_yaml, tmp_path):
        result = _run_cli(["experiment", "run", "--config", str(cli_yaml)], tmp_path)
        payload = _last_json(result.stdout)
        run_id = payload.get("run_id", "")
        runs_dir = tmp_path.joinpath("runs", "cli-ray-test")
        assert runs_dir.is_dir(), f"runs dir not found: {runs_dir}"

    def test_bad_config_exits_nonzero(self, tmp_path):
        result = _run_cli(["experiment", "run", "--config", "/nonexistent_xyz.yaml"], tmp_path)
        assert result.returncode != 0


class TestCliExperimentResume:
    """Tests CliExperimentResume."""
    def test_resume_exits_zero(self, cli_yaml, tmp_path):
        run_result = _run_cli(["experiment", "run", "--config", str(cli_yaml)], tmp_path)
        assert run_result.returncode == 0, run_result.stderr
        run_id = _last_json(run_result.stdout).get("run_id", "")
        result = _run_cli(
            ["experiment", "resume", "--run-id", run_id, "--output-dir", str(tmp_path.joinpath("runs"))],
            tmp_path,
        )
        assert result.returncode == 0, result.stderr

    def test_resume_status_completed(self, cli_yaml, tmp_path):
        run_result = _run_cli(["experiment", "run", "--config", str(cli_yaml)], tmp_path)
        run_id = _last_json(run_result.stdout).get("run_id", "")
        result = _run_cli(
            ["experiment", "resume", "--run-id", run_id, "--output-dir", str(tmp_path.joinpath("runs"))],
            tmp_path,
        )
        payload = _last_json(result.stdout)
        assert payload.get("status") == "COMPLETED"


class TestCliExperimentRepeat:
    """Tests CliExperimentRepeat."""
    def test_repeat_exits_zero(self, cli_yaml, tmp_path):
        run_result = _run_cli(["experiment", "run", "--config", str(cli_yaml)], tmp_path)
        run_id = _last_json(run_result.stdout).get("run_id", "")
        result = _run_cli(
            ["experiment", "repeat", "--run-id", run_id, "--output-dir", str(tmp_path.joinpath("runs"))],
            tmp_path,
        )
        assert result.returncode == 0, result.stderr

    def test_repeat_returns_new_run_id(self, cli_yaml, tmp_path):
        run_result = _run_cli(["experiment", "run", "--config", str(cli_yaml)], tmp_path)
        run_id = _last_json(run_result.stdout).get("run_id", "")
        result = _run_cli(
            ["experiment", "repeat", "--run-id", run_id, "--output-dir", str(tmp_path.joinpath("runs"))],
            tmp_path,
        )
        new_id = _last_json(result.stdout).get("run_id", "")
        assert new_id and new_id != run_id


class TestCliExperimentInspect:
    """Tests CliExperimentInspect."""
    def test_inspect_exits_zero(self, cli_yaml, tmp_path):
        run_result = _run_cli(["experiment", "run", "--config", str(cli_yaml)], tmp_path)
        run_id = _last_json(run_result.stdout).get("run_id", "")
        result = _run_cli(
            ["experiment", "inspect", "--run-id", run_id, "--output-dir", str(tmp_path.joinpath("runs"))],
            tmp_path,
        )
        assert result.returncode == 0, result.stderr

    def test_inspect_contains_checkpoint_iterations(self, cli_yaml, tmp_path):
        run_result = _run_cli(["experiment", "run", "--config", str(cli_yaml)], tmp_path)
        run_id = _last_json(run_result.stdout).get("run_id", "")
        result = _run_cli(
            ["experiment", "inspect", "--run-id", run_id, "--output-dir", str(tmp_path.joinpath("runs"))],
            tmp_path,
        )
        payload = _last_json(result.stdout)
        assert "checkpoint_iterations" in payload


class TestCliExperimentList:
    """Tests CliExperimentList."""
    def test_list_finds_completed_runs(self, cli_yaml, tmp_path):
        _run_cli(["experiment", "run", "--config", str(cli_yaml)], tmp_path)
        _run_cli(["experiment", "run", "--config", str(cli_yaml)], tmp_path)
        result = _run_cli(
            ["experiment", "list", "--output-dir", str(tmp_path.joinpath("runs"))],
            tmp_path,
        )
        assert result.returncode == 0, result.stderr
        runs = json.loads(result.stdout)
        assert len(runs) >= 2


class TestCliTune:
    """Tests CliTune."""
    @pytest.mark.timeout(300)
    def test_tune_exits_zero(self, tmp_path):
        tune_root = _short_run_dir("tune")
        stl_path = tune_root.joinpath("toy.stl")
        stl_path.write_text("solid toy\nendsolid toy\n")

        result = _run_cli(
            [
                "tune",
                "--stl",
                str(stl_path),
                "--num-samples",
                "1",
                "--max-iterations",
                "2",
                "--agents",
                "1",
                "--max-steps",
                "10",
                "--output-dir",
                str(tune_root.joinpath("out")),
            ],
            tmp_path,
            timeout_s=300,
        )

        assert result.returncode == 0, result.stderr

    @pytest.mark.timeout(300)
    def test_tune_outputs_best_config(self, tmp_path):
        tune_root = _short_run_dir("tune")
        stl_path = tune_root.joinpath("toy.stl")
        stl_path.write_text("solid toy\nendsolid toy\n")

        result = _run_cli(
            [
                "tune",
                "--stl",
                str(stl_path),
                "--num-samples",
                "1",
                "--max-iterations",
                "2",
                "--agents",
                "1",
                "--max-steps",
                "10",
                "--output-dir",
                str(tune_root.joinpath("out")),
            ],
            tmp_path,
            timeout_s=300,
        )

        payload = _last_json(result.stdout)
        assert "best_config" in payload
        assert "best_result" in payload
