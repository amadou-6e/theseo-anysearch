from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import pytest

ray = pytest.importorskip("ray", reason="ray not installed")
pytestmark = pytest.mark.ray

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG_DIR = Path(REPO_ROOT, "theseo_anysearch")

MINIMAL_EXPERIMENT_YAML = textwrap.dedent("""\
    experiment:
      name: train-cli-test
      output_dir: {output_dir}

    env:
      stl_path: /tmp/toy.stl
      scale: 1.0
      agent_count: 1
      max_steps: 10
      seed: 0

    training:
      algorithm: ppo
      model: voxel_encoder
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


def _run(cmd: list[str], cwd: Path, timeout_s: int = 180) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _parse_last_json_blob(stdout_text: str) -> dict:
    start = stdout_text.rfind("{")
    if start < 0:
        return {}
    return json.loads(stdout_text[start:])


@pytest.mark.integration
class TestTrainIntegration:
    def test_module_path_invocation_runs(self, tmp_path: Path):
        config_path = tmp_path / "experiment.yaml"
        output_dir = tmp_path / "runtime_out"
        config_path.write_text(MINIMAL_EXPERIMENT_YAML.format(output_dir=str(output_dir)))

        cmd = [
            str(Path(REPO_ROOT, ".venv", "Scripts", "python.exe")),
            "-m",
            "theseo_anysearch.cli.main",
            "experiment",
            "run",
            "--config",
            str(config_path),
        ]

        result = _run(cmd, cwd=REPO_ROOT)

        assert result.returncode == 0, result.stderr
        payload = _parse_last_json_blob(result.stdout)
        assert payload["status"] == "COMPLETED"
        assert payload["experiment_name"] == "train-cli-test"

    def test_user_style_invocation_from_package_dir_runs(self, tmp_path: Path):
        config_path = tmp_path / "experiment.yaml"
        output_dir = tmp_path / "runtime_out_from_pkg"
        config_path.write_text(MINIMAL_EXPERIMENT_YAML.format(output_dir=str(output_dir)))

        cmd = [
            str(Path("..", ".venv", "Scripts", "python.exe")),
            "-m",
            "cli.main",
            "experiment",
            "run",
            "--config",
            str(config_path),
        ]

        result = _run(cmd, cwd=PKG_DIR)

        assert result.returncode == 0, result.stderr
        payload = _parse_last_json_blob(result.stdout)
        assert payload["status"] == "COMPLETED"
        assert payload["experiment_name"] == "train-cli-test"
        assert not Path(PKG_DIR, "runtime").exists()
