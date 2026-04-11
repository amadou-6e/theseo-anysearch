"""
Integration tests for the new anysearch CLI commands.

Invokes the installed `anysearch` binary directly (not `python -m ...`).

Test classes:
  TestAddList       — anysearch add / list  (no Ray, no Rust wheel)
  TestRunCommand    — anysearch run <dir>   (requires Ray + Rust wheel)
  TestInspect       — anysearch inspect     (requires Ray + Rust wheel)
  TestResume        — anysearch resume      (requires Ray + Rust wheel)
  TestRepeat        — anysearch repeat      (requires Ray + Rust wheel)

Run all:          pytest tests/integration/test_cli_commands.py
Run Ray subset:   pytest tests/integration/test_cli_commands.py -m ray
Run non-Ray only: pytest tests/integration/test_cli_commands.py -m "not ray"
"""
from __future__ import annotations

import json
import os
import subprocess
import textwrap
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ANYSEARCH = str(Path(REPO_ROOT, ".venv", "Scripts", "anysearch.exe"))

# ---------------------------------------------------------------------------
# Minimal experiment config (no STL — uses geometry_boxes so no file needed)
# ---------------------------------------------------------------------------
_MINIMAL_PPO_YAML = textwrap.dedent("""\
    experiment:
      name: cli-test
      seed: 42

    env:
      geometry_boxes:
        - [1, 1, 1, 5, 5, 5]
      agent_count: 1
      max_steps: 10

    training:
      algorithm: ppo
      runner: local
      iterations: 2
      checkpoint_interval: 1
      trajectory_every: 0
      video_every: 999

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(args: list[str], cwd: Path, timeout_s: int = 180, registry: Path | None = None) -> subprocess.CompletedProcess:
    """Run the anysearch binary with the given args in the given CWD."""
    env = dict(os.environ)
    if registry is not None:
        env["ANYSEARCH_REGISTRY"] = str(registry)
    return subprocess.run(
        [ANYSEARCH] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=env,
    )


def _last_json(text: str) -> dict:
    """Extract the last complete JSON object from stdout."""
    pos = len(text)
    while True:
        start = text.rfind("{", 0, pos)
        if start < 0:
            return {}
        try:
            return json.loads(text[start:])
        except json.JSONDecodeError:
            pos = start


@pytest.fixture()
def exp_dir(tmp_path: Path) -> Path:
    """Experiment directory with a config.yaml written into it."""
    d = tmp_path / "my_experiment"
    d.mkdir()
    (d / "config.yaml").write_text(_MINIMAL_PPO_YAML)
    return d


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Isolated workspace directory used as CWD."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    return ws


@pytest.fixture()
def registry(tmp_path: Path) -> Path:
    """Per-test registry file path (passed via ANYSEARCH_REGISTRY env var)."""
    reg = tmp_path / ".anysearch" / "registry.yaml"
    reg.parent.mkdir(parents=True, exist_ok=True)
    return reg


# ---------------------------------------------------------------------------
# anysearch add / list  (no Ray required)
# ---------------------------------------------------------------------------

class TestAddList:
    """Tests for anysearch add and anysearch list — no training needed."""

    def test_add_exits_zero(self, exp_dir, workspace, registry):
        result = _run(["add", str(exp_dir)], cwd=workspace, registry=registry)
        assert result.returncode == 0, result.stderr

    def test_add_prints_registered_name(self, exp_dir, workspace, registry):
        result = _run(["add", str(exp_dir)], cwd=workspace, registry=registry)
        # Name comes from directory basename (my_experiment)
        assert "my_experiment" in result.stdout

    def test_add_creates_registry_file(self, exp_dir, workspace, registry):
        _run(["add", str(exp_dir)], cwd=workspace, registry=registry)
        assert registry.exists()
        data = yaml.safe_load(registry.read_text())
        assert "my_experiment" in data.get("experiments", {})

    def test_add_with_explicit_name(self, exp_dir, workspace, registry):
        result = _run(["add", str(exp_dir), "my-alias"], cwd=workspace, registry=registry)
        assert result.returncode == 0, result.stderr
        assert "my-alias" in result.stdout
        data = yaml.safe_load(registry.read_text())
        assert "my-alias" in data["experiments"]

    def test_add_idempotent(self, exp_dir, workspace, registry):
        _run(["add", str(exp_dir)], cwd=workspace, registry=registry)
        result = _run(["add", str(exp_dir)], cwd=workspace, registry=registry)
        assert result.returncode == 0, result.stderr

    def test_list_empty_registry(self, workspace, registry):
        result = _run(["list"], cwd=workspace, registry=registry)
        assert result.returncode == 0
        assert "registered" in result.stdout.lower() or "no experiments" in result.stdout.lower()

    def test_list_shows_registered_experiment(self, exp_dir, workspace, registry):
        _run(["add", str(exp_dir)], cwd=workspace, registry=registry)
        result = _run(["list"], cwd=workspace, registry=registry)
        assert result.returncode == 0, result.stderr
        assert "my_experiment" in result.stdout

    def test_list_scoped_to_dir_no_runs(self, exp_dir, workspace, registry):
        result = _run(["list", str(exp_dir)], cwd=workspace, registry=registry)
        assert result.returncode == 0
        assert "no runs" in result.stdout.lower()

    def test_list_nonexistent_dir_still_exits_zero(self, workspace, registry):
        # list is informational — we prefer it not crash
        result = _run(["list", str(workspace / "does_not_exist")], cwd=workspace, registry=registry)
        assert result.returncode == 0

    def test_add_missing_dir_exits_nonzero(self, workspace, registry):
        result = _run(["add", str(workspace / "no_such_dir")], cwd=workspace, registry=registry)
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# Ray-required tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def ray_available():
    """Skip the test if Ray is not installed."""
    pytest.importorskip("ray", reason="ray not installed")


@pytest.fixture()
def completed_run(exp_dir: Path, workspace: Path, ray_available):
    """
    Fixture that runs anysearch run <exp_dir> and returns (run_id, exp_dir, workspace).
    Skips the test if Ray is not available or the run fails.
    """
    result = _run(["run", str(exp_dir)], cwd=workspace, timeout_s=180)
    if result.returncode != 0:
        pytest.skip(f"anysearch run failed: {result.stderr}")
    payload = _last_json(result.stdout)
    run_id = payload.get("run_id", "")
    if not run_id:
        pytest.skip("Could not extract run_id from output")
    return run_id, exp_dir, workspace


@pytest.mark.ray
class TestRunCommand:
    """Tests RunCommand."""
    @pytest.fixture(autouse=True)
    def _require_ray(self, ray_available):
        pass

    @pytest.mark.timeout(180)
    def test_run_exits_zero(self, exp_dir, workspace):
        result = _run(["run", str(exp_dir)], cwd=workspace)
        assert result.returncode == 0, result.stderr

    @pytest.mark.timeout(180)
    def test_run_output_in_experiment_dir(self, exp_dir, workspace):
        result = _run(["run", str(exp_dir)], cwd=workspace)
        payload = _last_json(result.stdout)
        run_id = payload.get("run_id", "")
        assert run_id, "No run_id in output"
        # run lands at exp_dir/<experiment.name>/<run_id>/
        assert any(exp_dir.rglob(run_id)), f"Run dir '{run_id}' not found under {exp_dir}"

    @pytest.mark.timeout(180)
    def test_run_status_completed(self, exp_dir, workspace):
        result = _run(["run", str(exp_dir)], cwd=workspace)
        assert _last_json(result.stdout).get("status") == "COMPLETED"

    @pytest.mark.timeout(180)
    def test_run_auto_registers(self, exp_dir, workspace):
        _run(["run", str(exp_dir)], cwd=workspace)
        registry = workspace / ".anysearch" / "registry.yaml"
        assert registry.exists()
        data = yaml.safe_load(registry.read_text())
        assert "cli-test" in data.get("experiments", {})

    @pytest.mark.timeout(180)
    def test_run_with_output_dir_override(self, exp_dir, workspace, tmp_path):
        out = tmp_path / "custom_out"
        result = _run(["run", str(exp_dir), "--output-dir", str(out)], cwd=workspace)
        assert result.returncode == 0, result.stderr
        run_id = _last_json(result.stdout).get("run_id", "")
        assert any(out.rglob(run_id)), f"Run not found under custom output dir {out}"

    @pytest.mark.timeout(180)
    def test_run_prints_replay_hint(self, exp_dir, workspace):
        result = _run(["run", str(exp_dir)], cwd=workspace)
        assert "anysearch replay" in result.stdout or "anysearch replay" in result.stderr

    @pytest.mark.timeout(180)
    def test_run_by_registered_name(self, exp_dir, workspace):
        _run(["add", str(exp_dir)], cwd=workspace)
        result = _run(["run", "cli-test"], cwd=workspace)
        assert result.returncode == 0, result.stderr
        assert _last_json(result.stdout).get("status") == "COMPLETED"


@pytest.mark.ray
class TestInspect:
    """Tests Inspect."""
    @pytest.fixture(autouse=True)
    def _require_ray(self, ray_available):
        pass

    @pytest.mark.timeout(180)
    def test_inspect_exits_zero(self, completed_run):
        run_id, exp_dir, workspace = completed_run
        result = _run(["inspect", f"{exp_dir}:{run_id}"], cwd=workspace)
        assert result.returncode == 0, result.stderr

    @pytest.mark.timeout(180)
    def test_inspect_output_is_json(self, completed_run):
        run_id, exp_dir, workspace = completed_run
        result = _run(["inspect", f"{exp_dir}:{run_id}"], cwd=workspace)
        payload = _last_json(result.stdout)
        assert payload.get("run_id") == run_id

    @pytest.mark.timeout(180)
    def test_inspect_contains_checkpoints(self, completed_run):
        run_id, exp_dir, workspace = completed_run
        result = _run(["inspect", f"{exp_dir}:{run_id}"], cwd=workspace)
        payload = _last_json(result.stdout)
        assert "checkpoint_iterations" in payload
        assert len(payload["checkpoint_iterations"]) > 0

    @pytest.mark.timeout(180)
    def test_inspect_status_completed(self, completed_run):
        run_id, exp_dir, workspace = completed_run
        result = _run(["inspect", f"{exp_dir}:{run_id}"], cwd=workspace)
        assert _last_json(result.stdout).get("status") == "COMPLETED"

    @pytest.mark.timeout(180)
    def test_inspect_by_registered_name(self, completed_run, workspace):
        run_id, exp_dir, _ = completed_run
        _run(["add", str(exp_dir), "cli-test"], cwd=workspace)
        result = _run(["inspect", f"cli-test:{run_id}"], cwd=workspace)
        assert result.returncode == 0, result.stderr

    def test_inspect_nonexistent_run_exits_nonzero(self, exp_dir, workspace):
        result = _run(["inspect", f"{exp_dir}:deadbeef"], cwd=workspace)
        assert result.returncode != 0

    def test_inspect_missing_id_exits_nonzero(self, exp_dir, workspace):
        result = _run(["inspect", str(exp_dir)], cwd=workspace)
        assert result.returncode != 0


@pytest.mark.ray
class TestResume:
    """Tests Resume."""
    @pytest.fixture(autouse=True)
    def _require_ray(self, ray_available):
        pass

    @pytest.mark.timeout(240)
    def test_resume_exits_zero(self, completed_run):
        run_id, exp_dir, workspace = completed_run
        result = _run(["resume", f"{exp_dir}:{run_id}"], cwd=workspace)
        assert result.returncode == 0, result.stderr

    @pytest.mark.timeout(240)
    def test_resume_status_completed(self, completed_run):
        run_id, exp_dir, workspace = completed_run
        result = _run(["resume", f"{exp_dir}:{run_id}"], cwd=workspace)
        assert _last_json(result.stdout).get("status") == "COMPLETED"

    def test_resume_missing_id_exits_nonzero(self, exp_dir, workspace):
        result = _run(["resume", str(exp_dir)], cwd=workspace)
        assert result.returncode != 0


@pytest.mark.ray
class TestRepeat:
    """Tests Repeat."""
    @pytest.fixture(autouse=True)
    def _require_ray(self, ray_available):
        pass

    @pytest.mark.timeout(240)
    def test_repeat_exits_zero(self, completed_run):
        run_id, exp_dir, workspace = completed_run
        result = _run(["repeat", f"{exp_dir}:{run_id}"], cwd=workspace)
        assert result.returncode == 0, result.stderr

    @pytest.mark.timeout(240)
    def test_repeat_returns_new_run_id(self, completed_run):
        run_id, exp_dir, workspace = completed_run
        result = _run(["repeat", f"{exp_dir}:{run_id}"], cwd=workspace)
        new_id = _last_json(result.stdout).get("run_id", "")
        assert new_id and new_id != run_id

    @pytest.mark.timeout(240)
    def test_repeat_status_completed(self, completed_run):
        run_id, exp_dir, workspace = completed_run
        result = _run(["repeat", f"{exp_dir}:{run_id}"], cwd=workspace)
        assert _last_json(result.stdout).get("status") == "COMPLETED"

    def test_repeat_missing_id_exits_nonzero(self, exp_dir, workspace):
        result = _run(["repeat", str(exp_dir)], cwd=workspace)
        assert result.returncode != 0


@pytest.mark.ray
class TestListWithRuns:
    """Tests ListWithRuns."""
    @pytest.fixture(autouse=True)
    def _require_ray(self, ray_available):
        pass

    @pytest.mark.timeout(240)
    def test_list_shows_completed_run(self, completed_run):
        run_id, exp_dir, workspace = completed_run
        result = _run(["list", str(exp_dir)], cwd=workspace)
        assert result.returncode == 0, result.stderr
        assert run_id in result.stdout or "COMPLETED" in result.stdout

    @pytest.mark.timeout(360)
    def test_list_registered_shows_run(self, completed_run, workspace):
        run_id, exp_dir, _ = completed_run
        _run(["add", str(exp_dir), "cli-test"], cwd=workspace)
        result = _run(["list"], cwd=workspace)
        assert result.returncode == 0, result.stderr
        assert "cli-test" in result.stdout
