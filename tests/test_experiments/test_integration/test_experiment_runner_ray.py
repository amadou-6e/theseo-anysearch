"""
Ray integration tests for ExperimentRunner.

Requires ray[rllib] + torch + theseo_core wheel.  Run with:
    pytest tests/integration/test_experiment_runner_ray.py -m ray -v

Module-scoped shared fixtures keep algorithm builds to 3 (run + resume + repeat).
"""
from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import pytest

ray = pytest.importorskip("ray", reason="ray not installed")
pytestmark = pytest.mark.ray

from theseo_anysearch.experiments.runner import ExperimentRunner


RAY_EXPERIMENT_YAML = textwrap.dedent("""    experiment:
      name: ray-test
      output_dir: {output_dir}
      seed: 0

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
      trajectory_every: 1
      best_trajectory: true

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


# ---------------------------------------------------------------------------
# Module-scoped shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def module_tmp(tmp_path_factory):
    """Provide module tmp."""
    return tmp_path_factory.mktemp("ray_runner")


@pytest.fixture(scope="module")
def experiment_config(ray_session, module_tmp):
    """Provide experiment config."""
    from theseo_anysearch.experiments.loader import load_experiment
    yaml_path = module_tmp / "experiment.yaml"
    yaml_path.write_text(RAY_EXPERIMENT_YAML.format(output_dir=str(module_tmp)))
    return load_experiment(yaml_path)


@pytest.fixture(scope="module")
def run_info(ray_session, experiment_config):
    """Perform one full training run. Shared across TestExperimentRunnerRay."""
    return ExperimentRunner(experiment_config).run()


@pytest.fixture(scope="module")
def resumed_info(ray_session, experiment_config, run_info):
    """Resume the shared run once."""
    return ExperimentRunner(experiment_config).resume(run_info.run_id)


@pytest.fixture(scope="module")
def repeated_info(ray_session, experiment_config, run_info):
    """Repeat the shared run once."""
    return ExperimentRunner(experiment_config).repeat(run_info.run_id)


# ---------------------------------------------------------------------------
# 1. Basic run
# ---------------------------------------------------------------------------

class TestExperimentRunnerRay:
    """Tests ExperimentRunnerRay."""
    def test_status_completed(self, run_info):
        assert run_info.status == "COMPLETED"

    def test_run_id_is_8_char_hex(self, run_info):
        assert re.fullmatch(r"[0-9a-f]{8}", run_info.run_id)

    def test_run_dir_contains_experiment_yaml(self, run_info, experiment_config):
        run_dir = experiment_config.run_output_dir / run_info.run_id
        assert (run_dir / "experiment.yaml").exists()

    def test_checkpoints_subdir_populated(self, run_info, experiment_config):
        run_dir = experiment_config.run_output_dir / run_info.run_id
        ckpt_root = run_dir / "checkpoints"
        assert ckpt_root.is_dir()
        assert any(ckpt_root.iterdir())

    def test_run_json_has_correct_experiment_name(self, run_info, experiment_config):
        run_dir = experiment_config.run_output_dir / run_info.run_id
        data = json.loads((run_dir / "run.json").read_text())
        assert data["experiment_name"] == "ray-test"


# ---------------------------------------------------------------------------
# 2. Resume
# ---------------------------------------------------------------------------

class TestExperimentRunnerRayResume:
    """Tests ExperimentRunnerRayResume."""
    def test_resume_status_completed(self, resumed_info):
        assert resumed_info.status == "COMPLETED"

    def test_resume_preserves_run_id(self, run_info, resumed_info):
        assert resumed_info.run_id == run_info.run_id


# ---------------------------------------------------------------------------
# 3. Repeat
# ---------------------------------------------------------------------------

class TestExperimentRunnerRayRepeat:
    """Tests ExperimentRunnerRayRepeat."""
    def test_repeat_returns_new_run_id(self, run_info, repeated_info):
        assert repeated_info.run_id != run_info.run_id

    def test_repeat_status_completed(self, repeated_info):
        assert repeated_info.status == "COMPLETED"

    def test_repeat_creates_separate_run_dir(self, run_info, repeated_info, experiment_config):
        dir1 = experiment_config.run_output_dir / run_info.run_id
        dir2 = experiment_config.run_output_dir / repeated_info.run_id
        assert dir1 != dir2
        assert dir2.is_dir()


# ---------------------------------------------------------------------------
# 4. Trajectory output
# ---------------------------------------------------------------------------

def _has_filled_voxels() -> bool:
    try:
        import theseo_core
        return hasattr(theseo_core.PyVoxelEnv, "filled_voxels")
    except ImportError:
        return False


@pytest.mark.skipif(
    not _has_filled_voxels(),
    reason="theseo_core wheel missing filled_voxels — rebuild with maturin develop",
)
class TestTrajectoryOutput:
    """Asserts that trajectory JSON files are written by ExperimentRunner."""

    def test_periodic_trajectory_written(self, run_info, experiment_config):
        run_dir = experiment_config.run_output_dir / run_info.run_id
        traj_dir = run_dir / "trajectories"
        assert traj_dir.is_dir(), "trajectories/ directory not created"
        # trajectory_every=1 → file for every iteration
        assert (traj_dir / "iter_000001.json").exists()
        assert (traj_dir / "iter_000002.json").exists()

    def test_best_trajectory_written(self, run_info, experiment_config):
        run_dir = experiment_config.run_output_dir / run_info.run_id
        assert (run_dir / "trajectories" / "best.json").exists()
        assert (run_dir / "trajectories" / "best_meta.json").exists()

    def test_periodic_json_valid(self, run_info, experiment_config):
        run_dir = experiment_config.run_output_dir / run_info.run_id
        path = run_dir / "trajectories" / "iter_000001.json"
        data = json.loads(path.read_text())
        for key in ("experiment_name", "run_id", "iteration", "episode_reward_mean",
                    "grid_size", "agent_count", "max_steps", "obs_mode", "episode"):
            assert key in data, f"missing key: {key}"
        assert data["experiment_name"] == "ray-test"
        assert data["iteration"] == 1

    def test_best_json_episode_has_steps(self, run_info, experiment_config):
        run_dir = experiment_config.run_output_dir / run_info.run_id
        data = json.loads((run_dir / "trajectories" / "best.json").read_text())
        ep = data["episode"]
        assert isinstance(ep["steps"], list)
        assert len(ep["steps"]) > 0
        step = ep["steps"][0]
        for key in ("step", "action", "reward", "done",
                    "cursor_x", "cursor_y", "cursor_z", "voxel_count", "placed"):
            assert key in step, f"step missing key: {key}"

    def test_best_meta_iteration_valid(self, run_info, experiment_config):
        run_dir = experiment_config.run_output_dir / run_info.run_id
        meta = json.loads((run_dir / "trajectories" / "best_meta.json").read_text())
        assert meta["iteration"] in (1, 2)
        assert isinstance(meta["episode_reward_mean"], float)
