"""
Ray integration tests for PPOTrainer.

Requires ray[rllib] + torch + theseo_core wheel.  Run with:
    pytest tests/integration/test_ppo_trainer_ray.py -m ray -v

Module-scoped shared fixtures keep algorithm builds to 2 (train + resume).
"""
from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path

import pytest

ray = pytest.importorskip("ray", reason="ray not installed")
pytestmark = pytest.mark.ray

from theseo_anysearch.rllib.trainer.base import TrainResult
from theseo_anysearch.rllib.trainer.ppo import PPOTrainer, _build_rllib_ppo


RAY_TEST_YAML = textwrap.dedent("""    env:
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
      output_dir: {output_dir}
      video_every: 999

    anyscale:
      cluster_env: x
      compute_config: y
      project: z

    algorithm_config:
      lr: 1.0e-3
      gamma: 0.99
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
    return tmp_path_factory.mktemp("ray_ppo")


@pytest.fixture(scope="module")
def shared_settings(ray_session, module_tmp):
    """Settings loaded once; output_dir points at module_tmp."""
    from theseo_anysearch.settings import load_settings
    p = module_tmp / "settings.yaml"
    p.write_text(RAY_TEST_YAML.format(output_dir=str(module_tmp)))
    return load_settings(p)


@pytest.fixture(scope="module")
def trained(ray_session, shared_settings):
    """
    Run 2 training iterations once for the whole module.
    Returns (trainer, results).
    """
    t = PPOTrainer(shared_settings)
    results = t.train()
    return t, results


# ---------------------------------------------------------------------------
# 1. Build
# ---------------------------------------------------------------------------

class TestPPOTrainerRayBuild:
    def test_algo_has_train_method(self, trained):
        trainer, _ = trained
        assert hasattr(trainer._algo, "train")

    def test_algo_has_save_method(self, trained):
        trainer, _ = trained
        assert hasattr(trainer._algo, "save")

    def test_algo_has_restore_method(self, trained):
        trainer, _ = trained
        assert hasattr(trainer._algo, "restore")


# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------

class TestPPOTrainerRayTrain:
    def test_returns_list_of_train_results(self, trained):
        _, results = trained
        assert isinstance(results, list)
        assert all(isinstance(r, TrainResult) for r in results)

    def test_result_count_matches_iterations(self, trained, shared_settings):
        _, results = trained
        assert len(results) == shared_settings.training.iterations

    def test_episode_reward_mean_is_finite_float(self, trained):
        _, results = trained
        for r in results:
            assert isinstance(r.episode_reward_mean, float)
            assert math.isfinite(r.episode_reward_mean)

    def test_episodes_total_non_negative(self, trained):
        _, results = trained
        assert all(r.episodes_total >= 0 for r in results)

    def test_elapsed_s_positive(self, trained):
        _, results = trained
        assert all(r.elapsed_s > 0 for r in results)


# ---------------------------------------------------------------------------
# 3. Checkpoint
# ---------------------------------------------------------------------------

class TestPPOTrainerRayCheckpoint:
    def test_checkpoint_dir_created(self, trained, shared_settings):
        ckpt_root = Path(shared_settings.training.output_dir) / "checkpoints"
        assert ckpt_root.is_dir()
        assert any(ckpt_root.iterdir())

    def test_state_json_is_valid(self, trained, shared_settings):
        ckpt_root = Path(shared_settings.training.output_dir) / "checkpoints"
        state_files = list(ckpt_root.glob("*/state.json"))
        assert state_files, "No state.json found"
        state = json.loads(state_files[0].read_text())
        assert "iteration" in state

    def test_latest_json_points_to_checkpoint(self, trained, shared_settings):
        ckpt_root = Path(shared_settings.training.output_dir) / "checkpoints"
        latest = json.loads((ckpt_root / "latest.json").read_text())
        assert Path(latest["path"]).is_dir()


# ---------------------------------------------------------------------------
# 4. Resume
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def resumed(ray_session, trained, shared_settings):
    """
    Resume from the shared checkpoint.
    Stops the first algo first to free resources.
    """
    trainer, _ = trained
    if trainer._algo is not None:
        try:
            trainer._algo.stop()
        except Exception:
            pass
        trainer._algo = None

    trainer2 = PPOTrainer(shared_settings)
    ok = trainer2.resume()
    assert ok is True, "No checkpoint found to resume from"
    return trainer2


class TestPPOTrainerRayResume:
    def test_resumed_is_true(self, resumed):
        assert resumed._iteration > 0

    def test_iteration_matches_trained_count(self, resumed, shared_settings):
        assert resumed._iteration == shared_settings.training.iterations

    def test_train_after_resume_returns_list(self, ray_session, resumed):
        results = resumed.train()
        assert isinstance(results, list)
