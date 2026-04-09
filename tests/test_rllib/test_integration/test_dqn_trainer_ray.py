"""
Ray integration tests for DQNTrainer and RainbowTrainer.

Requires ray[rllib] + torch + theseo_core.  Run with:
    pytest tests/integration/test_dqn_trainer_ray.py -m ray -v

Module-scoped shared fixtures limit algorithm builds to 2 per trainer.
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
from theseo_anysearch.rllib.trainer.dqn import DQNTrainer
from theseo_anysearch.rllib.trainer.rainbow import RainbowTrainer


RAY_DQN_YAML = textwrap.dedent("""\
    env:
      stl_path: /tmp/toy.stl
      scale: 1.0
      agent_count: 1
      max_steps: 10
      seed: 0

    training:
      algorithm: dqn
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
      train_batch_size: 32
      n_step: 1
      num_atoms: 1
      dueling: true
      double_q: true
      noisy: false
      replay_buffer_capacity: 1000
      warmup_steps: 0

    model_config:
      hidden_sizes: [32]
      activation: relu
""")

RAY_RAINBOW_YAML = textwrap.dedent("""\
    env:
      stl_path: /tmp/toy.stl
      scale: 1.0
      agent_count: 1
      max_steps: 10
      seed: 0

    training:
      algorithm: rainbow
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
      train_batch_size: 32
      n_step: 1
      num_atoms: 5
      v_min: -1.0
      v_max: 1.0
      dueling: true
      double_q: true
      noisy: false
      replay_buffer_capacity: 1000
      warmup_steps: 0
      prioritized_replay_alpha: 0.6
      prioritized_replay_beta: 0.4

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
    return tmp_path_factory.mktemp("ray_dqn")


@pytest.fixture(scope="module")
def dqn_settings(ray_session, module_tmp):
    """Provide dqn settings."""
    from theseo_anysearch.settings import load_settings
    p = module_tmp / "dqn_settings.yaml"
    p.write_text(RAY_DQN_YAML.format(output_dir=str(module_tmp / "dqn")))
    return load_settings(p)


@pytest.fixture(scope="module")
def dqn_trained(ray_session, dqn_settings):
    """Run 2 DQN training iterations once for the whole module."""
    t = DQNTrainer(dqn_settings)
    results = t.train()
    return t, results


@pytest.fixture(scope="module")
def rainbow_settings(ray_session, module_tmp):
    """Provide rainbow settings."""
    from theseo_anysearch.settings import load_settings
    p = module_tmp / "rainbow_settings.yaml"
    p.write_text(RAY_RAINBOW_YAML.format(output_dir=str(module_tmp / "rainbow")))
    return load_settings(p)


@pytest.fixture(scope="module")
def rainbow_trained(ray_session, rainbow_settings):
    """Run 2 Rainbow training iterations once for the whole module."""
    t = RainbowTrainer(rainbow_settings)
    results = t.train()
    return t, results


# ---------------------------------------------------------------------------
# 1. DQN — Build
# ---------------------------------------------------------------------------

class TestDQNTrainerRayBuild:
    """Tests DQNTrainerRayBuild."""
    def test_algo_has_train_method(self, dqn_trained):
        trainer, _ = dqn_trained
        assert hasattr(trainer._algo, "train")

    def test_algo_has_save_method(self, dqn_trained):
        trainer, _ = dqn_trained
        assert hasattr(trainer._algo, "save")

    def test_algo_has_restore_method(self, dqn_trained):
        trainer, _ = dqn_trained
        assert hasattr(trainer._algo, "restore")


# ---------------------------------------------------------------------------
# 2. DQN — Train
# ---------------------------------------------------------------------------

class TestDQNTrainerRayTrain:
    """Tests DQNTrainerRayTrain."""
    def test_returns_list_of_train_results(self, dqn_trained):
        _, results = dqn_trained
        assert isinstance(results, list)
        assert all(isinstance(r, TrainResult) for r in results)

    def test_result_count_matches_iterations(self, dqn_trained, dqn_settings):
        _, results = dqn_trained
        assert len(results) == dqn_settings.training.iterations

    def test_episode_reward_mean_is_finite_float(self, dqn_trained):
        _, results = dqn_trained
        for r in results:
            assert isinstance(r.episode_reward_mean, float)
            assert math.isfinite(r.episode_reward_mean)

    def test_episodes_total_non_negative(self, dqn_trained):
        _, results = dqn_trained
        assert all(r.episodes_total >= 0 for r in results)

    def test_elapsed_s_positive(self, dqn_trained):
        _, results = dqn_trained
        assert all(r.elapsed_s > 0 for r in results)


# ---------------------------------------------------------------------------
# 3. DQN — Checkpoint
# ---------------------------------------------------------------------------

class TestDQNTrainerRayCheckpoint:
    """Tests DQNTrainerRayCheckpoint."""
    def test_checkpoint_dir_created(self, dqn_trained, dqn_settings):
        ckpt_root = Path(dqn_settings.training.output_dir) / "checkpoints"
        assert ckpt_root.is_dir()
        assert any(ckpt_root.iterdir())

    def test_state_json_is_valid(self, dqn_trained, dqn_settings):
        ckpt_root = Path(dqn_settings.training.output_dir) / "checkpoints"
        state_files = list(ckpt_root.glob("*/state.json"))
        assert state_files, "No state.json found"
        state = json.loads(state_files[0].read_text())
        assert "iteration" in state

    def test_latest_json_points_to_checkpoint(self, dqn_trained, dqn_settings):
        ckpt_root = Path(dqn_settings.training.output_dir) / "checkpoints"
        latest = json.loads((ckpt_root / "latest.json").read_text())
        assert Path(latest["path"]).is_dir()


# ---------------------------------------------------------------------------
# 4. Rainbow — Train
# ---------------------------------------------------------------------------

class TestRainbowTrainerRayTrain:
    """Tests RainbowTrainerRayTrain."""
    def test_returns_list_of_train_results(self, rainbow_trained):
        _, results = rainbow_trained
        assert isinstance(results, list)
        assert all(isinstance(r, TrainResult) for r in results)

    def test_result_count_matches_iterations(self, rainbow_trained, rainbow_settings):
        _, results = rainbow_trained
        assert len(results) == rainbow_settings.training.iterations

    def test_episode_reward_mean_is_finite_float(self, rainbow_trained):
        _, results = rainbow_trained
        for r in results:
            assert isinstance(r.episode_reward_mean, float)
            assert math.isfinite(r.episode_reward_mean)

    def test_checkpoint_dir_created(self, rainbow_trained, rainbow_settings):
        ckpt_root = Path(rainbow_settings.training.output_dir) / "checkpoints"
        assert ckpt_root.is_dir()
        assert any(ckpt_root.iterdir())
