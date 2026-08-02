"""
Ray integration tests for PPOTrainer.

Requires ray[rllib] + torch + theseo_core wheel.  Run with:
    pytest tests/integration/test_ppo_trainer_ray.py -m ray -v

Module-scoped shared fixtures keep algorithm builds to 2 (train + resume).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ray = pytest.importorskip("ray", reason="ray not installed")
pytestmark = pytest.mark.ray

from theseo_anysearch.rllib.trainer.results import TrainResult
from theseo_anysearch.rllib.algorithms.ppo import PPOTrainer, _build_rllib_ppo

RAY_TEST_YAML = """env:
    stl_path: {stl_path}
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
    num_env_runners: 1
    num_envs_per_env_runner: 2
    num_gpus_per_env_runner: 0.0

evaluation:
    episodes: 2
    num_env_runners: 0
    num_envs_per_env_runner: 2

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
"""

# ---------------------------------------------------------------------------
# Module-scoped shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def module_tmp(tmp_path_factory):
    """Provide module tmp."""
    return tmp_path_factory.mktemp("ray_ppo")


@pytest.fixture(scope="module")
def shared_settings(ray_session, module_tmp):
    """Settings loaded once; output_dir points at module_tmp."""
    from theseo_anysearch.settings import load_settings
    stl_path = module_tmp / "toy.stl"
    stl_path.write_text(
        "solid toy\n"
        "facet normal 0 0 1\n"
        "outer loop\n"
        "vertex 0 0 0\n"
        "vertex 1 0 0\n"
        "vertex 0 1 0\n"
        "endloop\n"
        "endfacet\n"
        "endsolid toy\n",
        encoding="utf-8",
    )
    p = module_tmp / "settings.yaml"
    p.write_text(
        RAY_TEST_YAML.format(
            output_dir=json.dumps(str(module_tmp)),
            stl_path=json.dumps(str(stl_path)),
        ))
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
    """Tests PPOTrainerRayBuild."""

    def test_algo_has_train_method(self, trained):
        trainer, _ = trained
        assert hasattr(trainer._algo, "train")

    def test_algo_has_save_method(self, trained):
        trainer, _ = trained
        assert hasattr(trainer._algo, "save")

    def test_algo_has_restore_method(self, trained):
        trainer, _ = trained
        assert hasattr(trainer._algo, "restore")

    def test_algo_uses_vectorized_cpu_rollouts(self, trained):
        trainer, _ = trained
        assert trainer._algo.config.num_env_runners == 1
        assert trainer._algo.config.num_envs_per_env_runner == 2
        assert trainer._algo.config.num_gpus_per_env_runner == 0.0

    def test_trainer_uses_vectorized_evaluation(self, trained):
        trainer, results = trained
        assert trainer._config.evaluation.num_envs_per_env_runner == 2
        assert all(result.evaluation_episodes == 2 for result in results)


# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------


class TestPPOTrainerRayTrain:
    """Tests PPOTrainerRayTrain."""

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
    """Tests PPOTrainerRayCheckpoint."""

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
    """Tests PPOTrainerRayResume."""

    def test_resumed_is_true(self, resumed):
        assert resumed._iteration > 0

    def test_iteration_matches_trained_count(self, resumed, shared_settings):
        assert resumed._iteration == shared_settings.training.iterations

    def test_train_after_resume_returns_list(self, ray_session, resumed):
        results = resumed.train()
        assert isinstance(results, list)
