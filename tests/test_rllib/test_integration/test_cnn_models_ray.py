"""
Ray integration tests for VoxelBox2DCNN, VoxelBox3DCNN, and
VoxelHierarchicalBox3DCNN with PPOTrainer.

Requires ray[rllib] + torch + theseo_core wheel.  Run with:
    pytest tests/integration/test_cnn_models_ray.py -m ray -v
"""
from __future__ import annotations

import math
import textwrap

import pytest

ray = pytest.importorskip("ray", reason="ray not installed")
pytestmark = pytest.mark.ray

from theseo_anysearch.rllib.trainer.ppo import PPOTrainer  # noqa: E402
from theseo_anysearch.settings import load_settings  # noqa: E402


CNN_YAML_TEMPLATE = textwrap.dedent("""
    env:
      stl_path: /tmp/toy.stl
      scale: 1.0
      agent_count: 1
      max_steps: 10
      seed: 0
      obs_mode: "box"
      box_radius: {box_radius}

    training:
      algorithm: ppo
      model: voxel_encoder
      runner: local
      iterations: 1
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
      custom_model: "{custom_model}"
      custom_model_config:
        box_radius: {box_radius}
        conv_channels: [8, 16]
        fc_hiddens: [64]
""")

HIERARCHICAL_YAML_TEMPLATE = textwrap.dedent("""
    env:
      stl_path: /tmp/toy.stl
      scale: 1.0
      agent_count: 1
      max_steps: 10
      seed: 0
      obs_mode: "hierarchical_box"
      box_radii: {box_radii}

    training:
      algorithm: ppo
      model: voxel_encoder
      runner: local
      iterations: 1
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
      custom_model: "voxel_hierarchical_box_3d_cnn"
      custom_model_config:
        box_radii: {box_radii}
        conv_channels: [8, 16]
        fc_hiddens: [64]
""")


def _make_yaml(tmp_path, custom_model: str, box_radius: int) -> str:
    return CNN_YAML_TEMPLATE.format(
        output_dir=str(tmp_path),
        custom_model=custom_model,
        box_radius=box_radius,
    )


def _make_hierarchical_yaml(tmp_path, box_radii: list[int]) -> str:
    return HIERARCHICAL_YAML_TEMPLATE.format(
        output_dir=str(tmp_path),
        box_radii=box_radii,
    )


@pytest.fixture(scope="module")
def ray_session():
    """Session-scoped Ray fixture (reuse from conftest or re-init)."""
    import ray as _ray
    if not _ray.is_initialized():
        _ray.init(num_cpus=1, ignore_reinit_error=True, include_dashboard=False)
    yield
    # Don't shut down — session fixture in conftest handles that


@pytest.fixture(scope="module")
def trained_2d(ray_session, tmp_path_factory):
    """Provide trained 2d."""
    tmp = tmp_path_factory.mktemp("cnn_2d")
    yaml_text = _make_yaml(tmp, "voxel_box_2d_cnn", box_radius=2)
    yaml_path = tmp / "exp.yaml"
    yaml_path.write_text(yaml_text)
    settings = load_settings(yaml_path)
    trainer = PPOTrainer(settings)
    results = trainer.train()
    return results, trainer, tmp


@pytest.fixture(scope="module")
def trained_3d(ray_session, tmp_path_factory):
    """Provide trained 3d."""
    tmp = tmp_path_factory.mktemp("cnn_3d")
    yaml_text = _make_yaml(tmp, "voxel_box_3d_cnn", box_radius=2)
    yaml_path = tmp / "exp.yaml"
    yaml_path.write_text(yaml_text)
    settings = load_settings(yaml_path)
    trainer = PPOTrainer(settings)
    results = trainer.train()
    return results, trainer, tmp


class TestPPO2DCNN:
    """Tests PPO2DCNN."""
    def test_returns_train_result(self, trained_2d):
        results, _, _ = trained_2d
        assert len(results) == 1

    def test_episode_reward_mean_finite(self, trained_2d):
        results, _, _ = trained_2d
        assert math.isfinite(results[0].episode_reward_mean)

    def test_checkpoint_created(self, trained_2d):
        _, trainer, tmp = trained_2d
        ckpt_dir = tmp / "checkpoints"
        assert ckpt_dir.exists()
        assert any(ckpt_dir.iterdir())


class TestPPO3DCNN:
    """Tests PPO3DCNN."""
    def test_returns_train_result(self, trained_3d):
        results, _, _ = trained_3d
        assert len(results) == 1

    def test_episode_reward_mean_finite(self, trained_3d):
        results, _, _ = trained_3d
        assert math.isfinite(results[0].episode_reward_mean)

    def test_checkpoint_created(self, trained_3d):
        _, trainer, tmp = trained_3d
        ckpt_dir = tmp / "checkpoints"
        assert ckpt_dir.exists()
        assert any(ckpt_dir.iterdir())


class TestPPO3DCNNRadius3:
    """Tests PPO3DCNNRadius3."""
    def test_radius3_trains(self, ray_session, tmp_path):
        yaml_text = _make_yaml(tmp_path, "voxel_box_3d_cnn", box_radius=3)
        yaml_path = tmp_path / "exp.yaml"
        yaml_path.write_text(yaml_text)
        settings = load_settings(yaml_path)
        trainer = PPOTrainer(settings)
        results = trainer.train()
        assert len(results) == 1
        assert math.isfinite(results[0].episode_reward_mean)


class TestCNNCheckpointRestore:
    """Tests CNNCheckpointRestore."""
    def test_checkpoint_and_resume(self, ray_session, tmp_path):
        yaml_text = _make_yaml(tmp_path, "voxel_box_2d_cnn", box_radius=2)
        yaml_path = tmp_path / "exp.yaml"
        yaml_path.write_text(yaml_text)
        settings = load_settings(yaml_path)
        trainer = PPOTrainer(settings)
        results = trainer.train()
        assert results[0].iteration == 1

        # Resume: build a fresh trainer pointing at the same output_dir
        trainer2 = PPOTrainer(settings)
        resumed = trainer2.resume()
        assert resumed is True
        assert trainer2._iteration == 1


@pytest.fixture(scope="module")
def trained_hierarchical(ray_session, tmp_path_factory):
    """Provide trained hierarchical."""
    tmp = tmp_path_factory.mktemp("cnn_hier")
    yaml_text = _make_hierarchical_yaml(tmp, [1, 3])
    yaml_path = tmp / "exp.yaml"
    yaml_path.write_text(yaml_text)
    settings = load_settings(yaml_path)
    trainer = PPOTrainer(settings)
    results = trainer.train()
    return results, trainer, tmp


class TestPPOHierarchicalBox3DCNN:
    """Tests PPOHierarchicalBox3DCNN."""
    def test_returns_train_result(self, trained_hierarchical):
        results, _, _ = trained_hierarchical
        assert len(results) == 1

    def test_episode_reward_mean_finite(self, trained_hierarchical):
        results, _, _ = trained_hierarchical
        assert math.isfinite(results[0].episode_reward_mean)

    def test_checkpoint_created(self, trained_hierarchical):
        _, trainer, tmp = trained_hierarchical
        ckpt_dir = tmp / "checkpoints"
        assert ckpt_dir.exists()
        assert any(ckpt_dir.iterdir())

    def test_iteration_1(self, trained_hierarchical):
        results, _, _ = trained_hierarchical
        assert results[0].iteration == 1


class TestHierarchicalRadiiVariants:
    """Verify that different radius combinations train without error."""

    def test_two_tight_radii(self, ray_session, tmp_path):
        yaml_text = _make_hierarchical_yaml(tmp_path, [1, 2])
        yaml_path = tmp_path / "exp.yaml"
        yaml_path.write_text(yaml_text)
        settings = load_settings(yaml_path)
        trainer = PPOTrainer(settings)
        results = trainer.train()
        assert math.isfinite(results[0].episode_reward_mean)

    def test_three_radii(self, ray_session, tmp_path):
        yaml_text = _make_hierarchical_yaml(tmp_path, [1, 2, 3])
        yaml_path = tmp_path / "exp.yaml"
        yaml_path.write_text(yaml_text)
        settings = load_settings(yaml_path)
        trainer = PPOTrainer(settings)
        results = trainer.train()
        assert math.isfinite(results[0].episode_reward_mean)

    def test_coarse_radii(self, ray_session, tmp_path):
        yaml_text = _make_hierarchical_yaml(tmp_path, [2, 4])
        yaml_path = tmp_path / "exp.yaml"
        yaml_path.write_text(yaml_text)
        settings = load_settings(yaml_path)
        trainer = PPOTrainer(settings)
        results = trainer.train()
        assert math.isfinite(results[0].episode_reward_mean)


class TestHierarchicalCheckpointRestore:
    """Tests HierarchicalCheckpointRestore."""
    def test_checkpoint_and_resume(self, ray_session, tmp_path):
        yaml_text = _make_hierarchical_yaml(tmp_path, [1, 3])
        yaml_path = tmp_path / "exp.yaml"
        yaml_path.write_text(yaml_text)
        settings = load_settings(yaml_path)
        trainer = PPOTrainer(settings)
        results = trainer.train()
        assert results[0].iteration == 1

        trainer2 = PPOTrainer(settings)
        resumed = trainer2.resume()
        assert resumed is True
        assert trainer2._iteration == 1
