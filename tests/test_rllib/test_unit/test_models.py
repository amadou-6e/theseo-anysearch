"""Unit tests for shared config models and algorithm/model-specific config types."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from theseo_anysearch.models import (
    AlgorithmConfig,
    AnyscaleConfig,
    EnvConfig,
    ModelConfig,
    Settings,
    TrainingConfig,
)


def make_settings(**overrides) -> dict:
    """Make settings."""
    base = dict(
        env=EnvConfig(stl_path=Path("/tmp/test.stl"), agent_count=1),
        training=TrainingConfig(algorithm="ppo", model="voxel_encoder"),
        anyscale=AnyscaleConfig(
            cluster_env="c", compute_config="cc", project="p"
        ),
        algorithm_config=AlgorithmConfig(),
        model_config=ModelConfig(),
    )
    base.update(overrides)
    return base


class TestEnvConfig:
    """Tests EnvConfig."""
    def test_can_construct_without_stl_path(self):
        cfg = EnvConfig()
        assert cfg.geometry__stl_path is None

    def test_defaults(self):
        cfg = EnvConfig(stl_path=Path("/tmp/a.stl"))
        assert cfg.geometry__scale == 1.0
        assert cfg.agent_count == 4
        assert cfg.max_steps == 200
        assert cfg.seed == 42

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            EnvConfig(stl_path=Path("/tmp/a.stl"), unknown_field=1)


class TestTrainingConfig:
    """Tests TrainingConfig."""
    def test_defaults(self):
        cfg = TrainingConfig(algorithm="ppo", model="voxel_encoder")
        assert cfg.runner == "local"
        assert cfg.iterations == 100
        assert cfg.checkpoint_interval == 10
        assert cfg.video_every == 10
        assert cfg.require_gpu is False

    def test_require_gpu_can_be_set(self):
        cfg = TrainingConfig(algorithm="ppo", require_gpu=True)
        assert cfg.require_gpu is True

    def test_requires_algorithm(self):
        with pytest.raises(ValidationError):
            TrainingConfig(model="voxel_encoder")

    def test_runner_literal(self):
        with pytest.raises(ValidationError):
            TrainingConfig(algorithm="ppo", model="m", runner="invalid")


class TestAnyscaleConfig:
    """Tests AnyscaleConfig."""
    def test_round_trip(self):
        cfg = AnyscaleConfig(cluster_env="env", compute_config="cc", project="proj")
        assert cfg.cluster_env == "env"
        assert cfg.compute_config == "cc"
        assert cfg.project == "proj"

    def test_rejects_extra(self):
        with pytest.raises(ValidationError):
            AnyscaleConfig(cluster_env="e", compute_config="c", project="p", extra=True)


class TestAlgorithmConfig:
    """Tests AlgorithmConfig."""
    def test_defaults(self):
        cfg = AlgorithmConfig()
        assert cfg.lr == pytest.approx(3e-4)
        assert cfg.gamma == pytest.approx(0.99)
        assert cfg.train_batch_size == 4096

    def test_rejects_extra(self):
        with pytest.raises(ValidationError):
            AlgorithmConfig(unknown_field=1)

    def test_custom_lr(self):
        cfg = AlgorithmConfig(lr=1e-3)
        assert cfg.lr == pytest.approx(1e-3)


class TestPPOConfig:
    """Tests PPOConfig."""
    def setup_method(self):
        from theseo_anysearch.rllib.algorithms.models import PPOConfig
        self.PPOConfig = PPOConfig

    def test_inherits_algorithm_config(self):
        cfg = self.PPOConfig()
        assert isinstance(cfg, AlgorithmConfig)

    def test_ppo_defaults(self):
        cfg = self.PPOConfig()
        assert cfg.clip_param == pytest.approx(0.2)
        assert cfg.num_sgd_iter == 10
        assert cfg.lambda_ == pytest.approx(0.95)
        assert cfg.kl_coeff == pytest.approx(0.2)

    def test_inherits_base_fields(self):
        cfg = self.PPOConfig(lr=5e-4, gamma=0.95)
        assert cfg.lr == pytest.approx(5e-4)
        assert cfg.gamma == pytest.approx(0.95)

    def test_ppo_field_override(self):
        cfg = self.PPOConfig(clip_param=0.3, num_sgd_iter=5)
        assert cfg.clip_param == pytest.approx(0.3)
        assert cfg.num_sgd_iter == 5

    def test_rejects_extra(self):
        with pytest.raises(ValidationError):
            self.PPOConfig(unknown_field=1)


class TestSACConfig:
    """Tests SACConfig."""
    def setup_method(self):
        from theseo_anysearch.rllib.algorithms.models import SACConfig
        self.SACConfig = SACConfig

    def test_inherits_algorithm_config(self):
        cfg = self.SACConfig()
        assert isinstance(cfg, AlgorithmConfig)

    def test_sac_defaults(self):
        cfg = self.SACConfig()
        assert cfg.tau == pytest.approx(0.005)
        assert cfg.target_entropy == "auto"
        assert cfg.n_step == 1

    def test_sac_field_override(self):
        cfg = self.SACConfig(tau=0.01, n_step=3)
        assert cfg.tau == pytest.approx(0.01)
        assert cfg.n_step == 3

    def test_inherits_base_fields(self):
        cfg = self.SACConfig(lr=1e-4)
        assert cfg.lr == pytest.approx(1e-4)


class TestModelConfig:
    """Tests ModelConfig."""
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.hidden_sizes == [256, 256]
        assert cfg.activation == "relu"

    def test_custom_hidden_sizes(self):
        cfg = ModelConfig(hidden_sizes=[128, 64, 32])
        assert cfg.hidden_sizes == [128, 64, 32]

    def test_activation_literal_valid(self):
        for act in ("relu", "tanh", "elu"):
            cfg = ModelConfig(activation=act)
            assert cfg.activation == act

    def test_activation_literal_invalid(self):
        with pytest.raises(ValidationError):
            ModelConfig(activation="sigmoid")

    def test_rejects_extra(self):
        with pytest.raises(ValidationError):
            ModelConfig(unknown=True)


class TestVoxelEncoderConfig:
    """Tests VoxelEncoderConfig."""
    def setup_method(self):
        from theseo_anysearch.rllib.models.models import VoxelEncoderConfig
        self.VoxelEncoderConfig = VoxelEncoderConfig

    def test_inherits_model_config(self):
        cfg = self.VoxelEncoderConfig()
        assert isinstance(cfg, ModelConfig)

    def test_encoder_defaults(self):
        cfg = self.VoxelEncoderConfig()
        assert cfg.use_position_encoding is True
        assert cfg.encoder_depth == 3

    def test_inherits_base_fields(self):
        cfg = self.VoxelEncoderConfig(hidden_sizes=[128], activation="tanh")
        assert cfg.hidden_sizes == [128]
        assert cfg.activation == "tanh"

    def test_encoder_field_override(self):
        cfg = self.VoxelEncoderConfig(use_position_encoding=False, encoder_depth=5)
        assert cfg.use_position_encoding is False
        assert cfg.encoder_depth == 5

    def test_rejects_extra(self):
        with pytest.raises(ValidationError):
            self.VoxelEncoderConfig(unknown_key=1)


class TestDQNConfig:
    """Tests DQNConfig."""
    def setup_method(self):
        from theseo_anysearch.rllib.algorithms.models import DQNConfig
        self.DQNConfig = DQNConfig

    def test_inherits_algorithm_config(self):
        assert isinstance(self.DQNConfig(), AlgorithmConfig)

    def test_defaults(self):
        cfg = self.DQNConfig()
        assert cfg.n_step == 1
        assert cfg.num_atoms == 1
        assert cfg.dueling is True
        assert cfg.double_q is True
        assert cfg.noisy is False
        assert cfg.replay_buffer_capacity == 50_000
        assert cfg.warmup_steps == 0

    def test_field_override(self):
        cfg = self.DQNConfig(n_step=3, num_atoms=51, noisy=True, replay_buffer_capacity=1000)
        assert cfg.n_step == 3
        assert cfg.num_atoms == 51
        assert cfg.noisy is True
        assert cfg.replay_buffer_capacity == 1000

    def test_inherits_base_fields(self):
        cfg = self.DQNConfig(lr=1e-4, gamma=0.95)
        assert cfg.lr == pytest.approx(1e-4)
        assert cfg.gamma == pytest.approx(0.95)

    def test_rejects_extra(self):
        with pytest.raises(ValidationError):
            self.DQNConfig(unknown=1)


class TestRainbowConfig:
    """Tests RainbowConfig."""
    def setup_method(self):
        from theseo_anysearch.rllib.algorithms.models import DQNConfig, RainbowConfig
        self.RainbowConfig = RainbowConfig
        self.DQNConfig = DQNConfig

    def test_inherits_dqn_config(self):
        assert isinstance(self.RainbowConfig(), self.DQNConfig)

    def test_defaults(self):
        cfg = self.RainbowConfig()
        assert cfg.n_step == 3
        assert cfg.num_atoms == 51
        assert cfg.noisy is True
        assert cfg.v_min == pytest.approx(-10.0)
        assert cfg.v_max == pytest.approx(10.0)
        assert cfg.prioritized_replay_alpha == pytest.approx(0.6)
        assert cfg.prioritized_replay_beta == pytest.approx(0.4)

    def test_field_override(self):
        cfg = self.RainbowConfig(v_min=-5.0, v_max=5.0, n_step=1)
        assert cfg.v_min == pytest.approx(-5.0)
        assert cfg.v_max == pytest.approx(5.0)
        assert cfg.n_step == 1

    def test_inherits_base_fields(self):
        cfg = self.RainbowConfig(lr=5e-5)
        assert cfg.lr == pytest.approx(5e-5)

    def test_rejects_extra(self):
        with pytest.raises(ValidationError):
            self.RainbowConfig(unknown=1)


class TestTD3Config:
    """Tests TD3Config."""
    def setup_method(self):
        from theseo_anysearch.rllib.algorithms.models import TD3Config
        self.TD3Config = TD3Config

    def test_inherits_algorithm_config(self):
        assert isinstance(self.TD3Config(), AlgorithmConfig)

    def test_defaults(self):
        cfg = self.TD3Config()
        assert cfg.twin_q is True
        assert cfg.policy_delay == 2
        assert cfg.smooth_target_policy is True
        assert cfg.target_noise == pytest.approx(0.2)
        assert cfg.target_noise_clip == pytest.approx(0.5)
        assert cfg.tau == pytest.approx(0.005)

    def test_field_override(self):
        cfg = self.TD3Config(policy_delay=1, tau=0.01)
        assert cfg.policy_delay == 1
        assert cfg.tau == pytest.approx(0.01)

    def test_inherits_base_fields(self):
        cfg = self.TD3Config(lr=1e-4, gamma=0.95)
        assert cfg.lr == pytest.approx(1e-4)
        assert cfg.gamma == pytest.approx(0.95)

    def test_rejects_extra(self):
        with pytest.raises(ValidationError):
            self.TD3Config(unknown=1)


class TestDDPGConfig:
    """Tests DDPGConfig."""
    def setup_method(self):
        from theseo_anysearch.rllib.algorithms.models import DDPGConfig
        self.DDPGConfig = DDPGConfig

    def test_inherits_algorithm_config(self):
        assert isinstance(self.DDPGConfig(), AlgorithmConfig)

    def test_defaults(self):
        cfg = self.DDPGConfig()
        assert cfg.tau == pytest.approx(0.002)
        assert cfg.l2_reg == pytest.approx(1e-6)

    def test_field_override(self):
        cfg = self.DDPGConfig(tau=0.005, l2_reg=1e-5)
        assert cfg.tau == pytest.approx(0.005)
        assert cfg.l2_reg == pytest.approx(1e-5)

    def test_inherits_base_fields(self):
        cfg = self.DDPGConfig(lr=1e-4)
        assert cfg.lr == pytest.approx(1e-4)

    def test_rejects_extra(self):
        with pytest.raises(ValidationError):
            self.DDPGConfig(unknown=1)


class TestSettings:
    """Tests Settings."""
    def test_constructs(self):
        s = Settings(**make_settings())
        assert s.env.seed == 42
        assert s.training.algorithm == "ppo"

    def test_rejects_extra_top_level(self):
        with pytest.raises(ValidationError):
            Settings(**make_settings(), unknown_key="x")

    def test_model_cfg_attribute_accessible(self):
        s = Settings(**make_settings())
        assert isinstance(s.model_cfg, ModelConfig)

    def test_model_cfg_via_alias_key(self):
        # "model_config" is the YAML/alias key; Pydantic populates model_cfg
        s = Settings(
            env=EnvConfig(stl_path=Path("/tmp/x.stl"), agent_count=1),
            training=TrainingConfig(algorithm="ppo"),
            anyscale=AnyscaleConfig(cluster_env="c", compute_config="cc", project="p"),
            algorithm_config=AlgorithmConfig(),
            model_config=ModelConfig(),
        )
        assert isinstance(s.model_cfg, ModelConfig)

    def test_round_trip(self):
        s = Settings(**make_settings())
        dumped = s.model_dump(by_alias=True)
        s2 = Settings(**dumped)
        assert s2.env.seed == s.env.seed
        assert s2.training.algorithm == s.training.algorithm
        assert s2.algorithm_config.lr == pytest.approx(s.algorithm_config.lr)
