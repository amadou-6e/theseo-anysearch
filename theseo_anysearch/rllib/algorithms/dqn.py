"""DQN trainer integration and configuration wiring."""

from __future__ import annotations

from typing import Any

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.settings import Settings
from theseo_anysearch.rllib.algorithms.models import DQNConfig
from theseo_anysearch.rllib.trainer.trainer import Trainer
from theseo_anysearch.rllib.trainer.runtime import _detect_num_gpus, _resolve_pool_dir
from theseo_anysearch.rllib.trainer.evaluation.parallel import configure_rllib_evaluation
from theseo_anysearch.rllib.algorithms.ppo import _ensure_ray_runtime


class DQNTrainer(Trainer):
    """
    DQN trainer backed by ray.rllib.algorithms.dqn.DQN (legacy API stack).

    Note: DQNConfig.train_batch_size is the replay buffer sample batch size per
    gradient update, not the on-policy rollout length as in PPO. Use smaller
    values (e.g. 32) than you would for PPO (e.g. 4096).
    """

    algorithm_name = "dqn"

    @classmethod
    def from_settings(cls, config: Settings) -> "DQNTrainer":
        """Construct the algorithm adapter from project settings.

        Parameters
        ----------
        config : Settings
            Validated experiment settings.

        Returns
        -------
        Trainer
            Configured algorithm adapter.
        """
        return cls(config)

    @staticmethod
    def build_algorithm_from_settings(config: Settings) -> Any:
        """Build the configured RLlib algorithm.

        Parameters
        ----------
        config : Settings
            Validated experiment settings.

        Returns
        -------
        Any
            Built RLlib algorithm instance.
        """
        from ray.rllib.algorithms.dqn import DQNConfig as RllibDQNConfig

        _ensure_ray_runtime(
            str(config.training.output_dir),
            config.training.num_env_runners
            + config.evaluation.num_env_runners,
        )

        env = config.env
        algo_cfg = config.algorithm_config
        if not isinstance(algo_cfg, DQNConfig):
            algo_cfg = DQNConfig(**algo_cfg.model_dump())

        env_config = env.to_runtime_dict()
        env_config["geometry_pool"] = _resolve_pool_dir(env.geometry.pool)
        env_id = VoxelEnv.register_with_ray(env_config=env_config)

        from theseo_anysearch.rllib.models import build_rllib_model_dict
        model_cfg = config.model_cfg
        rllib_model = build_rllib_model_dict(model_cfg)

        rllib_config = (
            RllibDQNConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env=env_id, env_config=env_config)
            .training(
                lr=algo_cfg.lr,
                gamma=algo_cfg.gamma,
                train_batch_size=algo_cfg.train_batch_size,
                n_step=algo_cfg.n_step,
                num_atoms=algo_cfg.num_atoms,
                dueling=algo_cfg.dueling,
                double_q=algo_cfg.double_q,
                noisy=algo_cfg.noisy,
                replay_buffer_config={
                    "type": "MultiAgentReplayBuffer",
                    "capacity": algo_cfg.replay_buffer_capacity,
                },
                num_steps_sampled_before_learning_starts=algo_cfg.warmup_steps,
                model=rllib_model,
            )
            .resources(num_gpus=_detect_num_gpus(config.training.require_gpu, num_gpus=config.training.num_gpus))
            .framework("torch")
        )

        rllib_config.num_env_runners = config.training.num_env_runners
        rllib_config = configure_rllib_evaluation(
            rllib_config,
            num_env_runners=config.evaluation.num_env_runners,
        )
        return rllib_config.build_algo()

    def _build_algorithm(self) -> Any:
        return self.build_algorithm_from_settings(self._config)
