"""Rainbow DQN trainer integration and configuration wiring."""

from __future__ import annotations

from typing import Any

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.models import Settings
from theseo_anysearch.rllib.algorithms.models import RainbowConfig
from theseo_anysearch.rllib.trainer.base import Trainer, _detect_num_gpus
from theseo_anysearch.rllib.trainer.parallel_evaluation import configure_rllib_evaluation
from theseo_anysearch.rllib.trainer.ppo import _ensure_ray_runtime


class RainbowTrainer(Trainer):
    """
    Rainbow DQN trainer backed by ray.rllib.algorithms.dqn.DQN (legacy API stack).

    Enables the full Rainbow suite: distributional Q (num_atoms>1), noisy nets,
    dueling architecture, double Q, n-step returns, and prioritised replay (PER).
    """

    algorithm_name = "rainbow"

    @classmethod
    def from_settings(cls, config: Settings) -> "RainbowTrainer":
        return cls(config)

    @staticmethod
    def build_algorithm_from_settings(config: Settings) -> Any:
        from ray.rllib.algorithms.dqn import DQNConfig as RllibDQNConfig

        _ensure_ray_runtime(
            str(config.training.output_dir),
            config.training.num_env_runners
            + config.training.evaluation_num_env_runners,
        )

        env = config.env
        algo_cfg = config.algorithm_config
        if not isinstance(algo_cfg, RainbowConfig):
            algo_cfg = RainbowConfig(**algo_cfg.model_dump())

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
                v_min=algo_cfg.v_min,
                v_max=algo_cfg.v_max,
                dueling=algo_cfg.dueling,
                double_q=algo_cfg.double_q,
                noisy=algo_cfg.noisy,
                replay_buffer_config={
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": algo_cfg.replay_buffer_capacity,
                    "prioritized_replay_alpha": algo_cfg.prioritized_replay_alpha,
                    "prioritized_replay_beta": algo_cfg.prioritized_replay_beta,
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
            num_env_runners=config.training.evaluation_num_env_runners,
        )
        return rllib_config.build_algo()

    def _build_algorithm(self) -> Any:
        return self.build_algorithm_from_settings(self._config)
