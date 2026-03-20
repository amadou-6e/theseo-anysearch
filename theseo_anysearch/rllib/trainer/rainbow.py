from __future__ import annotations

from typing import Any

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.models import Settings
from theseo_anysearch.rllib.algorithms.models import RainbowConfig
from theseo_anysearch.rllib.trainer.base import Trainer, _detect_num_gpus
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

        _ensure_ray_runtime(str(config.training.output_dir), config.training.num_env_runners)

        env = config.env
        algo_cfg = config.algorithm_config
        if not isinstance(algo_cfg, RainbowConfig):
            algo_cfg = RainbowConfig(**algo_cfg.model_dump())

        env_config = {
            "stl_path": str(env.stl_path),
            "scale": env.scale,
            "agent_count": env.agent_count,
            "max_steps": env.max_steps,
            "seed": env.seed,
            "obs_mode": env.obs_mode,
            "box_radius": env.box_radius,
            "box_radii": env.box_radii,
            "ray_max_len": env.ray_max_len,
            "trail_mode": env.trail_mode,
            "geometry_boxes": env.geometry_boxes,
            "waypoints_file": env.waypoints_file,
            "step_cost": env.step_cost,
            "goal_reward": env.goal_reward,
            "distance_shaping": env.distance_shaping,
        }
        env_id = VoxelEnv.register_with_ray(env_config=env_config)

        from theseo_anysearch.rllib.models import _build_rllib_model_dict
        model_cfg = config.model_cfg
        rllib_model = _build_rllib_model_dict(model_cfg)

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
        return rllib_config.build_algo()

    def _build_algorithm(self) -> Any:
        return self.build_algorithm_from_settings(self._config)
