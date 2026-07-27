"""SAC trainer integration and configuration wiring."""

from __future__ import annotations

from typing import Any

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.models import Settings
from theseo_anysearch.rllib.algorithms.models import SACConfig
from theseo_anysearch.rllib.trainer.base import Trainer, _detect_num_gpus, _resolve_pool_dir
from theseo_anysearch.rllib.trainer.ppo import _ensure_ray_runtime


class SACTrainer(Trainer):
    """
    SAC trainer backed by ray.rllib.algorithms.sac.SAC (legacy API stack).

    Supports discrete action spaces (Discrete(3) VoxelEnv) via RLlib's discrete
    SAC policy. For continuous environments, set the appropriate action space on
    the registered env.
    """

    algorithm_name = "sac"

    @classmethod
    def from_settings(cls, config: Settings) -> "SACTrainer":
        return cls(config)

    @staticmethod
    def build_algorithm_from_settings(config: Settings) -> Any:
        from ray.rllib.algorithms.sac import SACConfig as RllibSACConfig

        _ensure_ray_runtime(str(config.training.output_dir), config.training.num_env_runners)

        env = config.env
        algo_cfg = config.algorithm_config
        if not isinstance(algo_cfg, SACConfig):
            algo_cfg = SACConfig(**algo_cfg.model_dump())

        env_config = {
            "stl_path": str(env.stl_path) if env.stl_path else None,
            "scale": env.scale,
            "scale_range": env.scale_range,
            "agent_count": env.agent_count,
            "max_steps": env.max_steps,
            "seed": env.seed,
            "obs_mode": env.obs_mode,
            "box_radius": env.box_radius,
            "box_radii": env.box_radii,
            "ray_max_len": env.ray_max_len,
            "include_voxel_count": env.include_voxel_count,
            "grid_size": env.grid_size,
            "trail_mode": env.trail_mode,
            "geometry_boxes": env.geometry_boxes,
            "geometry_pool": _resolve_pool_dir(env.geometry_pool),
            "geometry_padding": env.geometry_padding,
            "waypoints_file": env.waypoints_file,
            "step_cost": env.step_cost,
            "collision_cost": env.collision_cost,
            "goal_reward": env.goal_reward,
            "distance_shaping": env.distance_shaping,
            "distance_reward_mode": env.distance_reward_mode,
            "zone_reward_min": env.zone_reward_min,
            "zone_reward_max": env.zone_reward_max,
            "zone_reward_curve": env.zone_reward_curve,
        }
        env_id = VoxelEnv.register_with_ray(env_config=env_config)

        from theseo_anysearch.rllib.models import build_rllib_model_dict
        model_cfg = config.model_cfg
        rllib_model = build_rllib_model_dict(model_cfg)

        rllib_config = (
            RllibSACConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env=env_id, env_config=env_config)
            .training(
                lr=algo_cfg.lr,
                gamma=algo_cfg.gamma,
                train_batch_size=algo_cfg.train_batch_size,
                tau=algo_cfg.tau,
                n_step=algo_cfg.n_step,
                num_steps_sampled_before_learning_starts=algo_cfg.warmup_steps,
                policy_model_config=rllib_model,
                q_model_config=rllib_model,
                # Ray legacy SAC requires MultiAgentPrioritizedReplayBuffer;
                # the new-API default (EpisodeReplayBuffer) is rejected by validate().
                replay_buffer_config={
                    "type": "MultiAgentPrioritizedReplayBuffer",
                    "capacity": 50_000,
                    "prioritized_replay_alpha": 0.5,
                    "prioritized_replay_beta": 0.4,
                    "prioritized_replay_eps": 1e-6,
                },
            )
            .resources(num_gpus=_detect_num_gpus(config.training.require_gpu, num_gpus=config.training.num_gpus))
            .framework("torch")
        )

        rllib_config.num_env_runners = config.training.num_env_runners
        return rllib_config.build_algo()

    def _build_algorithm(self) -> Any:
        return self.build_algorithm_from_settings(self._config)
