from __future__ import annotations

from typing import Any

from theseo_anysearch.models import Settings
from theseo_anysearch.rllib.algorithms.models import PPOConfig
from theseo_anysearch.rllib.trainer.base import Trainer, _detect_num_gpus
from theseo_anysearch.rllib.trainer.ppo import VoxelEnvPathHelper, _ensure_ray_runtime


class MultiAgentVoxelPPOTrainer(Trainer):
    """
    Multi-agent PPO trainer for MultiVoxelEnv (PettingZoo parallel API).

    All agents share a single "shared_policy". Uses RLlib's legacy API stack
    with ParallelPettingZooEnv to wrap the PettingZoo environment.
    """

    algorithm_name = "multi_agent_voxel_ppo"

    @staticmethod
    def build_algorithm_from_settings(config: Settings) -> Any:
        from ray.rllib.algorithms.ppo import PPOConfig as RllibPPOConfig
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        from ray.rllib.policy.policy import PolicySpec
        from ray.tune.registry import register_env

        from theseo_anysearch.environments.pettingzoo.multi_voxel_env import MultiVoxelEnv

        _ensure_ray_runtime(str(config.training.output_dir), config.training.num_env_runners)

        env_cfg = config.env
        algo_cfg = config.algorithm_config
        if not isinstance(algo_cfg, PPOConfig):
            algo_cfg = PPOConfig(**algo_cfg.model_dump())

        env_config = {
            "agent_count":      env_cfg.agent_count,
            "max_steps":        env_cfg.max_steps,
            "seed":             env_cfg.seed,
            "trail_mode":       env_cfg.trail_mode,
            "geometry_boxes":   env_cfg.geometry_boxes,
            "step_cost":        env_cfg.step_cost,
            "goal_reward":      env_cfg.goal_reward,
            "distance_shaping": env_cfg.distance_shaping,
            "collision_cost":   env_cfg.collision_cost,
            "box_radius":       getattr(env_cfg, "box_radius", 2),
            "ray_max_len":      getattr(env_cfg, "ray_max_len", 16),
            "distance_metric":  getattr(env_cfg, "distance_metric", "euclidean"),
        }

        env_id = "multi_voxel_ppo_env"
        register_env(
            env_id,
            lambda cfg: ParallelPettingZooEnv(MultiVoxelEnv(cfg or env_config)),
        )

        # Build a temp env to introspect spaces for the policy spec.
        _tmp = MultiVoxelEnv(env_config)
        obs_space = _tmp.observation_space("agent_0")
        act_space = _tmp.action_space("agent_0")

        minibatch_size = algo_cfg.minibatch_size or min(128, algo_cfg.train_batch_size)

        rllib_config = (
            RllibPPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env=env_id, env_config=env_config)
            .multi_agent(
                policies={
                    "shared_policy": PolicySpec(
                        observation_space=obs_space,
                        action_space=act_space,
                    )
                },
                policy_mapping_fn=lambda agent_id, episode=None, worker=None, **kw: "shared_policy",
            )
            .training(
                lr=algo_cfg.lr,
                gamma=algo_cfg.gamma,
                train_batch_size=algo_cfg.train_batch_size,
                minibatch_size=minibatch_size,
                clip_param=algo_cfg.clip_param,
                num_epochs=algo_cfg.num_sgd_iter,
                lambda_=algo_cfg.lambda_,
                kl_coeff=algo_cfg.kl_coeff,
            )
            .resources(num_gpus=_detect_num_gpus(config.training.require_gpu, num_gpus=config.training.num_gpus))
            .framework("torch")
        )
        rllib_config.num_env_runners = config.training.num_env_runners

        return rllib_config.build_algo()

    def _build_algorithm(self) -> Any:
        return self.build_algorithm_from_settings(self._config)
