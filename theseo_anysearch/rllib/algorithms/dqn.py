"""DQN trainer integration and configuration wiring."""

from __future__ import annotations

from typing import Any

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.settings import Settings
from theseo_anysearch.rllib.algorithms.models import DQNConfig
from theseo_anysearch.rllib.trainer.trainer import Trainer
from theseo_anysearch.rllib.trainer.runtime import _detect_num_gpus, _resolve_pool_dir
from theseo_anysearch.rllib.trainer.evaluation.parallel import (
    bind_anysearch_evaluation_function,
    configure_rllib_evaluation,
)
from theseo_anysearch.rllib.algorithms.ppo import (
    _configure_rllib_env_runners,
    _ensure_ray_runtime,
)
from theseo_anysearch.rllib.algorithms.dqn_runtime import (
    AnySearchDQN,
    AnySearchDQNTorchLearner,
)


def _dqn_replay_buffer_config(algo_cfg: DQNConfig) -> dict[str, Any]:
    """Build a modern DQN replay buffer configuration from public settings."""
    replay_type = (
        "EpisodeReplayBuffer"
        if algo_cfg.replay_buffer_type == "uniform"
        else "PrioritizedEpisodeReplayBuffer"
    )
    replay = {
        "type": replay_type,
        "capacity": algo_cfg.replay_buffer_capacity,
    }
    if algo_cfg.replay_buffer_type == "prioritized":
        replay["alpha"] = getattr(algo_cfg, "prioritized_replay_alpha", 0.6)
        replay["beta"] = getattr(algo_cfg, "prioritized_replay_beta", 0.4)
    return replay


def _learner_gpu_allocation(config: Settings) -> float:
    """Resolve the GPU allocation assigned to each Learner."""
    configured = config.training.num_gpus_per_learner
    return _detect_num_gpus(
        config.training.require_gpu,
        num_gpus=(configured if configured is not None else config.training.num_gpus),
    )

class DQNTrainer(Trainer):
    """
    DQN trainer backed by RLlib's RLModule, Learner, and EnvRunner stack.

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
    def build_algorithm_from_settings(
        config: Settings,
        env_config: dict[str, Any] | None = None,
    ) -> Any:
        """Build the configured RLlib algorithm.

        Parameters
        ----------
        config : Settings
            Validated experiment settings.
        env_config : dict[str, Any] or None, optional
            Fully resolved runtime environment configuration, including copied
            extension paths. When omitted, build the basic environment mapping
            directly from ``config``.

        Returns
        -------
        Any
            Built RLlib algorithm instance.
        """
        from ray.rllib.algorithms.dqn import DQNConfig as RllibDQNConfig

        _ensure_ray_runtime(
            str(config.training.output_dir),
            config.training.num_env_runners
            + config.evaluation.num_env_runners
            + (
                config.training.num_learners
                * config.training.num_cpus_per_learner
            ),
        )

        env = config.env
        algo_cfg = config.algorithm_config
        if not isinstance(algo_cfg, DQNConfig):
            algo_cfg = DQNConfig(**algo_cfg.model_dump())

        env_config = env_config or env.to_runtime_dict()
        env_config["geometry_pool"] = _resolve_pool_dir(env.geometry.pool)
        if env.waypoint_curriculum.enabled:
            from theseo_anysearch.rllib.trainer.waypoint_curriculum import (
                WaypointCurriculum,
            )

            initial_curriculum = WaypointCurriculum(
                env.waypoint_curriculum,
                env_config,
            )
            initial_state = initial_curriculum.state
            if initial_state.waypoints:
                env_config["waypoint_route"] = {
                    "start": initial_state.start,
                    "waypoints": initial_state.waypoints,
                }
            else:
                env_config["waypoints"] = {
                    "start": initial_state.start,
                    "goal": initial_state.goal,
                }
        env_id = VoxelEnv.register_with_ray(env_config=env_config)

        from theseo_anysearch.rllib.models import (
            build_rllib_rl_module_model_config,
        )
        model_cfg = config.model_cfg
        rllib_model = build_rllib_rl_module_model_config(model_cfg)

        rllib_config = (
            RllibDQNConfig(algo_class=AnySearchDQN)
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .environment(env=env_id, env_config=env_config)
            .training(
                lr=algo_cfg.lr,
                gamma=algo_cfg.gamma,
                train_batch_size_per_learner=algo_cfg.train_batch_size,
                n_step=algo_cfg.n_step,
                num_atoms=algo_cfg.num_atoms,
                dueling=algo_cfg.dueling,
                double_q=algo_cfg.double_q,
                noisy=algo_cfg.noisy,
                replay_buffer_config=_dqn_replay_buffer_config(algo_cfg),
                num_steps_sampled_before_learning_starts=algo_cfg.warmup_steps,
                training_intensity=algo_cfg.training_intensity,
            )
            .rl_module(model_config=rllib_model)
            .learners(
                num_learners=config.training.num_learners,
                num_cpus_per_learner=config.training.num_cpus_per_learner,
                num_gpus_per_learner=_learner_gpu_allocation(config),
                learner_class=AnySearchDQNTorchLearner,
                learner_config_dict={
                    "report_td_errors": algo_cfg.replay_buffer_type == "prioritized",
                    "weight_sync_interval": config.training.weight_sync_interval,
                },
            )
            .framework("torch")
        )

        rllib_config = _configure_rllib_env_runners(
            rllib_config, config.training
        )
        rllib_config = rllib_config.env_runners(
            rollout_fragment_length=algo_cfg.rollout_fragment_length,
        )
        rllib_config = configure_rllib_evaluation(
            rllib_config,
            num_env_runners=config.evaluation.num_env_runners,
            parallel_to_training=config.evaluation.parallel_to_training,
            frequency=config.evaluation.frequency,
            env_config=env_config,
            episodes=config.evaluation.episodes,
            seed=config.evaluation.seed,
            num_envs_per_env_runner=config.evaluation.num_envs_per_env_runner,
        )
        return bind_anysearch_evaluation_function(rllib_config.build_algo())

    def _build_algorithm(self) -> Any:
        return self.build_algorithm_from_settings(
            self._config,
            env_config=self._env_config_dict(),
        )
