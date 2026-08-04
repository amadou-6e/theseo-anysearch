"""Single-agent PPO trainer integration for the voxel environment."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.settings import Settings
from theseo_anysearch.rllib.algorithms.models import PPOConfig
from theseo_anysearch.rllib.trainer.trainer import Trainer
from theseo_anysearch.rllib.trainer.runtime import _detect_num_gpus, _resolve_pool_dir
from theseo_anysearch.rllib.trainer.evaluation.parallel import configure_rllib_evaluation


# TODO! instead of private functions, create ustils file
def _build_rllib_ppo(config: Settings) -> Any:
    return PPOTrainer.build_algorithm_from_settings(config)


def _configure_rllib_env_runners(rllib_config: Any, training: Any) -> Any:
    """Apply rollout parallelism, vectorization, and inference resources."""
    return rllib_config.env_runners(
        num_env_runners=training.num_env_runners,
        num_envs_per_env_runner=training.num_envs_per_env_runner,
        num_gpus_per_env_runner=training.num_gpus_per_env_runner,
        max_requests_in_flight_per_env_runner=(
            training.max_requests_in_flight_per_env_runner
        ),
    )


def _log_stage(message: str) -> None:
    """Print a timestamped PPO startup message for foreground debugging."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[ppo] {ts} {message}", flush=True)


def _append_stage_log(output_dir: str, scope: str, message: str) -> None:
    """Append a timestamped stage marker to the run-local debug log."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = Path(output_dir, "debug_stage.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[{scope}] {ts} {message}\n")


def _set_rllib_storage_path(output_dir: str) -> Path:
    """Keep Ray Trainable temporary directories inside the writable run."""
    from ray.tune.trainable import trainable as ray_trainable

    storage_path = Path(output_dir, "rllib")
    storage_path.mkdir(parents=True, exist_ok=True)
    ray_trainable.DEFAULT_STORAGE_PATH = str(storage_path)
    return storage_path


def _ensure_ray_runtime(output_dir: str, num_env_runners: int = 0) -> None:
    import os as _os
    from pathlib import Path

    import ray

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _set_rllib_storage_path(output_dir)

    if ray.is_initialized():
        _log_stage("Ray runtime already initialized")
        _append_stage_log(output_dir, "ppo", "Ray runtime already initialized")
        _write_ray_runtime_metadata(output_path)
        return

    _os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray_root = VoxelEnvPathHelper.ray_root(output_dir)
    # num_env_runners workers @ 1 CPU each + 1 for the learner/driver
    num_cpus = max(num_env_runners + 1, 1)
    _log_stage(
        f"Initializing Ray runtime with num_cpus={num_cpus} temp_dir={ray_root}"
    )
    _append_stage_log(
        output_dir,
        "ppo",
        f"Initializing Ray runtime with num_cpus={num_cpus} temp_dir={ray_root}",
    )
    ray.init(
        num_cpus=num_cpus,
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        _temp_dir=str(ray_root),
    )
    _log_stage("Ray runtime initialized")
    _append_stage_log(output_dir, "ppo", "Ray runtime initialized")
    _write_ray_runtime_metadata(output_path)


# TODO! why is this a class? make helper if no reason
class VoxelEnvPathHelper:
    """Path helper for PPO runtime artifacts tied to voxel environment training."""

    @staticmethod
    def ray_root(output_dir: str) -> str:
        """Return the Ray runtime directory for a training run.

        Parameters
        ----------
        output_dir : str
            Training output directory.

        Returns
        -------
        str
            Absolute Ray runtime directory.
        """
        import os
        import tempfile
        from pathlib import Path

        # Use the system temp dir for Ray's internal files (logs, spilled objects)
        # to avoid exceeding Windows MAX_PATH when output_dir is deeply nested
        # (e.g. under pytest's tmp_path hierarchy).
        candidate = Path(output_dir, "ray")
        # Ray appends ~120 chars (session dir + spilled-objects hash) to this path.
        # Windows MAX_PATH is 260, so trigger the fallback when base exceeds 130 chars.
        if len(str(candidate.resolve())) > 130:
            ray_root = Path(tempfile.gettempdir(), f"anysearch_ray_{os.getpid()}")
        else:
            ray_root = candidate
        ray_root.mkdir(parents=True, exist_ok=True)
        return str(ray_root.resolve())


def _write_ray_runtime_metadata(output_dir: "Path") -> None:
    import json

    session_dir = None
    try:
        import ray

        node = getattr(ray._private.worker, "_global_node", None)
        if node is not None:
            session_dir = node.get_session_dir_path()
    except Exception:
        session_dir = None

    output_dir.joinpath("ray_runtime.json").write_text(
        json.dumps(
            {
                "ray_output_dir": str(output_dir.joinpath("ray")),
                "ray_session_dir": session_dir,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


class PPOTrainer(Trainer):
    """
    PPO trainer backed by ray.rllib.algorithms.ppo.PPO.

    The trainer reads PPO hyper-parameters from config.algorithm_config
    and environment settings from config.env.
    """

    algorithm_name = "ppo"

    @classmethod
    def from_settings(cls, config: Settings) -> "PPOTrainer":
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
        config: Settings, env_config: dict | None = None
    ) -> Any:
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
        from ray.rllib.algorithms.ppo import PPOConfig as RllibPPOConfig

        _log_stage("Starting PPO algorithm build")
        _append_stage_log(
            str(config.training.output_dir),
            "ppo",
            "Starting PPO algorithm build",
        )
        _ensure_ray_runtime(
            str(config.training.output_dir),
            config.training.num_env_runners
            + config.evaluation.num_env_runners,
        )

        env = config.env
        algo_cfg = config.algorithm_config
        if not isinstance(algo_cfg, PPOConfig):
            algo_cfg = PPOConfig(**algo_cfg.model_dump())

        env_config = env_config or env.to_runtime_dict()
        env_config["geometry_pool"] = _resolve_pool_dir(env.geometry.pool)
        env_config["debug_log_path"] = str(
            Path(config.training.output_dir, "env_debug.log")
        )
        _log_stage(
            f"Registering VoxelEnv with obs_mode={env.observation.mode} grid_size={env.geometry__grid_size} max_steps={env.max_steps}"
        )
        _append_stage_log(
            str(config.training.output_dir),
            "ppo",
            (
                "Registering VoxelEnv with "
                f"obs_mode={env.observation.mode} grid_size={env.geometry__grid_size} max_steps={env.max_steps}"
            ),
        )
        env_id = VoxelEnv.register_with_ray(env_config=env_config)
        _log_stage(f"Registered VoxelEnv as {env_id}")
        _append_stage_log(
            str(config.training.output_dir),
            "ppo",
            f"Registered VoxelEnv as {env_id}",
        )

        from theseo_anysearch.rllib.models import build_rllib_model_dict
        model_cfg = config.model_cfg
        _log_stage(
            f"Building RLlib model config type={type(model_cfg).__name__}"
        )
        _append_stage_log(
            str(config.training.output_dir),
            "ppo",
            f"Building RLlib model config type={type(model_cfg).__name__}",
        )
        rllib_model = build_rllib_model_dict(model_cfg)
        minibatch_size = algo_cfg.minibatch_size or min(
            128, algo_cfg.train_batch_size)

        _log_stage(
            "Constructing RLlib PPOConfig "
            f"train_batch_size={algo_cfg.train_batch_size} "
            f"num_env_runners={config.training.num_env_runners} "
            f"num_envs_per_env_runner={config.training.num_envs_per_env_runner}"
        )
        _append_stage_log(
            str(config.training.output_dir),
            "ppo",
            (
                "Constructing RLlib PPOConfig "
                f"train_batch_size={algo_cfg.train_batch_size} "
                f"num_env_runners={config.training.num_env_runners} "
                f"num_envs_per_env_runner={config.training.num_envs_per_env_runner}"
            ),
        )
        rllib_config = (RllibPPOConfig().api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        ).environment(env=env_id, env_config=env_config).training(
            lr=algo_cfg.lr,
            gamma=algo_cfg.gamma,
            train_batch_size=algo_cfg.train_batch_size,
            minibatch_size=minibatch_size,
            clip_param=algo_cfg.clip_param,
            num_epochs=algo_cfg.num_sgd_iter,
            lambda_=algo_cfg.lambda_,
            kl_coeff=algo_cfg.kl_coeff,
            grad_clip=algo_cfg.grad_clip,
            model=rllib_model,
        ).resources(num_gpus=_detect_num_gpus(config.training.require_gpu, num_gpus=config.training.num_gpus)).framework("torch"))

        rllib_config = _configure_rllib_env_runners(
            rllib_config, config.training
        )
        rllib_config = configure_rllib_evaluation(
            rllib_config,
            num_env_runners=config.evaluation.num_env_runners,
        )

        _log_stage("Calling RLlib build_algo()")
        _append_stage_log(
            str(config.training.output_dir),
            "ppo",
            "Calling RLlib build_algo()",
        )
        algo = rllib_config.build_algo()
        _log_stage("RLlib algorithm build completed")
        _append_stage_log(
            str(config.training.output_dir),
            "ppo",
            "RLlib algorithm build completed",
        )
        return algo

    def _build_algorithm(self) -> Any:
        return self.build_algorithm_from_settings(
            self._config, self._env_config_dict()
        )

class MultiAgentVoxelPPOTrainer(Trainer):
    """
    Multi-agent PPO trainer for MultiVoxelEnv (PettingZoo parallel API).

    All agents share a single "shared_policy". Uses RLlib's legacy API stack
    with ParallelPettingZooEnv to wrap the PettingZoo environment.
    """

    algorithm_name = "multi_agent_voxel_ppo"

    @staticmethod
    def build_algorithm_from_settings(config: Settings) -> Any:
        """Build the configured multi-agent RLlib algorithm.

        Parameters
        ----------
        config : Settings
            Validated experiment settings.

        Returns
        -------
        Any
            Built multi-agent RLlib algorithm instance.
        """
        from ray.rllib.algorithms.ppo import PPOConfig as RllibPPOConfig
        from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
        from ray.rllib.policy.policy import PolicySpec
        from ray.tune.registry import register_env

        from theseo_anysearch.environments.pettingzoo.multi_voxel_env import MultiVoxelEnv

        _ensure_ray_runtime(
            str(config.training.output_dir),
            config.training.num_env_runners
            + config.evaluation.num_env_runners,
        )

        env_cfg = config.env
        algo_cfg = config.algorithm_config
        if not isinstance(algo_cfg, PPOConfig):
            algo_cfg = PPOConfig(**algo_cfg.model_dump())

        env_config = env_cfg.to_runtime_dict()

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
        rllib_config = configure_rllib_evaluation(
            rllib_config,
            num_env_runners=config.evaluation.num_env_runners,
        )

        return rllib_config.build_algo()

    def _build_algorithm(self) -> Any:
        return self.build_algorithm_from_settings(self._config)
