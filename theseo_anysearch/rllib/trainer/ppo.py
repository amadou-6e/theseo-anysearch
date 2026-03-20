from __future__ import annotations

from typing import Any

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.models import Settings
from theseo_anysearch.rllib.algorithms.models import PPOConfig
from theseo_anysearch.rllib.trainer.base import Trainer, _detect_num_gpus


# TODO! instead of private functions, create ustils file
def _build_rllib_ppo(config: Settings) -> Any:
    return PPOTrainer.build_algorithm_from_settings(config)


def _ensure_ray_runtime(output_dir: str, num_env_runners: int = 0) -> None:
    import os as _os
    from pathlib import Path

    import ray

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if ray.is_initialized():
        _write_ray_runtime_metadata(output_path)
        return

    _os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray_root = VoxelEnvPathHelper.ray_root(output_dir)
    # num_env_runners workers @ 1 CPU each + 1 for the learner/driver
    num_cpus = max(num_env_runners + 1, 1)
    ray.init(
        num_cpus=num_cpus,
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        _temp_dir=str(ray_root),
    )
    _write_ray_runtime_metadata(output_path)


# TODO! why is this a class? make helper if no reason
class VoxelEnvPathHelper:

    @staticmethod
    def ray_root(output_dir: str) -> str:
        import os
        import tempfile
        from pathlib import Path

        # Use the system temp dir for Ray's internal files (logs, spilled objects)
        # to avoid exceeding Windows MAX_PATH when output_dir is deeply nested
        # (e.g. under pytest's tmp_path hierarchy).
        candidate = Path(output_dir) / "ray"
        # Ray appends ~120 chars (session dir + spilled-objects hash) to this path.
        # Windows MAX_PATH is 260, so trigger the fallback when base exceeds 130 chars.
        if len(str(candidate.resolve())) > 130:
            ray_root = Path(tempfile.gettempdir()) / f"anysearch_ray_{os.getpid()}"
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
        return cls(config)

    @staticmethod
    def build_algorithm_from_settings(config: Settings) -> Any:
        from ray.rllib.algorithms.ppo import PPOConfig as RllibPPOConfig

        _ensure_ray_runtime(str(config.training.output_dir), config.training.num_env_runners)

        env = config.env
        algo_cfg = config.algorithm_config
        if not isinstance(algo_cfg, PPOConfig):
            algo_cfg = PPOConfig(**algo_cfg.model_dump())

        env_config = {
            "stl_path": str(env.stl_path) if env.stl_path else None,
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
        minibatch_size = algo_cfg.minibatch_size or min(
            128, algo_cfg.train_batch_size)

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

        rllib_config.num_env_runners = config.training.num_env_runners

        return rllib_config.build_algo()

    def _build_algorithm(self) -> Any:
        return self.build_algorithm_from_settings(self._config)
