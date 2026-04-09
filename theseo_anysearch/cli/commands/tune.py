"""CLI helpers for hyperparameter search and sweep execution."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Optional

import typer

from theseo_anysearch.models import AnyscaleConfig, EnvConfig, ModelConfig, Settings, TrainingConfig
from theseo_anysearch.rllib.algorithms.models import PPOConfig

app = typer.Typer(help="[deprecated] Use: anysearch run <dir> --tag <tag>")


def _print_error(message: str, **extra: Any) -> None:
    payload = {"error": message}
    payload.update(extra)
    print(json.dumps(payload), file=sys.stderr)


def _print_run_overview(
    *,
    run_tag: str,
    num_samples: int,
    max_concurrent: int,
    max_iterations: int,
    metric: str,
    mode: str,
    output_dir: "Path",
    tb_logdir: str,
    mlflow_tracking_uri: str,
    parent_run_id: str,
) -> None:
    sep = "-" * 60
    lines = [
        "",
        sep,
        f"  Tune run: {run_tag}",
        sep,
        f"  Trials      : {num_samples} total, {max_concurrent} concurrent",
        f"  Iterations  : up to {max_iterations} per trial",
        f"  Optimising  : {metric} ({mode})",
        "",
        "  Outputs",
        f"    Trial data : {output_dir.resolve() / run_tag}",
        f"    Metadata   : {output_dir.resolve() / 'ray_runtime.json'}",
        "",
        "  Monitoring",
        f"    TensorBoard: tensorboard --logdir \"{tb_logdir}\"",
    ]
    if mlflow_tracking_uri:
        lines.append(f"    MLflow URI : {mlflow_tracking_uri}")
        if parent_run_id:
            lines.append(f"    MLflow run : {parent_run_id}")
    lines += [
        "",
        "  Tips",
        "    - Ray prints a progress table to stdout -- scroll up to see it.",
        "    - Each trial writes its own subdir under the trial data path above.",
        "    - The best config + metrics are printed as JSON when all trials finish.",
        "    - Re-open TensorBoard at any time during the run -- it live-updates.",
        sep,
        "",
    ]
    print("\n".join(lines), flush=True)


def _print_existing_run(run_dir: "Path", output_dir: "Path") -> None:
    """Print a human-readable summary of an existing run and suggest next steps."""
    import math

    sep = "-" * 60
    meta_path = output_dir / "ray_runtime.json"
    tb_logdir = str(run_dir)

    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            tb_logdir = meta.get("tensorboard_logdir", tb_logdir)
        except Exception:
            pass

    # Scan per-trial subdirectories for params.json + result.json (JSONL).
    trial_dirs = sorted(p for p in run_dir.iterdir() if p.is_dir() and (p / "params.json").exists())
    total = len(trial_dirs)
    completed: list[dict] = []   # trials with at least one result line
    errored: list[str] = []

    for td in trial_dirs:
        result_lines = []
        result_path = td / "result.json"
        if result_path.exists():
            for line in result_path.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line:
                    try:
                        result_lines.append(json.loads(line))
                    except Exception:
                        pass
        if result_lines:
            last = result_lines[-1]
            params = json.loads((td / "params.json").read_text(encoding="utf-8"))
            completed.append({"trial": td.name, "params": params, "last_result": last})
        else:
            errored.append(td.name)

    # Find best among completed trials (maximise episode_reward_mean if present).
    best: dict | None = None
    if completed:
        def _reward(t: dict) -> float:
            v = t["last_result"].get("episode_reward_mean")
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                return -1e9
            return float(v)
        best = max(completed, key=_reward)

    lines = [
        "",
        sep,
        f"  Run exists: {run_dir.name}",
        sep,
        f"  Trials     : {total} total  |  {len(completed)} completed  |  {len(errored)} errored",
    ]

    if best:
        r = best["last_result"]
        reward = r.get("episode_reward_mean", "n/a")
        itr = r.get("training_iteration", "?")
        lines += [
            "",
            f"  Best trial : {best['trial']}",
            f"    reward   : {reward:.4f}" if isinstance(reward, float) else f"    reward   : {reward}",
            f"    iteration: {itr}",
            "    params   :",
        ]
        for k, v in sorted(best["params"].items()):
            lines.append(f"      {k:<20} {v}")
    elif errored:
        lines += [
            "",
            f"  All trials errored -- check error files under:",
            f"    {run_dir / errored[0]}",
        ]

    lines += [
        "",
        "  Outputs",
        f"    Trial data : {run_dir}",
        "",
        "  Monitoring",
        f"    TensorBoard: tensorboard --logdir \"{tb_logdir}\"",
        "",
        "  Options",
        "    --overwrite     Delete this run and start a new search.",
        "    --tag <name>    Use a different tag to start a parallel run.",
        sep,
        "",
    ]
    print("\n".join(lines), flush=True)


_CHOICE_FNS = frozenset({"choice", "grid_search"})

_VALID_TUNE_FNS = frozenset({
    "uniform", "quniform", "loguniform", "qloguniform",
    "randn", "qrandn", "randint", "qrandint",
    "lograndint", "qlograndint", "choice", "grid_search",
})


def _parse_search_space(raw: dict[str, Any], tune_module: Any) -> dict[str, Any]:
    """Convert a YAML search_space dict into ray.tune.* sampler objects.

    The YAML format uses the Ray Tune function name as the key and its positional
    arguments as a list::

        algorithm_config:
          lr:               {loguniform: [1.0e-5, 1.0e-2]}
          train_batch_size: {choice: [512, 1024, 2048]}
        model_config:
          num_layers:       {randint: [1, 5]}

    Both ``algorithm_config`` and ``model_config`` sections are flattened into a
    single dict of ``{param_name: tune_sampler}`` pairs.
    """
    result: dict[str, Any] = {}
    for section in ("algorithm_config", "model_config"):
        for param, spec in raw.get(section, {}).items():
            if not isinstance(spec, dict) or len(spec) != 1:
                raise ValueError(
                    f"search_space.{section}.{param}: expected {{fn_name: [args]}}, got {spec!r}"
                )
            fn_name, args = next(iter(spec.items()))
            if fn_name not in _VALID_TUNE_FNS:
                raise ValueError(
                    f"search_space.{section}.{param}: unknown tune function '{fn_name}'. "
                    f"Valid options: {sorted(_VALID_TUNE_FNS)}"
                )
            fn = getattr(tune_module, fn_name)
            if fn_name in _CHOICE_FNS:
                result[param] = fn(args)   # args is the values list itself
            else:
                result[param] = fn(*args)  # args are positional arguments
    return result


def _load_tune_config_yaml(config_path: Path) -> dict[str, Any]:
    """Load a tune YAML config and return the raw dict."""
    import yaml
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Tune config YAML must be a mapping, got {type(raw).__name__}")
    return raw


def _flatten_yaml_base_config(raw_cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten top-level algorithm_config + model_config into a single param dict.

    These become the fixed defaults for any param not covered by the search space.
    Search-space samples always take priority when merged in _real_trainable.
    """
    base: dict[str, Any] = {}
    for section in ("algorithm_config", "model_config"):
        for k, v in raw_cfg.get(section, {}).items():
            # Skip non-scalar values (e.g. nested dicts like custom_model_config)
            if not isinstance(v, (int, float, str, bool)):
                continue
            base[k] = v
    return base


# Params that _real_trainable knows how to consume.
_KNOWN_PPO_PARAMS: frozenset[str] = frozenset({
    "lr", "gamma", "train_batch_size", "clip_param",
    "num_sgd_iter", "lambda_", "kl_coeff", "grad_clip",
})
_KNOWN_MODEL_PARAMS: frozenset[str] = frozenset({
    "num_layers", "encoder_depth", "layer_size",
    "use_position_encoding",  # TODO: not yet wired — see VoxelEncoderConfig
})
_KNOWN_TUNE_PARAMS: frozenset[str] = _KNOWN_PPO_PARAMS | _KNOWN_MODEL_PARAMS


def _validate_tune_config(
    base_config: dict[str, Any],
    search_space: dict[str, Any],
    num_gpus: int,
) -> list[str]:
    """Return a list of human-readable warnings about the combined tune config.

    Does not raise — caller prints the warnings and decides whether to abort.
    """
    import torch

    issues: list[str] = []
    all_provided = set(base_config) | set(search_space)

    unknown = all_provided - _KNOWN_TUNE_PARAMS
    if unknown:
        issues.append(
            f"Unrecognised params (will be ignored by trainable): {sorted(unknown)}"
        )

    if num_gpus > 0 and torch.cuda.device_count() == 0:
        issues.append(
            f"num_gpus={num_gpus} requested but torch.cuda.device_count()=0. "
            "Install CUDA-enabled torch or set training.num_gpus: 0 in your YAML."
        )

    return issues


def _default_search_space(tune_module: Any, algorithm: str) -> dict[str, Any]:
    if algorithm == "ppo":
        return {
            # PPO hyperparameters
            "lr":               tune_module.loguniform(1.0e-5, 1.0e-2),
            "train_batch_size": tune_module.choice([512, 1024, 2048, 4096]),
            "gamma":            tune_module.uniform(0.95, 0.999),
            "clip_param":       tune_module.uniform(0.1, 0.3),
            "kl_coeff":         tune_module.uniform(0.05, 0.5),
            "num_sgd_iter":     tune_module.choice([5, 10, 20, 30]),
            "lambda_":          tune_module.uniform(0.9, 1.0),
            # Network architecture: depth (number of layers) and width (size per layer)
            "num_layers":       tune_module.choice([1, 2, 3, 4]),
            "layer_size":       tune_module.choice([64, 128, 256, 512]),
        }

    raise NotImplementedError(f"Unsupported tuning algorithm '{algorithm}'.")


_ALL_SCHEDULERS = (
    "asha", "pbt",
    "random", "hyperband", "bohb",
    "optuna", "cmaes", "flaml",
)


def _build_scheduler(
    scheduler_name: str,
    max_iterations: int,
    hyperparam_mutations: dict[str, Any],
    asha_config: "Any | None" = None,
) -> Any:
    from ray.tune.schedulers import (
        ASHAScheduler,
        FIFOScheduler,
        HyperBandScheduler,
        PopulationBasedTraining,
    )

    if scheduler_name == "asha":
        if asha_config is not None:
            grace_period = asha_config.grace_period
            reduction_factor = asha_config.reduction_factor
        else:
            # Default: let trials run for at least 20% of max_t before pruning.
            grace_period = max(10, max_iterations // 5)
            reduction_factor = 3
        return ASHAScheduler(
            time_attr="training_iteration",
            max_t=max_iterations,
            grace_period=min(grace_period, max_iterations),
            reduction_factor=reduction_factor,
        )

    if scheduler_name == "pbt":
        return PopulationBasedTraining(
            time_attr="training_iteration",
            perturbation_interval=max(1, min(5, max_iterations)),
            hyperparam_mutations=hyperparam_mutations,
        )

    if scheduler_name == "random":
        return FIFOScheduler()

    if scheduler_name == "hyperband":
        return HyperBandScheduler(
            time_attr="training_iteration",
            max_t=max_iterations,
            reduction_factor=3,
        )

    if scheduler_name == "bohb":
        try:
            from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
        except ImportError as exc:
            raise ValueError(
                "BOHB requires ConfigSpace: pip install ConfigSpace"
            ) from exc
        return HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=max_iterations,
            reduction_factor=3,
        )

    if scheduler_name == "optuna":
        grace_period = max(10, max_iterations // 5)
        return ASHAScheduler(
            time_attr="training_iteration",
            max_t=max_iterations,
            grace_period=min(grace_period, max_iterations),
            reduction_factor=3,
        )

    if scheduler_name in ("cmaes", "flaml"):
        return FIFOScheduler()

    raise ValueError(
        f"Unsupported scheduler '{scheduler_name}'. "
        f"Supported: {', '.join(_ALL_SCHEDULERS)}"
    )


def _build_search_alg(
    scheduler_name: str,
    metric: str,
    mode: str,
) -> Any | None:
    """Return a Ray Tune SearchAlgorithm for schedulers that need one, else None."""

    if scheduler_name == "bohb":
        try:
            from ray.tune.search.bohb import TuneBOHB
            return TuneBOHB(metric=metric, mode=mode)
        except ImportError as exc:
            raise ValueError(
                "BOHB search requires ConfigSpace: pip install ConfigSpace"
            ) from exc

    if scheduler_name == "optuna":
        try:
            from ray.tune.search.optuna import OptunaSearch
            return OptunaSearch(metric=metric, mode=mode)
        except ImportError as exc:
            raise ValueError(
                "Optuna search requires optuna: pip install optuna"
            ) from exc

    if scheduler_name == "cmaes":
        try:
            import nevergrad as ng
            from ray.tune.search.nevergrad import NevergradSearch
            return NevergradSearch(
                optimizer=ng.optimizers.ParametrizedCMA(sigma=0.5),
                metric=metric,
                mode=mode,
            )
        except ImportError as exc:
            raise ValueError(
                "CMA-ES search requires nevergrad: pip install nevergrad"
            ) from exc

    if scheduler_name == "flaml":
        try:
            from ray.tune.search.flaml import CFO
            return CFO(metric=metric, mode=mode)
        except ImportError as exc:
            raise ValueError(
                "FLAML search requires flaml: pip install flaml"
            ) from exc

    return None


def _sanitize_metric(value: float, mode: str) -> float:
    """Replace NaN/Inf with a worst-case finite sentinel so schedulers rank the trial last."""
    import math
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return -1e9 if mode == "max" else 1e9
    return value


def _real_trainable(
    config: dict[str, Any],
    stl: str,
    agents: int,
    max_steps: int,
    seed: int,
    max_iterations: int,
    output_dir: str,
    metric: str,
    mode: str,
    num_gpus: int,
    base_config: dict[str, Any],
    env_base: dict[str, Any],
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_parent_run_id: str,
    run_tag: str,
    trajectory_every: int,
    best_trajectory: bool,
) -> None:
    import torch.version  # noqa — must be imported before RLlib accesses torch.version.hip
    from ray import tune as _tune

    from theseo_anysearch.experiments.tracking import MLflowTracker
    from theseo_anysearch.experiments.models import MLflowConfig
    from theseo_anysearch.rllib.trainer.ppo import PPOTrainer

    # ray.train.get_context() is deprecated in Ray 2.54 when used from Tune.
    # Use ray.tune.get_context() instead.
    try:
        context = _tune.get_context()
        trial_id = context.get_trial_id() or "trial"
    except Exception:
        trial_id = "trial"
    # Place trial outputs under run_tag so paths match Ray Tune's native layout:
    # output_dir/run_tag/trial_{id}/ — required for --tune-dir viewer to scan correctly.
    trial_dir = Path(output_dir, run_tag, trial_id)
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Merge: YAML base defaults < trial search-space sample.
    # This means any param not in the search space falls back to the YAML value,
    # and any param not in either falls back to the Pydantic model default.
    cfg = {**base_config, **config}

    settings = Settings(
        env=EnvConfig(
            stl_path=Path(stl),
            scale=float(env_base.get("scale", 1.0)),
            agent_count=agents,
            max_steps=max_steps,
            seed=seed,
            obs_mode=env_base.get("obs_mode", "scalar"),
            box_radius=int(env_base.get("box_radius", 2)),
            ray_max_len=int(env_base.get("ray_max_len", 16)),
            trail_mode=bool(env_base.get("trail_mode", True)),
            geometry_boxes=env_base.get("geometry_boxes"),
            waypoints_file=env_base.get("waypoints_file"),
            step_cost=float(env_base.get("step_cost", -0.01)),
            collision_cost=float(env_base.get("collision_cost", 0.0)),
            goal_reward=float(env_base.get("goal_reward", 1.0)),
            distance_shaping=float(env_base.get("distance_shaping", 0.0)),
        ),
        training=TrainingConfig(
            algorithm="ppo",
            model="voxel_encoder",
            runner="local",
            iterations=max_iterations,
            checkpoint_interval=max(1, max_iterations),
            output_dir=trial_dir,
            video_every=max_iterations,
            require_gpu=num_gpus > 0,
        ),
        anyscale=AnyscaleConfig(cluster_env="", compute_config="", project=""),
        algorithm_config=PPOConfig(
            lr=float(cfg.get("lr", 3e-4)),
            gamma=float(cfg.get("gamma", 0.99)),
            train_batch_size=int(cfg.get("train_batch_size", 4096)),
            clip_param=float(cfg.get("clip_param", 0.2)),
            minibatch_size=min(128, int(cfg.get("train_batch_size", 4096))),
            num_sgd_iter=int(cfg.get("num_sgd_iter", 10)),
            lambda_=float(cfg.get("lambda_", 0.95)),
            kl_coeff=float(cfg.get("kl_coeff", 0.2)),
            grad_clip=float(cfg.get("grad_clip", 40.0)),
        ),
        model_config=ModelConfig(
            # encoder_depth (YAML alias) and num_layers (default search space) both map
            # to the number of hidden layers; layer_size sets the width of each layer.
            hidden_sizes=[int(cfg.get("layer_size", 256))]
                         * int(cfg.get("num_layers", cfg.get("encoder_depth", 2))),
        ),
    )

    # Build env_config dict matching what PPOTrainer uses, so collect_eval_episode
    # can create a VoxelEnv with the same settings as the training environment.
    env = settings.env
    _env_config_for_eval: dict[str, Any] = {
        "stl_path": str(env.stl_path) if env.stl_path else None,
        "scale": env.scale,
        "agent_count": env.agent_count,
        "max_steps": env.max_steps,
        "seed": env.seed,
        "obs_mode": env.obs_mode,
        "box_radius": env.box_radius,
        "ray_max_len": env.ray_max_len,
        "trail_mode": env.trail_mode,
        "geometry_boxes": env.geometry_boxes,
        "waypoints_file": str(env.waypoints_file) if env.waypoints_file else None,
        "step_cost": env.step_cost,
        "goal_reward": env.goal_reward,
        "distance_shaping": env.distance_shaping,
    }

    # Set up trajectory writer if enabled.
    _traj_writer = None
    if trajectory_every > 0 or best_trajectory:
        from theseo_anysearch.experiments.output import OutputStore
        from theseo_anysearch.experiments.trajectory import TrajectoryWriter
        _traj_writer = TrajectoryWriter(
            OutputStore(trial_dir),
            trajectory_every=trajectory_every,
            best_trajectory=best_trajectory,
        )

    mlflow_cfg = MLflowConfig(tracking_uri=mlflow_tracking_uri) if mlflow_tracking_uri else None
    tracker = MLflowTracker(mlflow_cfg, mlflow_experiment_name)
    tracker.start_child_run(
        run_name=trial_id,
        tags={"trial_id": trial_id},
        parent_run_id=mlflow_parent_run_id or None,
    )
    tracker.log_params({k: str(v) for k, v in config.items()})

    try:
        trainer = PPOTrainer(settings)

        # Attach trajectory-recording hook if enabled.
        # `trainer.on_iteration_end` is a virtual method; replacing the instance
        # attribute shadows the class method and gets called by trainer.train().
        if _traj_writer is not None:
            _writer = _traj_writer
            _env_cfg = _env_config_for_eval
            _tid = trial_id

            def _traj_on_iter(result: Any) -> None:
                import warnings
                try:
                    from theseo_anysearch.experiments.trajectory import collect_eval_episode as _coll
                    ep = _coll(trainer._algo, _env_cfg)
                    _writer.record(ep)
                    _writer.on_iteration_end(
                        result.iteration,
                        result.episode_reward_mean,
                        "tune",
                        _tid,
                    )
                except Exception as _exc:
                    warnings.warn(
                        f"Trajectory recording skipped at iter {result.iteration}: {_exc}"
                    )

            trainer.on_iteration_end = _traj_on_iter

        results = trainer.train()
        for result in results:
            tracker.log_metrics(
                {
                    "episode_reward_mean": result.episode_reward_mean,
                    "episode_len_mean": result.episode_len_mean,
                    "episodes_total": float(result.episodes_total),
                },
                step=result.iteration,
            )
            reported_metric = _sanitize_metric(result.episode_reward_mean, mode)
            _tune.report(
                {
                    metric: reported_metric,
                    "episode_reward_mean": reported_metric,
                    "episode_len_mean": result.episode_len_mean,
                    "episodes_total": result.episodes_total,
                    "training_iteration": result.iteration,
                }
            )
        tracker.end_child_run("FINISHED")
    except Exception:
        tracker.end_child_run("FAILED")
        raise


def _make_trainable(
    stl: Path,
    agents: int,
    max_steps: int,
    seed: int,
    max_iterations: int,
    output_dir: Path,
    metric: str,
    mode: str,
    num_gpus: int,
    base_config: dict[str, Any],
    env_base: dict[str, Any],
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_parent_run_id: str,
    run_tag: str,
    trajectory_every: int,
    best_trajectory: bool,
) -> Any:
    from ray import tune

    trainable = tune.with_parameters(
        _real_trainable,
        stl=str(stl.resolve()),            # absolute path — Ray workers may run from a different CWD
        agents=agents,
        max_steps=max_steps,
        seed=seed,
        max_iterations=max_iterations,
        output_dir=str(output_dir.resolve()),  # absolute path
        metric=metric,
        mode=mode,
        num_gpus=num_gpus,
        base_config=base_config,
        env_base=env_base,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_experiment_name=mlflow_experiment_name,
        mlflow_parent_run_id=mlflow_parent_run_id,
        run_tag=run_tag,
        trajectory_every=trajectory_every,
        best_trajectory=best_trajectory,
    )
    # Tell Ray to allocate GPU resources per trial so CUDA_VISIBLE_DEVICES is
    # set before RLlib tries to claim GPU IDs from the Ray resource pool.
    if num_gpus > 0:
        trainable = tune.with_resources(trainable, resources={"gpu": num_gpus})
    return trainable


def _trial_label(trial: Any) -> str:
    return f"trial_{trial.trial_id}"


def _ray_runtime_metadata(output_dir: Path) -> Path:
    return output_dir.joinpath("ray_runtime.json")


def _write_ray_runtime_metadata(output_dir: Path, tb_logdir: str | None = None) -> None:
    session_dir = None
    try:
        import ray

        node = getattr(ray._private.worker, "_global_node", None)
        if node is not None:
            session_dir = node.get_session_dir_path()
    except Exception:
        session_dir = None

    _ray_runtime_metadata(output_dir).write_text(
        json.dumps(
            {
                "ray_output_dir": str(output_dir.joinpath("ray")),
                "ray_session_dir": session_dir,
                "tensorboard_logdir": tb_logdir,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _close_child_runs(tracking_uri: str, parent_run_id: str, results: Any) -> None:
    """Mark all child runs of parent_run_id as FINISHED or FAILED from the driver.

    Ray kills worker processes after tune.report(), so workers cannot call
    end_child_run() themselves.  This must be called from the driver process
    after Tuner.fit() completes.
    """
    if not tracking_uri or not parent_run_id:
        return
    try:
        import mlflow
        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        experiment_id = client.get_run(parent_run_id).info.experiment_id
        child_runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        )
        errored_trial_ids: set[str] = set()
        if results is not None:
            for r in results:
                if r.error:
                    tid = (r.config or {}).get("trial_id", "")
                    if tid:
                        errored_trial_ids.add(tid)
        for run in child_runs:
            if run.info.status == "RUNNING":
                trial_id = run.data.tags.get("trial_id", "")
                status = "FAILED" if trial_id in errored_trial_ids else "FINISHED"
                client.set_terminated(run.info.run_id, status)
    except Exception as exc:
        import warnings
        warnings.warn(f"MLflow _close_child_runs failed: {exc}", stacklevel=2)


@app.callback(invoke_without_command=True)
def tune(
    stl: Optional[Path] = typer.Option(None, help="Path to the STL geometry file. Inferred from --config env.stl_path when omitted."),
    algorithm: str = typer.Option("ppo", help="Base RLlib algorithm (ppo, sac)."),
    scheduler: str = typer.Option(
        "asha",
        help=(
            "Tune trial scheduler / search algorithm. "
            "Phase 1: asha, pbt. "
            "Phase 2: random, hyperband, bohb (requires ConfigSpace). "
            "Phase 3: optuna (requires optuna), cmaes (requires nevergrad), "
            "flaml (requires flaml)."
        ),
    ),
    num_samples: int = typer.Option(20, help="Number of hyperparameter trials."),
    max_concurrent: int = typer.Option(2, help="Max trials running concurrently (limits RAM use)."),
    max_iterations: int = typer.Option(50, help="Max training iterations per trial."),
    metric: str = typer.Option("episode_reward_mean", help="Result key to optimise."),
    mode: str = typer.Option("max", help="Optimisation direction (max, min)."),
    output_dir: Path = typer.Option(Path("runtime", "tune"), help="Ray Tune experiment output directory."),
    agents: int = typer.Option(4, help="Number of concurrent agents."),
    max_steps: int = typer.Option(200, help="Max steps per episode."),
    seed: int = typer.Option(42, help="Global RNG seed."),
    mlflow_tracking_uri: str = typer.Option(
        "sqlite:///experiments/mlflow.db",
        help="MLflow tracking URI. Pass '' to disable MLflow tracking.",
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help=(
            "Path to a tune YAML config file. If provided, ``tune_config.search_space`` "
            "overrides the default search space and ``tune_config`` settings "
            "(scheduler, num_samples, max_concurrent, metric, mode) override CLI defaults. "
            "CLI flags still take precedence over YAML values."
        ),
    ),
    tag: str | None = typer.Option(
        None,
        "--tag",
        help=(
            "Experiment tag used as the Ray Tune run name and output subdirectory. "
            "Defaults to '{algorithm}_{scheduler}'. Same tag → same directory; "
            "use a different tag to run multiple searches in parallel."
        ),
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Delete an existing run with the same tag and start fresh.",
        is_flag=True,
    ),
) -> None:
    """Run a hyperparameter search experiment via Ray Tune."""
    typer.echo(
        "[deprecated] anysearch tune is deprecated. Use: anysearch run <dir> --tag <tag>\n",
        err=True,
    )
    try:
        import ray
        from ray import tune as ray_tune
    except ImportError:
        _print_error("ray not found", hint="pip install ray[tune]")
        raise typer.Exit(code=1)

    # Infer stl from YAML env.stl_path when not provided explicitly.
    if stl is None and config is not None:
        try:
            raw_cfg = _load_tune_config_yaml(config)
            stl_from_yaml = raw_cfg.get("env", {}).get("stl_path")
            if stl_from_yaml:
                stl = Path(stl_from_yaml)
        except Exception:
            pass
    if stl is None:
        _print_error("--stl is required (or set env.stl_path in --config)")
        raise typer.Exit(code=1)
    if not stl.exists():
        _print_error("stl file not found", path=str(stl))
        raise typer.Exit(code=1)

    algorithm = algorithm.lower()
    scheduler = scheduler.lower()
    mode = mode.lower()

    if mode not in {"max", "min"}:
        _print_error("invalid optimisation mode", mode=mode)
        raise typer.Exit(code=1)

    # Load YAML config (if provided) and apply tune_config overrides.
    # CLI flags always win; YAML values are applied only when the CLI option
    # was not explicitly set (i.e. still at its default value).
    base_config: dict[str, Any] = {}
    env_base: dict[str, Any] = {}
    num_gpus: int = 0
    if config is not None:
        if not config.exists():
            _print_error("tune config file not found", path=str(config))
            raise typer.Exit(code=1)
        try:
            raw_cfg = _load_tune_config_yaml(config)
        except Exception as exc:
            _print_error("failed to load tune config YAML", detail=str(exc))
            raise typer.Exit(code=1)
        tc = raw_cfg.get("tune_config", {})
        # Apply YAML defaults for options the user did not pass explicitly.
        if num_samples == 20 and "num_samples" in tc:
            num_samples = int(tc["num_samples"])
        if max_concurrent == 2 and "max_concurrent" in tc:
            max_concurrent = int(tc["max_concurrent"])
        if max_iterations == 50 and "max_iterations" in tc:
            max_iterations = int(tc["max_iterations"])
        if metric == "episode_reward_mean" and "metric" in tc:
            metric = str(tc["metric"])
        if mode == "max" and "mode" in tc:
            mode = str(tc["mode"])
        if scheduler == "asha" and "scheduler" in tc:
            scheduler = str(tc["scheduler"]).lower()

        # Fixed defaults: algorithm_config + model_config outside search_space.
        base_config = _flatten_yaml_base_config(raw_cfg)
        # env: section — passed verbatim to EnvConfig (scale, geometry, reward params, etc.)
        env_base = {k: v for k, v in raw_cfg.get("env", {}).items()
                    if k not in ("stl_path", "agent_count", "max_steps", "seed")}
        # GPU count from training.num_gpus in YAML; fall back to auto-detect.
        training_cfg = raw_cfg.get("training", {})
        if "num_gpus" in training_cfg:
            num_gpus = int(training_cfg["num_gpus"])
        else:
            import torch
            num_gpus = min(1, torch.cuda.device_count())
        trajectory_every: int = int(training_cfg.get("trajectory_every", 0))
        best_trajectory: bool = bool(training_cfg.get("best_trajectory", True))

        try:
            search_space = (
                _parse_search_space(tc["search_space"], ray_tune)
                if "search_space" in tc
                else _default_search_space(ray_tune, algorithm)
            )
        except (ValueError, AttributeError) as exc:
            _print_error("invalid search_space in tune config", detail=str(exc))
            raise typer.Exit(code=1)
    else:
        import torch
        num_gpus = min(1, torch.cuda.device_count())
        trajectory_every = 0
        best_trajectory = True
        try:
            search_space = _default_search_space(ray_tune, algorithm)
        except NotImplementedError as exc:
            _print_error(str(exc))
            raise typer.Exit(code=1)

    # Pre-flight config validation — print warnings but do not abort.
    config_issues = _validate_tune_config(base_config, search_space, num_gpus)
    if config_issues:
        sep = "-" * 60
        print(f"\n{sep}\n  Config warnings\n{sep}", flush=True)
        for issue in config_issues:
            print(f"  [!] {issue}", flush=True)
        print(sep, flush=True)

    try:
        scheduler_obj = _build_scheduler(
            scheduler_name=scheduler,
            max_iterations=max_iterations,
            hyperparam_mutations=search_space,
        )
    except ValueError as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1)

    try:
        search_alg_obj = _build_search_alg(
            scheduler_name=scheduler,
            metric=metric,
            mode=mode,
        )
    except ValueError as exc:
        _print_error(str(exc))
        raise typer.Exit(code=1)

    run_tag = tag or f"{algorithm}_{scheduler}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Use os.path.normpath to ensure native OS separators (avoids mixed-separator
    # path errors on Windows when Ray Tune internally joins paths with os.path.join).
    storage_path = os.path.normpath(str(output_dir.resolve()))

    run_dir = Path(storage_path) / run_tag
    if run_dir.exists():
        if overwrite:
            shutil.rmtree(run_dir)
        else:
            _print_existing_run(run_dir, output_dir)
            return

    # Ray appends ~188 chars of session/artifact subdirs + filename to _temp_dir.
    # Windows MAX_PATH is 260, so use the system temp dir to keep paths short.
    # The output_dir alone is ~90 chars, which leaves no room for Ray's additions.
    import tempfile
    ray_temp_dir = os.path.join(
        tempfile.gettempdir(), f"anysearch_tune_{os.getpid()}"
    )
    os.makedirs(ray_temp_dir, exist_ok=True)

    # Suppress Ray's FutureWarning about accelerator env var override on zero GPUs
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        _temp_dir=ray_temp_dir,
    )

    # TensorBoard logdir: Ray Tune writes one subdir per trial under storage_path/run_tag
    tb_logdir = os.path.join(storage_path, run_tag)
    _write_ray_runtime_metadata(output_dir, tb_logdir=tb_logdir)

    experiment_name = f"tune/{run_tag}"
    from theseo_anysearch.experiments.models import MLflowConfig
    from theseo_anysearch.experiments.tracking import MLflowTracker
    # Resolve sqlite:///relative/path → absolute so Ray worker processes (which
    # run in a different CWD) connect to the same database file.
    if mlflow_tracking_uri and mlflow_tracking_uri.startswith("sqlite:///"):
        rel = mlflow_tracking_uri[len("sqlite:///"):]
        if not Path(rel).is_absolute():
            mlflow_tracking_uri = "sqlite:///" + str(Path(rel).resolve())
    mlflow_cfg = MLflowConfig(tracking_uri=mlflow_tracking_uri) if mlflow_tracking_uri else None
    tracker = MLflowTracker(mlflow_cfg, experiment_name)
    parent_run_id = tracker.start_run(
        run_name=run_tag,
        tags={"algorithm": algorithm, "scheduler": scheduler, "num_samples": str(num_samples), "run_tag": run_tag},
    ) or ""
    tracker.log_params({
        "algorithm": algorithm, "scheduler": scheduler,
        "num_samples": num_samples, "max_concurrent": max_concurrent,
        "max_iterations": max_iterations, "metric": metric, "mode": mode,
        "agents": agents, "max_steps": max_steps, "seed": seed,
        "stl": str(stl),
    })

    _print_run_overview(
        run_tag=run_tag,
        num_samples=num_samples,
        max_concurrent=max_concurrent,
        max_iterations=max_iterations,
        metric=metric,
        mode=mode,
        output_dir=output_dir,
        tb_logdir=tb_logdir,
        mlflow_tracking_uri=mlflow_tracking_uri,
        parent_run_id=parent_run_id,
    )

    # On Windows, Ray's storage layer uses `.as_posix()` which produces forward-slash
    # paths that mix with os.path.join backslashes, causing FileNotFoundError.
    # apply_win_atomic_save_patch() normalises separators and emits an actionable
    # error message (pointing to spec/bugs.md) if the path still exceeds MAX_PATH.
    from theseo_anysearch.experiments.tune_runner import apply_win_atomic_save_patch
    apply_win_atomic_save_patch()

    # Attach TensorBoard callback if tensorboard/tensorboardX is available
    tb_callbacks = []
    try:
        from ray.tune.logger import TBXLoggerCallback
        tb_callbacks = [TBXLoggerCallback()]
    except ImportError:
        pass  # tensorboard not installed — skip silently

    try:
        tuner = ray_tune.Tuner(
            _make_trainable(
                stl=stl,
                agents=agents,
                max_steps=max_steps,
                seed=seed,
                max_iterations=max_iterations,
                output_dir=output_dir,
                metric=metric,
                mode=mode,
                num_gpus=num_gpus,
                base_config=base_config,
                env_base=env_base,
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_experiment_name=experiment_name,
                mlflow_parent_run_id=parent_run_id,
                run_tag=run_tag,
                trajectory_every=trajectory_every,
                best_trajectory=best_trajectory,
            ),
            param_space=search_space,
            tune_config=ray_tune.TuneConfig(
                metric=metric,
                mode=mode,
                num_samples=num_samples,
                max_concurrent_trials=max_concurrent,
                scheduler=scheduler_obj,
                search_alg=search_alg_obj,
                trial_name_creator=_trial_label,
                trial_dirname_creator=_trial_label,
                reuse_actors=False,  # True causes stale PG bundle refs after ~14 trials
            ),
            run_config=ray_tune.RunConfig(
                name=run_tag,
                storage_path=storage_path,
                verbose=1,
                callbacks=tb_callbacks,
            ),
        )
        results = tuner.fit()
        best_result = results.get_best_result(metric=metric, mode=mode)
        tracker.log_params({"best_" + k: str(v) for k, v in (best_result.config or {}).items()})
        # Terminate all child runs from the driver process — worker processes are
        # killed by Ray Tune after tune.report(), so end_child_run() in the worker
        # never executes.  The driver is the only safe place to close child runs.
        _close_child_runs(mlflow_tracking_uri, parent_run_id, results)
        tracker.end_run("FINISHED")
    except Exception as exc:
        _close_child_runs(mlflow_tracking_uri, parent_run_id, results if "results" in dir() else None)
        tracker.end_run("FAILED")
        _print_error("tune run failed", detail=str(exc))
        raise typer.Exit(code=1)
    finally:
        ray.shutdown()

    payload = {
        "best_config": best_result.config,
        "best_result": {
            metric: best_result.metrics.get(metric),
            "training_iteration": best_result.metrics.get("training_iteration"),
            "episode_reward_mean": best_result.metrics.get("episode_reward_mean"),
        },
        "experiment_dir": str(output_dir.resolve()),
        "tensorboard_logdir": tb_logdir,
    }
    print(json.dumps(payload, indent=2, default=str))
    print(f"\nTensorBoard: tensorboard --logdir \"{tb_logdir}\"", flush=True)
