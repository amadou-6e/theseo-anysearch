"""Ray Tune orchestration for experiment-level hyperparameter sweeps."""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from theseo_anysearch.experiments.models import ExperimentConfig

_tune: Any | None = None
Trainer: Any | None = None


def _tune_stop_criteria(
    tune_config: Any,
    *,
    max_iterations: int,
) -> dict[str, float | int]:
    """Build Tune stop criteria from the explicit trial budget."""
    stop: dict[str, float | int] = {
        "training_iteration": max_iterations,
    }
    if tune_config.max_environment_steps is not None:
        stop["environment_steps_total"] = tune_config.max_environment_steps
    if tune_config.max_wall_time_s is not None:
        stop["time_total_s"] = tune_config.max_wall_time_s
    if tune_config.target_success_rate is not None:
        stop["evaluation_success_rate"] = tune_config.target_success_rate
    return stop


def _selected_metric(
    payload: dict[str, Any],
    metric: str,
    *,
    fallback: float,
    mode: str,
) -> float:
    """Resolve and sanitize the actual metric configured for Tune."""
    value = float(payload.get(metric, fallback))
    if math.isfinite(value):
        return value
    return -1e9 if mode == "max" else 1e9


def _apply_sampled_model_config(model: Any, config: dict[str, Any]) -> Any:
    """Apply sampled architecture dimensions as one coherent model shape."""
    model_updates: dict[str, Any] = {}
    if "base_width" in config:
        base_width = int(config["base_width"])
        head_width = int(config.get("head_width", base_width))
        layers_per_width = int(config.get("layers_per_width", 1))
        if base_width < 1 or head_width < 1 or layers_per_width < 1:
            raise ValueError(
                "sampled model widths and repetitions must be positive")

        widths: list[int] = []
        width = base_width
        while width != head_width:
            widths.extend([width] * layers_per_width)
            if width > head_width:
                width = max(width // 2, head_width)
            else:
                width = min(width * 2, head_width)
        widths.extend([head_width] * layers_per_width)
        model_updates["hidden_sizes"] = widths
        if "encoder_depth" in type(model).model_fields:
            model_updates["encoder_depth"] = len(widths)
        return model.model_copy(update=model_updates)

    sampled_depth = config.get("encoder_depth", config.get("num_layers"))
    hidden_sizes = list(getattr(model, "hidden_sizes", []))
    depth = int(sampled_depth) if sampled_depth is not None else len(
        hidden_sizes)

    if sampled_depth is not None and "encoder_depth" in type(
            model).model_fields:
        model_updates["encoder_depth"] = max(depth, 1)
    if "layer_size" in config:
        model_updates["hidden_sizes"] = [int(config["layer_size"])] * max(
            depth, 1)
    elif sampled_depth is not None:
        width = hidden_sizes[-1] if hidden_sizes else 256
        model_updates["hidden_sizes"] = [int(width)] * max(depth, 1)
    if ("use_position_encoding" in config
            and "use_position_encoding" in type(model).model_fields):
        model_updates["use_position_encoding"] = bool(
            config["use_position_encoding"])

    return model.model_copy(update=model_updates)


def _trial_resource_metrics(trainer: Any, settings: Any) -> dict[str, float]:
    """Return comparable per-trial compute and architecture metadata."""
    algorithm = settings.algorithm_config
    model = settings.model_cfg
    hidden_sizes = list(getattr(model, "hidden_sizes", []))
    parameter_count = 0
    observation_size = 0
    try:
        policy = trainer._algo.get_policy()
        policy_model = getattr(policy, "model", None)
        if policy_model is not None:
            parameter_count = sum(
                int(parameter.numel())
                for parameter in policy_model.parameters())
        from gymnasium.spaces.utils import flatdim

        observation_size = int(flatdim(policy.observation_space))
    except (AttributeError, NotImplementedError, TypeError, ValueError):
        pass

    return {
        "resource/train_batch_size":
        float(getattr(algorithm, "train_batch_size", 0)),
        "resource/num_sgd_iter":
        float(getattr(algorithm, "num_sgd_iter", 0)),
        "resource/model_parameter_count":
        float(parameter_count),
        "resource/observation_size":
        float(observation_size),
        "resource/hidden_layer_count":
        float(len(hidden_sizes)),
        "resource/hidden_layer_width":
        float(max(hidden_sizes, default=0)),
        "resource/num_env_runners":
        float(settings.training.num_env_runners),
        "resource/num_envs_per_env_runner":
        float(getattr(settings.training, "num_envs_per_env_runner", 1)),
        "resource/num_gpus_per_env_runner":
        float(getattr(settings.training, "num_gpus_per_env_runner", 0.0)),
        "resource/evaluation_num_env_runners":
        float(
            getattr(getattr(settings, "evaluation", None), "num_env_runners",
                    0)),
        "resource/evaluation_num_envs_per_env_runner":
        float(
            getattr(
                getattr(settings, "evaluation", None),
                "num_envs_per_env_runner",
                1,
            )),
        "resource/num_gpus":
        float(settings.training.num_gpus or 0.0),
    }


def _append_tune_history(
    trial_dir: Path,
    payload: dict[str, Any],
) -> None:
    """Persist the same per-iteration payload reported to Tune."""
    history_path = trial_dir.joinpath("tune_history.jsonl")
    with history_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def _write_tune_status(
    trial_dir: Path,
    status: str,
    *,
    iteration: int,
) -> None:
    trial_dir.joinpath("tune_status.json").write_text(
        json.dumps(
            {
                "status": status,
                "training_iteration": iteration
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _persist_trial_outcomes(
    sweep_dir: Path,
    results: Any,
    *,
    max_iterations: int,
) -> None:
    """Classify completed, pruned, and failed trials in project artifacts."""
    if results is None:
        return
    for result in results:
        metrics = result.metrics or {}
        trial_id = metrics.get("trial_id")
        if not trial_id:
            continue
        iteration = int(metrics.get("training_iteration", 0))
        if result.error is not None:
            status = "FAILED"
        elif iteration < max_iterations:
            status = "PRUNED"
        else:
            status = "COMPLETED"
        trial_dir = sweep_dir.joinpath(str(trial_id))
        if trial_dir.is_dir():
            _write_tune_status(trial_dir, status, iteration=iteration)


def _restore_reported_checkpoint(trainer: Any, tune_api: Any) -> bool:
    """Restore a project trainer from Ray's incoming function checkpoint."""
    try:
        checkpoint = tune_api.get_checkpoint()
    except RuntimeError:
        return False
    if checkpoint is None:
        return False
    if not checkpoint.__class__.__module__.startswith("ray."):
        return False
    with checkpoint.as_directory() as checkpoint_dir:
        trainer.restore(Path(checkpoint_dir))
    return True


def _ray_checkpoint(checkpoint_dir: Any) -> Any | None:
    """Convert a real project checkpoint directory into a Ray checkpoint."""
    if not isinstance(checkpoint_dir, Path) or not checkpoint_dir.is_dir():
        return None
    from ray.train import Checkpoint

    return Checkpoint.from_directory(str(checkpoint_dir))


# ---------------------------------------------------------------------------
# Module-level trainable (must be picklable by cloudpickle / Ray)
# ---------------------------------------------------------------------------


def _experiment_trainable(
    config: dict[str, Any],
    *,
    experiment_dict: dict[str, Any],
    metric: str,
    mode: str,
    max_iterations: int,
    mlflow_tracking_uri: str,
    mlflow_experiment_name: str,
    mlflow_parent_run_id: str,
    run_tag: str,
    trial_prefix: str = "",
    checkpoint_frequency: int = 1,
    preserve_trial_artifacts: bool = True,
    metric_source_contents: dict[str, str] | None = None,
    reward_source_content: str | None = None,
    native_extension_bundle: dict[str, Any] | None = None,
) -> None:
    """
    Generic Ray Tune function trainable for any algorithm in the built-in algorithm registry.

    Parameters
    ----------
    config:
        Sampled hyperparameter values from the search space (populated by Ray
        Tune before each trial call).
    experiment_dict:
        ExperimentConfig serialised to a plain dict (by_alias=True, mode='json').
        Output paths have been resolved to absolute by TuneRunner.run() before
        serialisation so that workers running in a different CWD resolve
        correctly.
    """
    global _tune, Trainer
    if _tune is None:
        from ray import tune as ray_tune

        _tune = ray_tune
    if Trainer is None:
        from theseo_anysearch.rllib.trainer import Trainer as TrainerClass

        Trainer = TrainerClass
    from theseo_anysearch.experiments.loader import _resolve_typed_configs
    from theseo_anysearch.experiments.models import ExperimentConfig, MLflowConfig
    from theseo_anysearch.experiments.tracking import MLflowTracker
    from theseo_anysearch.rllib.trainer.results import TrainResult

    # ------------------------------------------------------------------ #
    # Trial identity                                                        #
    # ------------------------------------------------------------------ #
    try:
        context = _tune.get_context()
        trial_id = context.get_trial_id() or "trial"
    except Exception:
        trial_id = "trial"
    output_trial_id = f"{trial_prefix}{trial_id}"

    # ------------------------------------------------------------------ #
    # Reconstruct ExperimentConfig on the worker                           #
    # ------------------------------------------------------------------ #
    resolved = _resolve_typed_configs(experiment_dict)
    exp_config = ExperimentConfig(**resolved)

    # ------------------------------------------------------------------ #
    # Apply sampled algo hyper-params to algorithm_config                  #
    #                                                                      #
    # Filter to fields that actually exist on the concrete config class    #
    # (PPOConfig, SACConfig, DQNConfig, …).  A hardcoded PPO-field list   #
    # would silently drop SAC/DQN-specific search params (tau, n_step …)  #
    # and would raise a Pydantic validation error if PPO-only fields were  #
    # passed to a non-PPO config via model_copy().                         #
    # ------------------------------------------------------------------ #
    valid_algo_fields = set(
        type(exp_config.algorithm_config).model_fields.keys())
    algo_updates = {k: v for k, v in config.items() if k in valid_algo_fields}
    new_algo = exp_config.algorithm_config.model_copy(update=algo_updates)

    # ------------------------------------------------------------------ #
    # Apply sampled model hyper-params to model_cfg                        #
    # ------------------------------------------------------------------ #
    new_model = _apply_sampled_model_config(exp_config.model_cfg, config)

    # ------------------------------------------------------------------ #
    # Trial output directory                                               #
    # ------------------------------------------------------------------ #
    # experiment.output_dir was resolved to absolute by TuneRunner before  #
    # serialisation, so this path is safe to use from any worker CWD.      #
    trial_dir = Path(
        experiment_dict["experiment"]["output_dir"],
        experiment_dict["experiment"]["name"],
        run_tag,
        output_trial_id,
    )
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Write a true YAML snapshot so inspect/repeat works on trial directories.
    import yaml

    trial_dir.joinpath("experiment.yaml").write_text(
        yaml.safe_dump(experiment_dict, sort_keys=False),
        encoding="utf-8",
    )
    trial_dir.joinpath("sampled_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    # Archive the exact extension sources used by this trial.
    for filename, source in (metric_source_contents or {}).items():
        trial_dir.joinpath(filename).write_text(source, encoding="utf-8")
    if reward_source_content is not None:
        trial_dir.joinpath("rewards.py").write_text(
            reward_source_content,
            encoding="utf-8",
        )
    if native_extension_bundle is not None:
        native_dir = trial_dir.joinpath("native_extension")
        native_dir.mkdir(parents=True, exist_ok=True)
        native_dir.joinpath(native_extension_bundle["filename"]).write_bytes(
            native_extension_bundle["binary"]
        )
        native_dir.joinpath("extension.json").write_text(
            native_extension_bundle["manifest"], encoding="utf-8"
        )

    # ------------------------------------------------------------------ #
    # Build Settings for this trial                                        #
    # ------------------------------------------------------------------ #
    # Derive a stable per-trial seed offset so different trials explore
    # different start/goal position sequences during both training rollouts
    # and eval trajectory snapshots. Using a hash of trial_id makes the
    # offset deterministic (reproducible on resume) and unique per trial.
    import hashlib as _hashlib
    _seed_offset = int(
        _hashlib.md5(output_trial_id.encode()).hexdigest()[:8], 16) % 10000

    base_settings = exp_config.to_settings()
    trial_env = base_settings.env.model_copy(
        update={
            "seed": base_settings.env.seed + _seed_offset,
        })
    settings = base_settings.model_copy(
        update={
            "training":
            base_settings.training.model_copy(
                update={
                    "output_dir":
                    trial_dir,
                    "iterations":
                    max_iterations,
                    "checkpoint_interval":
                    max_iterations + 1,
                    "trajectory_every": (
                        1 if preserve_trial_artifacts else base_settings.
                        training.trajectory_every),
                    "best_trajectory": (
                        True if preserve_trial_artifacts else base_settings.
                        training.best_trajectory),
                }),
            "algorithm_config":
            new_algo,
            "model_cfg":
            new_model,
            "env":
            trial_env,
        })

    # ------------------------------------------------------------------ #
    # MLflow child run                                                     #
    # ------------------------------------------------------------------ #
    mlflow_cfg = MLflowConfig(
        tracking_uri=mlflow_tracking_uri) if mlflow_tracking_uri else None
    tracker = MLflowTracker(mlflow_cfg, mlflow_experiment_name)
    tracker.start_child_run(
        run_name=output_trial_id,
        tags={
            "trial_id": output_trial_id,
            "ray_trial_id": trial_id
        },
        parent_run_id=mlflow_parent_run_id or None,
    )
    tracker.log_params({k: str(v) for k, v in config.items()})

    # ------------------------------------------------------------------ #
    # Train                                                                #
    # ------------------------------------------------------------------ #
    trainer = Trainer.from_settings(settings)
    _restore_reported_checkpoint(trainer, _tune)
    resource_metrics: dict[str, float] | None = None
    last_iteration = 0

    def _iteration_hook(result: TrainResult) -> None:
        nonlocal resource_metrics, last_iteration
        last_iteration = result.iteration
        if resource_metrics is None:
            resource_metrics = _trial_resource_metrics(trainer, settings)
            trial_dir.joinpath("resources.json").write_text(
                json.dumps(resource_metrics, indent=2),
                encoding="utf-8",
            )

        payload: dict[str, Any] = {
            **result.standard_metrics(),
            **resource_metrics,
            "training_iteration": result.iteration,
            "trial_id": output_trial_id,
            "evaluation_status": result.evaluation_status,
        }
        sanitized = _selected_metric(
            payload,
            metric,
            fallback=result.episode_reward_mean,
            mode=mode,
        )
        payload[metric] = sanitized
        if metric == "episode_reward_mean":
            payload["episode_reward_mean"] = sanitized

        tracker.log_metrics(
            {
                key: value
                for key, value in payload.items()
                if isinstance(value, (int, float))
            },
            step=result.iteration,
        )
        _append_tune_history(trial_dir, payload)
        _write_tune_status(
            trial_dir,
            "RUNNING",
            iteration=result.iteration,
        )

        report_options: dict[str, Any] = {}
        if result.iteration % checkpoint_frequency == 0:
            tune_checkpoint = _ray_checkpoint(trainer.checkpoint())
            if tune_checkpoint is not None:
                report_options["checkpoint"] = tune_checkpoint
        _tune.report(payload, **report_options)

    trainer.on_iteration_end = _iteration_hook  # type: ignore[method-assign]

    try:
        trainer.train()
        _write_tune_status(trial_dir, "COMPLETED", iteration=last_iteration)
        tracker.end_child_run("FINISHED")
    except Exception:
        _write_tune_status(trial_dir, "FAILED", iteration=last_iteration)
        tracker.end_child_run("FAILED")
        raise
    finally:
        # Explicitly stop the RLlib algorithm so its env-runner actors are
        # released before the next trial reuses this worker process.
        if getattr(trainer, "_algo", None) is not None:
            try:
                trainer._algo.stop()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# TuneRunner
# ---------------------------------------------------------------------------


def apply_win_atomic_save_patch() -> None:
    """
    Patch Ray's internal _atomic_save to normalise mixed path separators on
    Windows and emit an actionable error if the path still exceeds MAX_PATH.

    Ray constructs checkpoint paths by joining a storage_path (which may use
    forward slashes from YAML) with internal suffixes using os.path.join
    (backslashes on Windows).  The result can be a hybrid like
    ``C:/some/path/driver_artifacts/.tmp-foo.json`` which open() rejects on
    Windows.  Applying os.path.normpath to the checkpoint_dir argument of
    _atomic_save converts all separators to backslashes before the file is
    created.

    If the normalised path still causes a FileNotFoundError (path > 260 chars),
    the error is re-raised with a clear message pointing to the ray_storage_dir
    and ray_temp_dir config options.  See spec/bugs.md for full details.

    This function is idempotent and no-ops on non-Windows platforms.  It is
    extracted so that tests that call rt.Tuner() directly (without TuneRunner)
    can apply the same fix, ensuring tests reflect what production does.
    """
    if os.name != "nt":
        return
    try:
        import ray.tune.utils.util as _tune_util
        _orig_save = _tune_util._atomic_save

        def _win_atomic_save(state, checkpoint_dir, file_name, tmp_file_name):
            norm_dir = os.path.normpath(checkpoint_dir)
            try:
                return _orig_save(state, norm_dir, file_name, tmp_file_name)
            except FileNotFoundError as exc:
                path_len = len(os.path.join(norm_dir,
                                            f".uuid-{tmp_file_name}"))
                raise FileNotFoundError(
                    f"\n\nRay could not write its internal state file — path too long "
                    f"for Windows MAX_PATH (260 chars).\n\n"
                    f"  Directory : {norm_dir}\n"
                    f"  Approx len: {path_len} chars\n\n"
                    f"Fix: add ray_storage_dir and ray_temp_dir to your tune_config\n"
                    f"pointing to a short absolute path (e.g. C:/ray_tmp):\n\n"
                    f"  tune_config:\n"
                    f"    ray_storage_dir: C:/ray_tmp\n"
                    f"    ray_temp_dir: C:/ray_tmp\n\n"
                    f"See spec/bugs.md for full details.") from exc

        _tune_util._atomic_save = _win_atomic_save
        try:
            import ray.tune.search.basic_variant as _bv
            _bv._atomic_save = _win_atomic_save
        except Exception:
            pass
    except Exception:
        pass


def _trial_dirname(trial: Any) -> str:
    return f"trial_{trial.trial_id}"


def _tune_trial_num_gpus(
    require_gpu: bool,
    max_concurrent: int,
    configured_num_gpus: float | None = None,
) -> float:
    """Return the explicit RLlib GPU allocation to embed in tune trial settings.

    Parameters
    ----------
    require_gpu : bool
        Whether the sweep requires CUDA-backed training.
    max_concurrent : int
        Maximum number of concurrent Ray Tune trials.
    configured_num_gpus : float or None
        Explicit GPU allocation per trial. When omitted, one GPU is split
        evenly across concurrent trials.

    Returns
    -------
    float
        Explicit GPU allocation per trial. CPU-only sweeps return ``0.0`` so
        RLlib does not fall back to local GPU auto-detection.
    """
    if not require_gpu:
        return 0.0
    if configured_num_gpus is not None:
        return configured_num_gpus
    if max_concurrent > 1:
        return 1.0 / max_concurrent
    return 1.0


def _sweep_runtime_path(sweep_dir: Path) -> Path:
    """Return the runtime metadata file for a sweep directory."""
    return sweep_dir.joinpath("tune_runtime.json")


def _load_sweep_runtime(sweep_dir: Path) -> dict[str, Any]:
    """Return persisted sweep runtime metadata, or an empty mapping."""
    runtime_path = _sweep_runtime_path(sweep_dir)
    if not runtime_path.exists():
        return {}
    try:
        return json.loads(runtime_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_sweep_runtime(sweep_dir: Path, payload: dict[str, Any]) -> None:
    """Persist sweep runtime metadata for resume and continuation workflows."""
    runtime_path = _sweep_runtime_path(sweep_dir)
    runtime_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _existing_trial_dirs(sweep_dir: Path) -> list[Path]:
    """Return trial output directories already present under a sweep tag."""
    if not sweep_dir.exists():
        return []
    return sorted(
        path for path in sweep_dir.iterdir()
        if path.is_dir() and path.joinpath("ray_runtime.json").exists())


def _segment_name(index: int) -> str:
    """Return a stable continuation segment name for a sweep."""
    if index <= 0:
        return "base"
    return f"cont_{index:03d}"


def _segment_trial_prefix(name: str) -> str:
    """Return the per-trial output directory prefix for a sweep segment."""
    if name == "base":
        return ""
    return f"{name}_"


def _segment_tune_run_name(run_tag: str, segment_name: str) -> str:
    """Return the Ray-internal run name used for a sweep segment."""
    if segment_name == "base":
        base_name = run_tag
    else:
        base_name = f"{run_tag}_{segment_name}"
    return base_name[:24] if len(base_name) > 24 else base_name


def _next_segment_index(runtime_meta: dict[str, Any]) -> int:
    """Return the next continuation index for a sweep runtime manifest."""
    segments = runtime_meta.get("segments", [])
    if not isinstance(segments, list):
        return 1
    cont_indices: list[int] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        name = str(segment.get("segment_name", ""))
        if not name.startswith("cont_"):
            continue
        try:
            cont_indices.append(int(name.split("_")[-1]))
        except ValueError:
            continue
    return (max(cont_indices) + 1) if cont_indices else 1


def _lookup_sweep_registered_name(output_dir: Path) -> str | None:
    """Return the registry name whose directory contains output_dir, or None."""
    try:
        from theseo_anysearch.cli.registry import load_registry
        reg = load_registry()
        resolved = output_dir.resolve()
        for name, dir_str in reg.items():
            try:
                resolved.relative_to(Path(dir_str).resolve())
                return name
            except ValueError:
                pass
    except Exception:
        pass
    return None


def _print_sweep_replay_hint(sweep_dir: Path, run_tag: str,
                             best_trial_id: str | None) -> None:
    sep = "-" * 62
    reg_name = _lookup_sweep_registered_name(sweep_dir.parent.parent)
    sweep_ref = f"{reg_name}:{run_tag}" if reg_name else str(sweep_dir)
    lines = [
        "",
        sep,
        f"  Sweep complete : {run_tag}",
        f"  Browse all trials (T/Y keys):  anysearch replay {sweep_ref}",
    ]
    if best_trial_id:
        best_ref = f"{reg_name}:{best_trial_id}" if reg_name else best_trial_id
        lines += [
            f"  Best trial     : {best_trial_id}",
            f"    Best episode  :  anysearch replay {best_ref} --best",
            f"    All snapshots :  anysearch replay {best_ref}",
        ]
    lines += [sep, ""]
    print("\n".join(lines), flush=True)


def _print_sweep_overview(
    *,
    config_path: Path | None,
    run_tag: str,
    algorithm: str,
    scheduler: str,
    num_samples: int,
    max_concurrent: int,
    iterations: int,
    num_env_runners: int,
    require_gpu: bool,
    search_space: dict[str, Any],
    sweep_dir: Path,
    mlflow_tracking_uri: str,
) -> None:
    import sys
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    tb_logdir = str(sweep_dir)

    def _kv_table() -> Table:
        t = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
        t.add_column(style="dim", min_width=14)
        t.add_column()
        return t

    # Main metadata
    meta = _kv_table()
    if config_path:
        meta.add_row("Source", str(config_path))
    meta.add_row("Algorithm", algorithm)
    meta.add_row("Scheduler", scheduler)
    meta.add_row("Trials", f"{num_samples} total, {max_concurrent} concurrent")
    meta.add_row("Iterations", f"up to {iterations} per trial")
    meta.add_row("Env runners", f"{num_env_runners} rollout workers per trial")
    meta.add_row("GPU", "yes" if require_gpu else "no")

    # Search space
    space_table = _kv_table()
    for param, sampler in sorted(search_space.items()):
        try:
            from ray.tune.search.sample import Categorical, Float, Integer, LogUniform
            if isinstance(sampler, Categorical):
                desc = f"choice{list(sampler.categories)}"
            elif isinstance(sampler, (Float, Integer)):
                lo, hi = sampler.lower, sampler.upper
                kind = "log_uniform" if isinstance(sampler,
                                                   LogUniform) else "uniform"
                desc = f"{kind}({lo}, {hi})"
            else:
                desc = repr(sampler)
        except Exception:
            desc = repr(sampler)
        space_table.add_row(param, desc)

    # Output / monitoring
    out_table = _kv_table()
    out_table.add_row("Trials", str(sweep_dir))
    if mlflow_tracking_uri:
        out_table.add_row("MLflow", mlflow_tracking_uri)
    out_table.add_row("TensorBoard", f'tensorboard --logdir "{tb_logdir}"')

    content = Group(
        meta,
        Text(""),
        Text("Search space", style="dim"),
        space_table,
        Text(""),
        Text("Output", style="dim"),
        out_table,
    )

    console = Console(file=sys.stdout, highlight=False)
    try:
        console.print(
            Panel(content,
                  title=f"[bold]Sweep[/bold]  [dim]{run_tag}[/dim]",
                  title_align="left"),
            new_line_start=True,
        )
    except UnicodeEncodeError:
        sep = "-" * 62
        lines = [
            "",
            sep,
            f"  Sweep : {run_tag}",
            sep,
            f"  Source        : {config_path or '(no file)'}",
            f"  Algorithm     : {algorithm}",
            f"  Scheduler     : {scheduler}",
            f"  Trials        : {num_samples} total, {max_concurrent} concurrent",
            f"  Iterations    : up to {iterations} per trial",
            f"  Env runners   : {num_env_runners} rollout workers per trial",
            f"  GPU           : {'yes' if require_gpu else 'no'}",
            "",
            "  Output",
            f"    Trials      : {sweep_dir}",
        ]
        if mlflow_tracking_uri:
            lines.append(f"    MLflow      : {mlflow_tracking_uri}")
        lines += [
            "", "  TensorBoard", f'    tensorboard --logdir "{tb_logdir}"',
            sep, ""
        ]
        print("\n".join(lines), flush=True)


def _close_child_runs(tracking_uri: str, parent_run_id: str,
                      results: Any) -> None:
    """Terminate open child MLflow runs from the driver after tuner.fit()."""
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
        for run in child_runs:
            if run.info.status == "RUNNING":
                client.set_terminated(run.info.run_id, "FINISHED")
    except Exception:
        import warnings
        warnings.warn(f"MLflow child-run cleanup failed: {tracking_uri!r}",
                      stacklevel=2)


def _first_trial_error(results: Any) -> str | None:
    """Return the first surfaced trial error string from a Tune result grid."""
    if results is None:
        return None
    try:
        for result in results:
            error = getattr(result, "error", None)
            if error:
                return str(error)
            path = getattr(result, "error_file", None)
            if path:
                return f"See trial error log: {path}"
    except Exception:
        return None
    return None


def _ray_runtime_options(address: str | None, temp_dir: str) -> dict[str, Any]:
    """Build Ray initialization options without mixing local and external runtimes."""
    options: dict[str, Any] = {
        "ignore_reinit_error": True,
        "include_dashboard": False,
    }
    if address:
        options["address"] = address
    else:
        options["_temp_dir"] = temp_dir
    return options


def _should_shutdown_ray(
    configured: bool | None,
    *,
    was_initialized: bool,
    address: str | None,
) -> bool:
    """Return whether this run owns the Ray runtime and should stop it."""
    if configured is not None:
        return configured
    return not was_initialized and not address

class TuneRunner:
    """
    Drives a Ray Tune hyperparameter sweep for an ExperimentConfig that
    carries a non-None tune_config.

    Supports any algorithm registered in the built-in algorithm registry
    (ppo, multi_agent_voxel_ppo, sac, …).  The search space is expressed
    using the same YAML format as ``anysearch tune --config``.

    Usage::

        from theseo_anysearch.experiments.loader import load_experiment
        from theseo_anysearch.experiments.tune_runner import TuneRunner

        exp = load_experiment(Path("multi_agent_ppo_asha.yaml"))
        runner = TuneRunner(exp)
        result = runner.run()     # blocks until all trials finish
        print(result)
    """

    def __init__(
        self,
        config: ExperimentConfig,
        config_path: Path | None = None,
        tag: str | None = None,
        resume: bool = False,
        extra_trials: int = 0,
    ) -> None:
        self._config = config
        self._config_path = config_path
        self._tag = tag
        self._resume = resume
        self._extra_trials = extra_trials

    def run(self) -> dict[str, Any]:
        import ray
        from ray import tune as ray_tune

        from theseo_anysearch.cli.commands.tune import (
            _build_scheduler,
            _build_search_alg,
            _parse_search_space,
        )
        from theseo_anysearch.experiments.models import MLflowConfig
        from theseo_anysearch.experiments.tracking import MLflowTracker

        tc = self._config.tune_config
        if tc is None:
            raise ValueError(
                "TuneRunner requires tune_config to be set in ExperimentConfig."
            )
        if self._resume and self._extra_trials > 0:
            raise ValueError(
                "resume and extra_trials cannot be used together for the same sweep launch."
            )
        if self._extra_trials < 0:
            raise ValueError("extra_trials must be non-negative.")

        # ------------------------------------------------------------------ #
        # Build Ray Tune objects from tune_config                             #
        # ------------------------------------------------------------------ #
        search_space = (_parse_search_space(tc.search_space, ray_tune)
                        if tc.search_space else {})
        scheduler_obj = _build_scheduler(
            scheduler_name=tc.scheduler,
            max_iterations=self._config.training.iterations,
            hyperparam_mutations=search_space,
            asha_config=tc.asha_config,
        )
        search_alg_obj = _build_search_alg(tc.scheduler, tc.metric, tc.mode)

        # ------------------------------------------------------------------ #
        # GPU fraction: split 1 GPU evenly across concurrent trials           #
        # e.g. max_concurrent=2 → 0.5 GPU per trial so both can run at once  #
        # ------------------------------------------------------------------ #
        require_gpu = self._config.training.require_gpu
        gpu_fraction = _tune_trial_num_gpus(
            require_gpu,
            tc.max_concurrent,
            self._config.training.num_gpus,
        )

        # ------------------------------------------------------------------ #
        # Serialise ExperimentConfig for workers                              #
        # Resolve output_dir to absolute so workers with a different CWD      #
        # write to the correct location.                                       #
        # Inject gpu_fraction so RLlib uses the same fractional value.        #
        # ------------------------------------------------------------------ #
        geometry = self._config.env.geometry
        absolute_geometry = geometry.model_copy(
            update={
                "stl_path":
                geometry.stl_path.resolve() if geometry.stl_path else None,
                "stl_paths": ([path.resolve() for path in geometry.
                               stl_paths] if geometry.stl_paths else None),
            })
        absolute_env = self._config.env.model_copy(
            update={
                "geometry":
                absolute_geometry,
                "waypoints_file": (
                    str(Path(self._config.env.waypoints_file).resolve()
                        ) if self._config.env.waypoints_file else None),
            })
        abs_config = self._config.model_copy(
            update={
                "experiment":
                self._config.experiment.model_copy(update={
                    "output_dir":
                    self._config.experiment.output_dir.resolve(),
                }),
                "env":
                absolute_env,
                "training":
                self._config.training.model_copy(update={
                    "num_gpus": gpu_fraction,
                }),
            })
        experiment_dict = abs_config.model_dump(by_alias=True, mode="json")
        from theseo_anysearch.experiments.custom_metrics import (
            discover_metric_sources,
        )

        metric_source_contents = {
            f"{scope}_metrics.py": path.read_text(encoding="utf-8")
            for scope, path in discover_metric_sources(self._config_path).items()
        }
        from theseo_anysearch.experiments.custom_rewards import (
            discover_reward_source,
        )

        reward_source = discover_reward_source(
            self._config_path,
            (
                self._config.env.rewards.provider.name
                if self._config.env.rewards.provider
                else None
            ),
        )
        reward_source_content = (
            reward_source.read_text(encoding="utf-8")
            if reward_source is not None
            else None
        )
        from theseo_anysearch.experiments.native_extensions import (
            NativeExtensionManifest,
            discover_native_manifest,
        )

        native_manifest_path = discover_native_manifest(self._config_path)
        native_extension_bundle = None
        if native_manifest_path is not None:
            native_manifest = NativeExtensionManifest.model_validate_json(
                native_manifest_path.read_text(encoding="utf-8")
            )
            native_binary = native_manifest_path.parent.joinpath(native_manifest.library)
            archived_manifest = native_manifest.model_copy(
                update={"library": native_binary.name}
            )
            native_extension_bundle = {
                "filename": native_binary.name,
                "binary": native_binary.read_bytes(),
                "manifest": archived_manifest.model_dump_json(indent=2),
            }

        # ------------------------------------------------------------------ #
        # MLflow parent run (optional)                                        #
        # ------------------------------------------------------------------ #
        mlflow_tracking_uri = ""
        mlflow_experiment_name = f"tune/{self._config.experiment.name}"
        parent_run_id = ""
        tracker: MLflowTracker | None = None

        if self._config.mlflow and self._config.mlflow.tracking_uri:
            mlflow_tracking_uri = self._config.mlflow.tracking_uri
            # Resolve relative sqlite:/// URIs so workers share the same DB.
            if mlflow_tracking_uri.startswith("sqlite:///"):
                rel = mlflow_tracking_uri[len("sqlite:///"):]
                if not Path(rel).is_absolute():
                    mlflow_tracking_uri = "sqlite:///" + str(
                        Path(rel).resolve())
            mlflow_cfg = MLflowConfig(tracking_uri=mlflow_tracking_uri)
            tracker = MLflowTracker(mlflow_cfg, mlflow_experiment_name)
            parent_run_id = tracker.start_run(
                run_name=self._config.experiment.name,
                tags={
                    "algorithm": self._config.training.algorithm,
                    "scheduler": tc.scheduler,
                    "num_samples": str(tc.num_samples),
                },
            ) or ""

        # ------------------------------------------------------------------ #
        # Output / storage paths                                              #
        #                                                                      #
        # Ray Tune writes its own internal bookkeeping under                   #
        # storage_path/{name}/driver_artifacts/ and per-trial state dirs.     #
        # On Windows these nested paths easily exceed MAX_PATH (260 chars)     #
        # when storage_path is the user's deep output_dir.                    #
        #                                                                      #
        # Fix: redirect Ray's storage to a short system temp dir.  Training   #
        # artifacts (checkpoints, videos, MLflow logs) are written by         #
        # _experiment_trainable directly to output_dir/run_tag/trial_id/,     #
        # which is fully under our control and separate from Ray's storage.   #
        # ------------------------------------------------------------------ #
        output_dir = abs_config.experiment.output_dir
        experiment_name = self._config.experiment.name
        output_dir.mkdir(parents=True, exist_ok=True)
        run_tag = self._tag or "latest"
        sweep_dir = output_dir.joinpath(experiment_name, run_tag)
        existing_trials = _existing_trial_dirs(sweep_dir)
        runtime_meta = _load_sweep_runtime(sweep_dir)

        # Ray internal storage: defaults to system temp so paths stay short.
        # Users may override via tune_config.ray_storage_dir if they need
        # results to survive a reboot or land in a specific location.
        # See spec/bugs.md for why this is separate from output_dir.
        _default_ray_storage = os.path.join(tempfile.gettempdir(),
                                            "anysearch_tune")
        ray_storage_root = str(
            tc.ray_storage_dir) if tc.ray_storage_dir else _default_ray_storage
        os.makedirs(ray_storage_root, exist_ok=True)
        storage_path = os.path.normpath(ray_storage_root)
        stored_ray_root = runtime_meta.get("ray_storage_root")
        if stored_ray_root and not tc.ray_storage_dir:
            ray_storage_root = str(stored_ray_root)
            os.makedirs(ray_storage_root, exist_ok=True)
            storage_path = os.path.normpath(ray_storage_root)
        # Truncate run_tag used as Tuner name so storage path stays ≤ ~180 chars
        # even with Ray's internal driver_artifacts and UUID temp filenames.
        tune_run_name = run_tag[:24] if len(run_tag) > 24 else run_tag
        continuation_mode = self._extra_trials > 0
        if existing_trials and not self._resume and not continuation_mode:
            raise FileExistsError(
                f"Tune sweep '{run_tag}' already exists at {sweep_dir}. "
                "Use --resume-tune to resume an interrupted sweep or --extra-trials N to append more trials."
            )
        if continuation_mode and not existing_trials:
            raise FileNotFoundError(
                f"No existing tune sweep '{run_tag}' was found at {sweep_dir}. "
                "Start the sweep first before appending extra trials.")

        if continuation_mode:
            segment_name = _segment_name(_next_segment_index(runtime_meta))
            trial_prefix = _segment_trial_prefix(segment_name)
            tune_run_name = _segment_tune_run_name(run_tag, segment_name)
            num_samples = self._extra_trials
            restore_path: Path | None = None
        elif self._resume:
            segments = runtime_meta.get("segments", [])
            last_segment = segments[-1] if isinstance(
                segments, list) and segments else {}
            segment_name = str(last_segment.get("segment_name") or "base")
            trial_prefix = str(
                last_segment.get("trial_prefix")
                or _segment_trial_prefix(segment_name))
            tune_run_name = str(
                last_segment.get("tune_run_name")
                or _segment_tune_run_name(run_tag, segment_name))
            restore_path = Path(storage_path, tune_run_name)
            num_samples = tc.num_samples
            if not ray_tune.Tuner.can_restore(str(restore_path)):
                raise FileNotFoundError(
                    f"Ray cannot restore sweep '{run_tag}' from {restore_path}. "
                    "If the original Ray temp storage was cleaned up, start a continuation with --extra-trials instead."
                )
        else:
            segment_name = "base"
            trial_prefix = ""
            tune_run_name = _segment_tune_run_name(run_tag, segment_name)
            num_samples = tc.num_samples
            restore_path = None

        # ------------------------------------------------------------------ #
        # Ray initialisation                                                  #
        # Keep temp dir short to avoid Windows MAX_PATH issues.               #
        # Users may override via tune_config.ray_temp_dir.                   #
        # ------------------------------------------------------------------ #
        _default_ray_temp = os.path.join(tempfile.gettempdir(),
                                         f"anysearch_tune_{os.getpid()}")
        ray_temp_dir = str(
            tc.ray_temp_dir) if tc.ray_temp_dir else _default_ray_temp
        os.makedirs(ray_temp_dir, exist_ok=True)
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        ray_was_initialized = ray.is_initialized()
        ray.init(**_ray_runtime_options(tc.ray_address, ray_temp_dir))


        # ------------------------------------------------------------------ #
        # Declare per-trial placement group bundles so RLlib's env-runner     #
        # workers have valid bundle indices.                                   #
        #                                                                      #
        # IMPORTANT: with_resources must wrap the BASE function before        #
        # with_parameters — applying it after with_parameters causes the      #
        # PlacementGroupFactory to be silently dropped in this Ray version.   #
        #                                                                      #
        # Tune creates 1 bundle by default (for the trial actor).  RLlib      #
        # (legacy API) schedules each rollout worker at the next bundle index  #
        # in the current placement group, so with N env runners we need        #
        # 1 + N bundles: bundle 0 for the trial actor, bundles 1..N for the  #
        # env runner actors.                                                   #
        # ------------------------------------------------------------------ #
        from ray.tune import PlacementGroupFactory

        num_env_runners = (self._config.training.num_env_runners +
                           self._config.evaluation.num_env_runners)

        driver_bundle: dict[str, float] = {"CPU": 1.0}
        if gpu_fraction > 0:
            driver_bundle["GPU"] = gpu_fraction
        rollout_worker_bundle: dict[str, float] = {"CPU": 1.0}
        rollout_gpu = self._config.training.num_gpus_per_env_runner
        if rollout_gpu > 0:
            rollout_worker_bundle["GPU"] = rollout_gpu
        worker_bundles = [
            dict(rollout_worker_bundle)
            for _ in range(self._config.training.num_env_runners)
        ] + [{
            "CPU": 1.0
        } for _ in range(self._config.evaluation.num_env_runners)]

        trainable_with_resources = ray_tune.with_resources(
            _experiment_trainable,
            PlacementGroupFactory(
                [driver_bundle] + worker_bundles,
                strategy="SPREAD",
            ),
        )

        # ------------------------------------------------------------------ #
        # Build the trainable with fixed parameters baked in                  #
        # ------------------------------------------------------------------ #
        trainable = ray_tune.with_parameters(
            trainable_with_resources,
            experiment_dict=experiment_dict,
            metric=tc.metric,
            mode=tc.mode,
            max_iterations=self._config.training.iterations,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_parent_run_id=parent_run_id,
            run_tag=run_tag,
            trial_prefix=trial_prefix,
            checkpoint_frequency=tc.checkpoint_frequency,
            preserve_trial_artifacts=tc.preserve_trial_artifacts,
            metric_source_contents=metric_source_contents,
            reward_source_content=reward_source_content,
            native_extension_bundle=native_extension_bundle,
        )

        # ------------------------------------------------------------------ #
        # Windows path normalisation patch (mirrors tune.py)                  #
        # ------------------------------------------------------------------ #
        apply_win_atomic_save_patch()

        # ------------------------------------------------------------------ #
        # Pre-flight summary                                                  #
        # ------------------------------------------------------------------ #
        sweep_dir.mkdir(parents=True, exist_ok=True)
        runtime_segments = runtime_meta.get("segments", [])
        if not isinstance(runtime_segments, list):
            runtime_segments = []
        active_segment_meta = {
            "segment_name": segment_name,
            "tune_run_name": tune_run_name,
            "trial_prefix": trial_prefix,
            "num_samples": num_samples,
            "status": "RUNNING",
        }
        runtime_meta = {
            **runtime_meta,
            "run_tag":
            run_tag,
            "experiment_name":
            experiment_name,
            "ray_storage_root":
            storage_path,
            "active_segment":
            segment_name,
            "segments": [
                *[
                    segment for segment in runtime_segments
                    if isinstance(segment, dict)
                    and segment.get("segment_name") != segment_name
                ],
                active_segment_meta,
            ],
        }
        _write_sweep_runtime(sweep_dir, runtime_meta)
        _print_sweep_overview(
            config_path=self._config_path,
            run_tag=run_tag,
            algorithm=self._config.training.algorithm,
            scheduler=tc.scheduler,
            num_samples=num_samples,
            max_concurrent=tc.max_concurrent,
            iterations=self._config.training.iterations,
            num_env_runners=self._config.training.num_env_runners,
            require_gpu=self._config.training.require_gpu,
            search_space=search_space,
            sweep_dir=sweep_dir,
            mlflow_tracking_uri=mlflow_tracking_uri,
        )

        stop = _tune_stop_criteria(
            tc,
            max_iterations=self._config.training.iterations,
        )
        # ------------------------------------------------------------------ #
        # TensorBoard callback (optional — skip if tensorboard not installed) #
        # Event files land under storage_path/run_tag/ (one subdir per trial) #
        # ------------------------------------------------------------------ #
        tb_callbacks = []
        try:
            from ray.tune.logger import TBXLoggerCallback
            tb_callbacks = [TBXLoggerCallback()]
        except ImportError:
            pass

        # ------------------------------------------------------------------ #
        # Launch the sweep                                                    #
        # ------------------------------------------------------------------ #
        results = None
        best_trial_id = None
        try:
            if restore_path is not None:
                tuner = ray_tune.Tuner.restore(
                    str(restore_path),
                    trainable=trainable,
                    resume_unfinished=True,
                    resume_errored=True,
                    param_space=search_space,
                )
            else:
                tuner = ray_tune.Tuner(
                    trainable,
                    param_space=search_space,
                    tune_config=ray_tune.TuneConfig(
                        metric=tc.metric,
                        mode=tc.mode,
                        num_samples=num_samples,
                        max_concurrent_trials=tc.max_concurrent,
                        scheduler=scheduler_obj,
                        search_alg=search_alg_obj,
                        reuse_actors=
                        False,  # reuse causes stale PG bundle refs after many trials
                        trial_name_creator=_trial_dirname,
                        trial_dirname_creator=_trial_dirname,
                    ),
                    run_config=ray_tune.RunConfig(
                        name=tune_run_name,
                        storage_path=storage_path,
                        verbose=1,
                        callbacks=tb_callbacks,
                        stop=stop,
                    ),
                )
            results = tuner.fit()

            try:
                best = results.get_best_result(metric=tc.metric, mode=tc.mode)
            except RuntimeError as exc:
                first_error = _first_trial_error(results)
                if first_error:
                    raise RuntimeError(
                        f"Tune sweep '{run_tag}' produced no valid '{tc.metric}' results. "
                        f"The first trial error was: {first_error}") from exc
                raise
            best_trial_id = best.metrics.get(
                "trial_id") if best.metrics else None
            _persist_trial_outcomes(
                sweep_dir,
                results,
                max_iterations=self._config.training.iterations,
            )
            runtime_meta["active_segment"] = None
            runtime_meta["segments"] = [{
                **segment,
                "status":
                "COMPLETED" if segment.get("segment_name") == segment_name else
                segment.get("status", "COMPLETED"),
            } for segment in runtime_meta["segments"]]
            _write_sweep_runtime(sweep_dir, runtime_meta)
            if tracker:
                _close_child_runs(mlflow_tracking_uri, parent_run_id, results)
                tracker.end_run("FINISHED")
            _print_sweep_replay_hint(sweep_dir, run_tag, best_trial_id)
            return {
                "best_config": best.config,
                "best_result": {
                    tc.metric: best.metrics.get(tc.metric),
                    "training_iteration":
                    best.metrics.get("training_iteration"),
                },
                "best_trial_id": best_trial_id,
                "experiment_dir": str(output_dir.joinpath(experiment_name)),
                "run_tag": run_tag,
                "segment_name": segment_name,
            }
        except KeyboardInterrupt:
            runtime_meta["segments"] = [{
                **segment,
                "status":
                "INTERRUPTED" if segment.get("segment_name") == segment_name
                else segment.get("status", "COMPLETED"),
            } for segment in runtime_meta["segments"]]
            _write_sweep_runtime(sweep_dir, runtime_meta)
            if tracker:
                _close_child_runs(mlflow_tracking_uri, parent_run_id, results)
                tracker.end_run("FAILED")
            _print_sweep_replay_hint(sweep_dir, run_tag, best_trial_id=None)
            raise
        except Exception:
            runtime_meta["segments"] = [{
                **segment,
                "status":
                "FAILED" if segment.get("segment_name") == segment_name else
                segment.get("status", "COMPLETED"),
            } for segment in runtime_meta["segments"]]
            _write_sweep_runtime(sweep_dir, runtime_meta)
            if tracker:
                _close_child_runs(mlflow_tracking_uri, parent_run_id, results)
                tracker.end_run("FAILED")
            raise
        finally:
            should_shutdown = _should_shutdown_ray(
                tc.shutdown_ray_on_complete,
                was_initialized=ray_was_initialized,
                address=tc.ray_address,
            )
            if should_shutdown and ray.is_initialized():
                ray.shutdown()
