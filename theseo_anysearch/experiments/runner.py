from __future__ import annotations

import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

import warnings

from theseo_anysearch.experiments.models import ExperimentConfig
from theseo_anysearch.experiments.output import OutputStore
from theseo_anysearch.experiments.tracking import MLflowTracker, _flatten_config
from theseo_anysearch.experiments.trajectory import (
    TrajectoryWriter, collect_eval_episode,
    MultiTrajectoryWriter, collect_multi_eval_episode,
)
from theseo_anysearch.rllib.trainer import Trainer
from theseo_anysearch.rllib.trainer.base import TrainResult


class RunInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    experiment_name: str
    start_time: str
    end_time: str | None = None
    status: str = "RUNNING"
    checkpoint_iterations: list[int] = Field(default_factory=list)
    trajectory_iterations: list[int] = Field(default_factory=list)
    render_files: list[str] = Field(default_factory=list)
    mlflow_run_url: str | None = None


class InspectResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    experiment_name: str
    status: str
    start_time: str
    end_time: str | None
    config: dict[str, Any]
    checkpoint_iterations: list[int]
    trajectory_iterations: list[int]
    render_files: list[str]
    mlflow_run_url: str | None


def _new_run_id() -> str:
    return secrets.token_hex(4)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _find_run_dir(output_base: Path, run_id: str) -> Path:
    for candidate in output_base.rglob(run_id):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Run '{run_id}' not found under '{output_base}'. "
        "Pass --output-dir if the experiment used a different location."
    )


def _resolve_mlflow_config(config: ExperimentConfig):
    """
    Return an MLflowConfig with an explicit tracking_uri.

    If tracking is disabled (mlflow key absent from YAML) → returns None.
    If tracking_uri is already set → returns config unchanged.
    If tracking_uri is None → defaults to a SQLite DB in the experiment
    output directory (e.g. runtime/experiments/mlflow.db), keeping all
    experiment data together and avoiding interference from system-level
    MLflow configuration.
    """
    from theseo_anysearch.experiments.models import MLflowConfig

    mlflow_cfg = config.mlflow
    if mlflow_cfg is None:
        return None
    if mlflow_cfg.tracking_uri:
        return mlflow_cfg
    Path("experiments").mkdir(exist_ok=True)
    return mlflow_cfg.model_copy(
        update={"tracking_uri": "sqlite:///experiments/mlflow.db"}
    )


def _build_trainer(config: ExperimentConfig, output_dir: Path) -> Trainer:
    settings = config.to_settings().model_copy(
        update={
            "training": config.training.model_copy(update={"output_dir": output_dir})
        }
    )
    return Trainer.from_settings(settings)


class ExperimentRunner:
    def __init__(self, config: ExperimentConfig, config_path: Path | None = None) -> None:
        self._config = config
        self._config_path = config_path

    def run(self) -> RunInfo:
        run_id = _new_run_id()
        run_dir = self._config.run_output_dir.joinpath(run_id)
        store = OutputStore(run_dir)

        if self._config_path:
            store.write_yaml("experiment.yaml", self._config_path)
        else:
            store.write_json(
                "experiment.yaml",
                self._config.model_dump(by_alias=True, mode="json"),
            )

        tracker = MLflowTracker(
            _resolve_mlflow_config(self._config),
            self._config.experiment.name,
        )
        tracker.start_run(run_name=run_id, tags={"project_run_id": run_id})

        run_info = RunInfo(
            run_id=run_id,
            experiment_name=self._config.experiment.name,
            start_time=_now_iso(),
            mlflow_run_url=tracker.run_url,
        )
        store.write_json("run.json", run_info.model_dump())

        try:
            trainer = _build_trainer(self._config, run_dir)
            tracker.log_params(_flatten_config(self._config))
            tracker.log_artifact(run_dir / "experiment.yaml")

            env_cfg = self._config.env
            env_config_dict = {
                "stl_path": str(env_cfg.stl_path) if env_cfg.stl_path else None,
                "scale": env_cfg.scale,
                "agent_count": env_cfg.agent_count,
                "max_steps": env_cfg.max_steps,
                "seed": env_cfg.seed,
                "obs_mode": env_cfg.obs_mode,
                "box_radius": env_cfg.box_radius,
                "box_radii": env_cfg.box_radii,
                "ray_max_len": env_cfg.ray_max_len,
                "trail_mode": env_cfg.trail_mode,
                "geometry_boxes": env_cfg.geometry_boxes,
                "waypoints_file": env_cfg.waypoints_file,
                "step_cost": env_cfg.step_cost,
                "goal_reward": env_cfg.goal_reward,
                "distance_shaping": env_cfg.distance_shaping,
            }
            is_multi_agent = self._config.training.algorithm == "multi_agent_voxel_ppo"
            if is_multi_agent:
                traj_writer: TrajectoryWriter | MultiTrajectoryWriter = MultiTrajectoryWriter(
                    store=store,
                    trajectory_every=self._config.training.trajectory_every,
                    best_trajectory=self._config.training.best_trajectory,
                )
            else:
                traj_writer = TrajectoryWriter(
                    store=store,
                    trajectory_every=self._config.training.trajectory_every,
                    best_trajectory=self._config.training.best_trajectory,
                )

            _orig_hook = trainer.on_iteration_end

            def _combined_hook(result: TrainResult) -> None:
                _orig_hook(result)
                tracker.log_metrics(
                    {
                        "episode_reward_mean": result.episode_reward_mean,
                        "episode_len_mean": result.episode_len_mean,
                        "episodes_total": float(result.episodes_total),
                        "elapsed_s": result.elapsed_s,
                    },
                    step=result.iteration,
                )
                # Collect one eval episode and record trajectory
                try:
                    if is_multi_agent:
                        episode = collect_multi_eval_episode(
                            trainer._algo, env_config_dict, seed=result.iteration
                        )
                    else:
                        episode = collect_eval_episode(
                            trainer._algo, env_config_dict, seed=result.iteration
                        )
                    traj_writer.record(episode)
                    traj_writer.on_iteration_end(
                        result.iteration,
                        result.episode_reward_mean,
                        self._config.experiment.name,
                        run_id,
                    )
                except Exception as exc:
                    warnings.warn(f"trajectory collection failed at iter {result.iteration}: {exc}", stacklevel=2)

            trainer.on_iteration_end = _combined_hook  # type: ignore[method-assign]

            trainer.train()
            tracker.end_run("FINISHED")
            return self._finalise(store, run_info, "COMPLETED")
        except Exception:
            tracker.end_run("FAILED")
            self._finalise(store, run_info, "FAILED")
            raise

    def resume(self, run_id: str) -> RunInfo:
        run_dir = _find_run_dir(self._config.run_output_dir, run_id)
        store = OutputStore(run_dir)

        run_info = RunInfo.model_validate(store.read_json("run.json"))
        run_info = run_info.model_copy(update={"status": "RUNNING"})
        store.write_json("run.json", run_info.model_dump())

        try:
            trainer = _build_trainer(self._config, run_dir)
            if not trainer.resume():
                raise FileNotFoundError(
                    f"No checkpoint found in {run_dir.joinpath('checkpoints')}. Cannot resume."
                )
            trainer.train()
            return self._finalise(store, run_info, "COMPLETED")
        except Exception:
            self._finalise(store, run_info, "FAILED")
            raise

    def repeat(self, run_id: str) -> RunInfo:
        src_run_dir = _find_run_dir(self._config.run_output_dir, run_id)
        src_yaml = src_run_dir.joinpath("experiment.yaml")
        if not src_yaml.exists():
            raise FileNotFoundError(
                f"experiment.yaml not found in run '{run_id}' - cannot repeat."
            )

        from theseo_anysearch.experiments.loader import load_experiment

        repeated_config = load_experiment(src_yaml)
        runner = ExperimentRunner(repeated_config, src_yaml)
        return runner.run()

    @staticmethod
    def inspect(run_id: str, output_base: Path) -> InspectResult:
        run_dir = _find_run_dir(output_base, run_id)
        store = OutputStore(run_dir)
        run_info = RunInfo.model_validate(store.read_json("run.json"))

        checkpoint_iterations = sorted(
            int(path.split("iter_")[-1])
            for path in store.list_dirs("checkpoints")
            if "iter_" in path
        )
        trajectory_iterations = sorted(
            int(Path(path).stem.split("iter_")[-1])
            for path in store.list("trajectories")
            if path.endswith(".json") and "iter_" in path
        )
        render_files = sorted(store.list("renders"))

        config_data: dict[str, Any] = {}
        if store.exists("experiment.yaml"):
            try:
                import yaml

                config_data = yaml.safe_load(run_dir.joinpath("experiment.yaml").read_text()) or {}
            except Exception:
                config_data = {"error": "could not parse experiment.yaml"}

        return InspectResult(
            run_id=run_id,
            experiment_name=run_info.experiment_name,
            status=run_info.status,
            start_time=run_info.start_time,
            end_time=run_info.end_time,
            config=config_data,
            checkpoint_iterations=checkpoint_iterations,
            trajectory_iterations=trajectory_iterations,
            render_files=render_files,
            mlflow_run_url=run_info.mlflow_run_url,
        )

    @staticmethod
    def list_runs(output_base: Path) -> list[dict[str, str]]:
        runs: list[dict[str, str]] = []
        if not output_base.exists():
            return runs

        for run_json in output_base.rglob("run.json"):
            try:
                info = RunInfo.model_validate(json.loads(run_json.read_text()))
            except Exception:
                continue
            runs.append(
                {
                    "run_id": info.run_id,
                    "experiment_name": info.experiment_name,
                    "status": info.status,
                    "start_time": info.start_time,
                    "path": str(run_json.parent),
                }
            )
        return sorted(runs, key=lambda run: run["start_time"])

    def _finalise(
        self,
        store: OutputStore,
        run_info: RunInfo,
        status: str,
    ) -> RunInfo:
        checkpoint_iterations = sorted(
            int(path.split("iter_")[-1])
            for path in store.list_dirs("checkpoints")
            if "iter_" in path
        )
        trajectory_iterations = sorted(
            int(Path(path).stem.split("iter_")[-1])
            for path in store.list("trajectories")
            if path.endswith(".json") and "iter_" in path
        )
        final_info = run_info.model_copy(
            update={
                "status": status,
                "end_time": _now_iso(),
                "checkpoint_iterations": checkpoint_iterations,
                "trajectory_iterations": trajectory_iterations,
            }
        )
        store.write_json("run.json", final_info.model_dump())
        return final_info
