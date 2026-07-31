"""Run, resume, inspect, and finalize single experiment executions."""

from __future__ import annotations

import itertools
import json
import secrets
import sys
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

_LOADING_WORDS = [
    "Loading", "Initializing", "Preparing", "Bootstrapping",
    "Configuring", "Warming up", "Assembling", "Starting",
    "Importing", "Building", "Setting up", "Provisioning",
]


@contextmanager
def _loading_throbber():
    """Display a cycling loading status using Rich Status while the block executes."""
    from rich.console import Console
    from rich.status import Status

    console = Console(stderr=True, highlight=False, force_terminal=True)
    words = itertools.cycle(_LOADING_WORDS)
    stop = threading.Event()

    with Status(f"{next(words)}...", console=console, spinner="dots") as status:
        def _tick():
            while not stop.wait(4.0):
                status.update(f"{next(words)}...")

        t = threading.Thread(target=_tick, daemon=True)
        t.start()
        try:
            yield
        finally:
            stop.set()
            t.join()


# Heavy dependencies (torch, ray, rllib) are imported lazily inside the methods
# that need them so that CLI commands like `list` and `inspect` start instantly.
if TYPE_CHECKING:
    from theseo_anysearch.experiments.models import ExperimentConfig
    from theseo_anysearch.experiments.output import OutputStore
    from theseo_anysearch.experiments.tracking import MLflowTracker
    from theseo_anysearch.rllib.trainer import Trainer
    from theseo_anysearch.rllib.trainer.base import TrainResult


class RunInfo(BaseModel):
    """Persisted metadata for a single experiment run.

    Parameters
    ----------
    run_id : str
        Unique identifier for the run directory.
    experiment_name : str
        Registered experiment name.
    start_time : str
        UTC ISO timestamp for run start.
    end_time : str | None
        UTC ISO timestamp for run end.
    status : str
        Current run status.
    checkpoint_iterations : list[int]
        Iterations for which checkpoints have been written.
    trajectory_iterations : list[int]
        Iterations for which periodic trajectories exist.
    render_files : list[str]
        Relative render artifact paths.
    mlflow_run_url : str | None
        Optional MLflow run URL for the experiment.
    """
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
    early_stop_reason: str | None = None
    early_stop_iteration: int | None = None
    early_stop_value: float | None = None
    early_stop_threshold: float | None = None


class InspectResult(BaseModel):
    """Structured result returned by run inspection.

    Parameters
    ----------
    run_id : str
        Unique identifier for the inspected run.
    experiment_name : str
        Experiment name.
    status : str
        Current run status.
    start_time : str
        UTC ISO timestamp for run start.
    end_time : str | None
        UTC ISO timestamp for run end.
    config : dict[str, Any]
        Parsed experiment configuration.
    checkpoint_iterations : list[int]
        Iterations with saved checkpoints.
    trajectory_iterations : list[int]
        Iterations with saved trajectories.
    render_files : list[str]
        Relative render artifact paths.
    mlflow_run_url : str | None
        Optional MLflow run URL for the experiment.
    """
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


def _append_run_stage(run_dir: Path, message: str) -> None:
    """Append a timestamped run-stage marker under the concrete run directory."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = run_dir.joinpath("debug_stage.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"[runner] {ts} {message}\n")


def _lookup_registered_name(run_dir: Path) -> str | None:
    """Return the registry name whose directory contains run_dir, or None."""
    try:
        from theseo_anysearch.cli.registry import load_registry
        reg = load_registry()
        resolved = run_dir.resolve()
        for name, dir_str in reg.items():
            try:
                resolved.relative_to(Path(dir_str).resolve())
                return name
            except ValueError:
                pass
    except Exception:
        pass
    return None


def _print_replay_hint(run_dir: Path, run_id: str, trajectory_iterations: list[int]) -> None:
    if not trajectory_iterations:
        return
    reg_name = _lookup_registered_name(run_dir)
    ref = f"{reg_name}:{run_id}" if reg_name else run_id
    try:
        import sys
        from rich.console import Console
        from rich.table import Table

        console = Console(file=sys.stdout, highlight=False)
        table = Table(box=None, show_header=False, padding=(0, 2, 0, 0))
        table.add_column(style="dim", min_width=16)
        table.add_column(style="cyan")
        table.add_row("Best episode", f"anysearch replay {ref} --best")
        table.add_row("All snapshots", f"anysearch replay {ref}")
        table.add_row("List available", f"anysearch replay {ref} list")
        table.add_row("Metrics UI", f"anysearch mlflow {reg_name or ''}")

        from rich.panel import Panel
        console.print(
            Panel(table, title=f"[bold]Run complete[/bold]  [dim]{ref}[/dim]", title_align="left"),
            new_line_start=True,
        )
    except Exception:
        sep = "-" * 62
        lines = [
            "",
            sep,
            f"  Replay (run: {ref})",
            f"    Best episode  :  anysearch replay {ref} --best",
            f"    All snapshots :  anysearch replay {ref}",
            f"    List available:  anysearch replay {ref} list",
            sep,
            "",
        ]
        print("\n".join(lines), flush=True)


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
    from theseo_anysearch.rllib.trainer import Trainer as _Trainer
    settings = config.to_settings().model_copy(
        update={
            "training": config.training.model_copy(update={"output_dir": output_dir})
        }
    )
    return _Trainer.from_settings(settings)


def _collect_heuristic_reference(
    config: ExperimentConfig,
    store: Any,
    run_dir: Path,
    run_id: str,
    tracker: Any,
    env_config: dict[str, Any],
    *,
    iteration: int,
) -> str:
    """Collect, persist, and log one configured heuristic trajectory."""

    from theseo_anysearch.experiments.trajectory import (
        collect_heuristic_episode,
        write_heuristic_trajectory,
    )

    heuristic_cfg = config.heuristic
    _append_run_stage(run_dir, "Collecting heuristic trajectory")
    episode = collect_heuristic_episode(
        env_config,
        heuristic_cfg.type,
        weight=heuristic_cfg.weight,
        seed=config.experiment.seed,
    )
    heuristic_path = write_heuristic_trajectory(
        store,
        episode,
        heuristic_type=heuristic_cfg.type,
        weight=heuristic_cfg.weight,
        iteration=iteration,
        experiment_name=config.experiment.name,
        run_id=run_id,
    )
    tracker.log_artifact(run_dir.joinpath(heuristic_path))
    _append_run_stage(run_dir, f"Heuristic trajectory written: {heuristic_path}")
    return heuristic_path


class ExperimentRunner:

    """Run, resume, repeat, inspect, and finalize single experiments.

    Parameters
    ----------
    config : ExperimentConfig
        Validated experiment configuration to execute.
    config_path : Path | None
        Optional source YAML path written into run artifacts.
    """

    def __init__(self, config: ExperimentConfig, config_path: Path | None = None) -> None:
        self._config = config
        self._config_path = config_path

    def run(self) -> RunInfo:
        # Register the run immediately using only stdlib so it shows in `anysearch list`
        # before any heavy imports (torch/ray) begin.
        run_id = _new_run_id()
        run_dir = self._config.run_output_dir.joinpath(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        _append_run_stage(run_dir, "Run directory created")
        run_info = RunInfo(
            run_id=run_id,
            experiment_name=self._config.experiment.name,
            start_time=_now_iso(),
        )
        (run_dir / "run.json").write_text(json.dumps(run_info.model_dump()))
        _append_run_stage(run_dir, "Initial run.json written")

        with _loading_throbber():
            _append_run_stage(run_dir, "Entering loading throbber setup")
            from theseo_anysearch.experiments.output import OutputStore
            from theseo_anysearch.experiments.tracking import MLflowTracker, _flatten_config

            run_dir = self._config.run_output_dir.joinpath(run_id)
            store = OutputStore(run_dir)
            _append_run_stage(run_dir, "OutputStore ready")

            if self._config_path:
                store.write_yaml("experiment.yaml", self._config_path)
            else:
                store.write_json(
                    "experiment.yaml",
                    self._config.model_dump(by_alias=True, mode="json"),
                )
            _append_run_stage(run_dir, "experiment.yaml written")
            from theseo_anysearch.experiments.custom_metrics import (
                copy_metric_sources,
            )

            copy_metric_sources(self._config_path, run_dir)
            from theseo_anysearch.experiments.custom_rewards import (
                copy_reward_source,
            )

            copy_reward_source(self._config_path, run_dir)
            from theseo_anysearch.experiments.native_extensions import copy_native_extension

            copy_native_extension(self._config_path, run_dir)
            _append_run_stage(run_dir, "Custom extension sources copied")

            tracker = MLflowTracker(
                _resolve_mlflow_config(self._config),
                self._config.experiment.name,
            )
            _append_run_stage(run_dir, "MLflowTracker created")
            tracker.start_run(run_name=run_id, tags={"project_run_id": run_id})
            _append_run_stage(run_dir, "MLflow run started")

            run_info = run_info.model_copy(update={"mlflow_run_url": tracker.run_url})
            store.write_json("run.json", run_info.model_dump())
            _append_run_stage(run_dir, "run.json updated with MLflow URL")
            standalone_heuristic = (
                self._config.training.algorithm.lower() == "heuristic"
            )
            trainer = None
            if not standalone_heuristic:
                trainer = _build_trainer(self._config, run_dir)
                _append_run_stage(run_dir, "Trainer built")
            else:
                _append_run_stage(run_dir, "Standalone heuristic mode selected")

        try:
            _append_run_stage(run_dir, "Logging MLflow params")
            tracker.log_params(_flatten_config(self._config))
            _append_run_stage(run_dir, "MLflow params logged")
            tracker.log_artifact(run_dir / "experiment.yaml")
            _append_run_stage(run_dir, "experiment.yaml logged to MLflow")

            if trainer is not None:
                _orig_hook = trainer.on_iteration_end

                def _combined_hook(result: TrainResult) -> None:
                    _orig_hook(result)
                    tracker.log_metrics(
                        result.standard_metrics(),
                        step=result.iteration,
                    )

                trainer.on_iteration_end = _combined_hook  # type: ignore[method-assign]

            if trainer is None:
                _collect_heuristic_reference(
                    self._config,
                    store,
                    run_dir,
                    run_id,
                    tracker,
                    self._config.env.to_runtime_dict(),
                    iteration=0,
                )
            else:
                _append_run_stage(run_dir, "Calling trainer.train()")
                trainer.train()
                _append_run_stage(run_dir, "trainer.train() returned")
                if self._config.heuristic.enabled:
                    _collect_heuristic_reference(
                        self._config,
                        store,
                        run_dir,
                        run_id,
                        tracker,
                        trainer._env_config_dict(),
                        iteration=self._config.training.iterations,
                    )

            tracker.end_run("FINISHED")
            _append_run_stage(run_dir, "MLflow run finished")
            return self._finalise(store, run_info, "COMPLETED")
        except KeyboardInterrupt:
            tracker.end_run("FAILED")
            _append_run_stage(run_dir, "KeyboardInterrupt during run")
            self._finalise(store, run_info, "INTERRUPTED")
            raise
        except Exception:
            tracker.end_run("FAILED")
            _append_run_stage(run_dir, "Exception during run")
            self._finalise(store, run_info, "FAILED")
            raise

    def resume(self, run_id: str) -> RunInfo:
        from theseo_anysearch.experiments.output import OutputStore

        run_dir = _find_run_dir(self._config.run_output_dir, run_id)
        store = OutputStore(run_dir)

        run_info = RunInfo.model_validate(store.read_json("run.json"))
        run_info = run_info.model_copy(update={"status": "RUNNING"})
        store.write_json("run.json", run_info.model_dump())

        try:
            with _loading_throbber():
                trainer = _build_trainer(self._config, run_dir)
            if not trainer.resume():
                raise FileNotFoundError(
                    f"No checkpoint found in {run_dir.joinpath('checkpoints')}. Cannot resume."
                )
            trainer.train()
            return self._finalise(store, run_info, "COMPLETED")
        except KeyboardInterrupt:
            self._finalise(store, run_info, "INTERRUPTED")
            raise
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
        from theseo_anysearch.experiments.output import OutputStore

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

        # Single runs — have a run.json
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

        # Tune sweep tags — layout is always output_base/<tag>/<trial_id>/ray_runtime.json
        seen_paths = {r["path"] for r in runs}

        def _sweep_entry(tag_dir: Path, trial_dirs: list[Path]) -> dict:
            still_running = any((d / "ray_runtime.json").exists() for d in trial_dirs)
            status = "RUNNING" if still_running else "COMPLETED"
            start_time = min((d.stat().st_mtime for d in trial_dirs), default=tag_dir.stat().st_mtime)
            start_iso = datetime.fromtimestamp(start_time, tz=timezone.utc).isoformat()
            return {
                "run_id": tag_dir.name,
                "experiment_name": "",
                "status": status,
                "start_time": start_iso,
                "path": str(tag_dir),
                "sweep_trials": str(len(trial_dirs)),
            }

        for candidate in output_base.iterdir():
            if not candidate.is_dir() or str(candidate) in seen_paths:
                continue
            child_trials = [d for d in candidate.iterdir() if d.is_dir() and (d / "ray_runtime.json").exists()]
            if child_trials:
                runs.append(_sweep_entry(candidate, child_trials))

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
        early_stop = store.read_json("early_stop.json") if store.exists("early_stop.json") else {}
        final_info = run_info.model_copy(
            update={
                "status": status,
                "end_time": _now_iso(),
                "checkpoint_iterations": checkpoint_iterations,
                "trajectory_iterations": trajectory_iterations,
                "early_stop_reason": early_stop.get("mode"),
                "early_stop_iteration": early_stop.get("iteration"),
                "early_stop_value": early_stop.get("value"),
                "early_stop_threshold": early_stop.get("threshold"),
            }
        )
        store.write_json("run.json", final_info.model_dump())
        _print_replay_hint(store.root, run_info.run_id, trajectory_iterations)
        return final_info
