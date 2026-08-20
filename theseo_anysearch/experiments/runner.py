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
    from theseo_anysearch.rllib.trainer.results import TrainResult


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
    mlflow_degraded : bool
        True when MLflow tracking was requested but at least one MLflow call
        failed during the run (tracking silently stopped working). False for
        both a healthy run and a run that never configured MLflow.
    mlflow_degraded_reason : str | None
        Human-readable reason for the most recent MLflow failure, if any.
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
    mlflow_degraded: bool = False
    mlflow_degraded_reason: str | None = None
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
    mlflow_degraded : bool
        True when MLflow tracking was requested but degraded during the run.
    mlflow_degraded_reason : str | None
        Human-readable reason for the degraded MLflow state, if any.
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
    mlflow_degraded: bool = False
    mlflow_degraded_reason: str | None = None


class StagingState(BaseModel):
    """Persisted position within an ordered staged training run."""

    model_config = ConfigDict(extra="forbid")

    stage_index: int = Field(default=0, ge=0)
    stage_name: str = ""
    completed_iterations: int = Field(default=0, ge=0)
    stage_started_at: int = Field(default=0, ge=0)
    completed_stages: list[str] = Field(default_factory=list)
    completion_state: dict[str, Any] = Field(default_factory=dict)


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


def _attach_tracking_hook(
    trainer: Trainer,
    tracker: Any | None,
    *,
    stage_index: int | None = None,
) -> None:
    """Attach MLflow reporting, including the active stage when configured."""
    if tracker is None:
        return
    _orig_hook = trainer.on_iteration_end

    def _combined_hook(result: TrainResult) -> None:
        _orig_hook(result)
        metrics = result.standard_metrics()
        if stage_index is not None:
            metrics["training_stage_index"] = float(stage_index)
        tracker.log_metrics(metrics, step=result.iteration)

    trainer.on_iteration_end = _combined_hook  # type: ignore[method-assign]


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
        from theseo_anysearch.environment_rules import preflight_environment_rules
        from theseo_anysearch.imitation.preflight import preflight_imitation_providers

        preflight_environment_rules(config, config_path)
        preflight_imitation_providers(config.imitation, config_path)
        self._config = config
        self._config_path = config_path

    @staticmethod
    def _write_staging_state(store: OutputStore, state: StagingState) -> None:
        store.write_json("staging_state.json", state.model_dump())

    def _run_staged_training(
        self,
        store: OutputStore,
        run_dir: Path,
        tracker: MLflowTracker | None,
        *,
        resume: bool = False,
    ) -> Trainer:
        """Execute configured stages while transferring policy state."""
        staging = self._config.staging
        if staging is None or not staging.enabled:
            raise RuntimeError("staged training requested without enabled staging")
        if resume and not staging.resume:
            raise ValueError("staging.resume is disabled for this experiment")

        state = (
            StagingState.model_validate(store.read_json("staging_state.json"))
            if resume and store.exists("staging_state.json")
            else StagingState()
        )
        previous_trainer: Trainer | None = None
        start_index = state.stage_index

        for stage_index in range(start_index, len(staging.stages)):
            stage = staging.stages[stage_index]
            resuming_active_stage = (
                resume
                and stage_index == start_index
                and state.stage_name == stage.name
            )
            if not resuming_active_stage:
                # Consecutive evaluation counters belong to one stage.  Carrying
                # them into a stage with different task settings can stop that
                # stage before it has had a chance to learn.
                store.remove("early_stop_state.json")
                store.remove("early_stop.json")
            stage_started_at = (
                state.stage_started_at
                if state.stage_name == stage.name
                else state.completed_iterations
            )
            stage_config = self._config.stage_experiment(
                stage_index,
                completed_iterations=stage_started_at,
            )
            trainer = _build_trainer(stage_config, run_dir)
            from theseo_anysearch.experiments.stage_completion import (
                StageCompletionController,
            )
            controller = StageCompletionController(
                stage.completion,
                stage_start_iteration=stage_started_at,
                state=state.completion_state,
            )
            original_iteration_hook = trainer.on_iteration_end

            def _stage_iteration_hook(result: TrainResult) -> None:
                original_iteration_hook(result)
                controller.evaluate(result)
                active_state = StagingState(
                    stage_index=stage_index,
                    stage_name=stage.name,
                    completed_iterations=result.iteration,
                    stage_started_at=stage_started_at,
                    completed_stages=state.completed_stages,
                    completion_state=controller.state,
                )
                self._write_staging_state(store, active_state)

            trainer.on_iteration_end = _stage_iteration_hook  # type: ignore[method-assign]
            trainer.should_stop_training = (  # type: ignore[method-assign]
                lambda result: controller.completed or controller.halted
            )
            _attach_tracking_hook(trainer, tracker, stage_index=stage_index)
            transition = stage.replay_transition or staging.replay_transition

            _append_run_stage(
                run_dir,
                f"Starting training stage {stage_index}: {stage.name}",
            )

            if resume and stage_index == start_index:
                if state.stage_name == stage.name or transition == "preserve":
                    if not trainer.resume():
                        raise FileNotFoundError(
                            "staging_state.json exists but no trainer checkpoint is available"
                        )
                elif stage_index > 0:
                    previous_stage = staging.stages[stage_index - 1]
                    previous_config = self._config.stage_experiment(
                        stage_index - 1,
                        completed_iterations=(
                            state.completed_iterations
                            - (previous_stage.completion.iteration_limit() or 0)
                        ),
                    )
                    checkpoint_trainer = _build_trainer(previous_config, run_dir)
                    if not checkpoint_trainer.resume():
                        raise FileNotFoundError(
                            "staging_state.json exists but no trainer checkpoint is available"
                        )
                    weights = checkpoint_trainer._algo.get_weights()
                    episodes_total = checkpoint_trainer._episodes_total
                    checkpoint_trainer._algo.stop()
                    trainer._algo = trainer._build_algorithm()
                    trainer._algo.set_weights(weights)
                    trainer._iteration = state.completed_iterations
                    trainer._episodes_total = episodes_total
            elif previous_trainer is not None:
                checkpoint_dir = previous_trainer.checkpoint()
                if transition == "preserve":
                    previous_trainer._algo.stop()
                    trainer.restore(checkpoint_dir)
                else:
                    weights = previous_trainer._algo.get_weights()
                    episodes_total = previous_trainer._episodes_total
                    previous_trainer._algo.stop()
                    trainer._algo = trainer._build_algorithm()
                    trainer._algo.set_weights(weights)
                    trainer._iteration = state.completed_iterations
                    trainer._episodes_total = episodes_total

            active_state = state.model_copy(
                update={
                    "stage_index": stage_index,
                    "stage_name": stage.name,
                    "stage_started_at": stage_started_at,
                }
            )
            self._write_staging_state(store, active_state)
            trainer.train()
            trainer.checkpoint()
            if store.exists("early_stop.json") and not controller.completed:
                raise RuntimeError(
                    f"training.early_stop stopped stage '{stage.name}' before "
                    "its completion condition succeeded"
                )
            if controller.halted:
                _append_run_stage(
                    run_dir,
                    f"Stopped training at incomplete stage {stage_index}: "
                    f"{stage.name} ({controller.reason})",
                )
                return trainer
            if not controller.completed:
                raise RuntimeError(
                    f"trainer returned before stage '{stage.name}' completed"
                )
            completed_stages = [*state.completed_stages, stage.name]
            state = StagingState(
                stage_index=stage_index + 1,
                stage_name="",
                completed_iterations=trainer._iteration,
                stage_started_at=trainer._iteration,
                completed_stages=completed_stages,
                completion_state={},
            )
            self._write_staging_state(store, state)
            _append_run_stage(
                run_dir,
                f"Completed training stage {stage_index}: {stage.name} "
                f"({controller.reason})",
            )
            previous_trainer = trainer
            resume = False

        if previous_trainer is None:
            final_index = len(staging.stages) - 1
            final_config = self._config.stage_experiment(
                final_index,
                completed_iterations=(
                    state.completed_iterations
                    - (staging.stages[final_index].completion.iteration_limit() or 0)
                ),
            )
            previous_trainer = _build_trainer(final_config, run_dir)
            if not previous_trainer.resume():
                raise FileNotFoundError("completed staged run has no checkpoint")
        return previous_trainer

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

            copy_reward_source(
                self._config_path,
                run_dir,
                (
                    self._config.env.rewards.provider.name
                    if self._config.env.rewards.provider
                    else None
                ),
            )
            from theseo_anysearch.experiments.custom_scenarios import (
                copy_scenario_source,
            )

            copy_scenario_source(
                self._config_path,
                run_dir,
                (
                    self._config.env.scenarios.provider.name
                    if self._config.env.scenarios.provider
                    else None
                ),
            )
            from theseo_anysearch.experiments.custom_imitation import (
                copy_generation_source,
            )

            copy_generation_source(
                self._config_path,
                run_dir,
                (
                    self._config.imitation.generation.provider.name
                    if self._config.imitation.enabled
                    else None
                ),
            )
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

            run_info = run_info.model_copy(
                update={
                    "mlflow_run_url": tracker.run_url,
                    "mlflow_degraded": tracker.degraded,
                    "mlflow_degraded_reason": tracker.degraded_reason,
                }
            )
            store.write_json("run.json", run_info.model_dump())
            _append_run_stage(run_dir, "run.json updated with MLflow URL")
            standalone_heuristic = (
                self._config.training.algorithm.lower() == "heuristic"
            )
            trainer = None
            staged_training = bool(
                self._config.staging is not None
                and self._config.staging.enabled
            )
            if not standalone_heuristic and not staged_training:
                trainer = _build_trainer(self._config, run_dir)
                _append_run_stage(run_dir, "Trainer built")
            elif staged_training:
                _append_run_stage(run_dir, "Staged training selected")
            else:
                _append_run_stage(run_dir, "Standalone heuristic mode selected")

        try:
            _append_run_stage(run_dir, "Logging MLflow params")
            tracker.log_params(_flatten_config(self._config))
            _append_run_stage(run_dir, "MLflow params logged")
            tracker.log_artifact(run_dir / "experiment.yaml")
            _append_run_stage(run_dir, "experiment.yaml logged to MLflow")

            if trainer is not None:
                _attach_tracking_hook(trainer, tracker)

            if standalone_heuristic:
                _collect_heuristic_reference(
                    self._config,
                    store,
                    run_dir,
                    run_id,
                    tracker,
                    self._config.env.to_runtime_dict(),
                    iteration=0,
                )
            elif staged_training:
                _append_run_stage(run_dir, "Calling staged training")
                trainer = self._run_staged_training(store, run_dir, tracker)
                _append_run_stage(run_dir, "Staged training returned")
                if self._config.heuristic.enabled:
                    _collect_heuristic_reference(
                        self._config,
                        store,
                        run_dir,
                        run_id,
                        tracker,
                        trainer._env_config_dict(),
                        iteration=trainer._iteration,
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
            return self._finalise(store, run_info, "COMPLETED", tracker)
        except KeyboardInterrupt:
            tracker.end_run("FAILED")
            _append_run_stage(run_dir, "KeyboardInterrupt during run")
            self._finalise(store, run_info, "INTERRUPTED", tracker)
            raise
        except Exception:
            tracker.end_run("FAILED")
            _append_run_stage(run_dir, "Exception during run")
            self._finalise(store, run_info, "FAILED", tracker)
            raise

    def resume(self, run_id: str) -> RunInfo:
        from theseo_anysearch.experiments.output import OutputStore

        run_dir = _find_run_dir(self._config.run_output_dir, run_id)
        store = OutputStore(run_dir)

        run_info = RunInfo.model_validate(store.read_json("run.json"))
        run_info = run_info.model_copy(update={"status": "RUNNING"})
        store.write_json("run.json", run_info.model_dump())

        try:
            staged_training = bool(
                self._config.staging is not None
                and self._config.staging.enabled
            )
            if staged_training:
                trainer = self._run_staged_training(
                    store,
                    run_dir,
                    None,
                    resume=True,
                )
            else:
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
            mlflow_degraded=run_info.mlflow_degraded,
            mlflow_degraded_reason=run_info.mlflow_degraded_reason,
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
        tracker: Any = None,
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
        update: dict[str, Any] = {
            "status": status,
            "end_time": _now_iso(),
            "checkpoint_iterations": checkpoint_iterations,
            "trajectory_iterations": trajectory_iterations,
            "early_stop_reason": early_stop.get("mode"),
            "early_stop_iteration": early_stop.get("iteration"),
            "early_stop_value": early_stop.get("value"),
            "early_stop_threshold": early_stop.get("threshold"),
        }
        if tracker is not None:
            # Re-read the tracker's degraded state at finalisation time: a
            # failure may occur any time after the initial run.json write
            # (e.g. mid-training log_metrics calls), so this is the
            # authoritative last word on whether tracking stayed healthy.
            update["mlflow_degraded"] = tracker.degraded
            update["mlflow_degraded_reason"] = tracker.degraded_reason
            update["mlflow_run_url"] = tracker.run_url
        final_info = run_info.model_copy(update=update)
        store.write_json("run.json", final_info.model_dump())
        _print_replay_hint(store.root, run_info.run_id, trajectory_iterations)
        return final_info
