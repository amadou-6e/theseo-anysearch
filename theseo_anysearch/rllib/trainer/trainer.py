"""Concrete orchestration for training, evaluation, and checkpointing."""

from __future__ import annotations

import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

from theseo_anysearch.rllib.trainer.base import BaseTrainer
from theseo_anysearch.rllib.trainer.checkpointing import (
    CheckpointManager,
    CheckpointState,
)
from theseo_anysearch.rllib.trainer.evaluation.coordinator import EvaluationCoordinator
from theseo_anysearch.rllib.trainer.lifecycle import TrainingLifecycle
from theseo_anysearch.rllib.trainer.reporting.metrics import TrainingMetricCoordinator
from theseo_anysearch.rllib.trainer.reporting.tensorboard import _TensorBoardRunWriter
from theseo_anysearch.rllib.trainer.reporting.trajectories import TrajectoryReporter
from theseo_anysearch.rllib.trainer.results import RllibTrainResult, TrainResult
from theseo_anysearch.rllib.trainer.runtime import (
    _append_trainer_stage_log,
    _detect_num_gpus,
    _resolve_pool_dir,
)
from theseo_anysearch.settings import Settings
from theseo_anysearch.worlds import world_contract


class Trainer(BaseTrainer):
    """Coordinate one configured RLlib training run.

    Parameters
    ----------
    config : Settings
        Validated experiment settings.

    Notes
    -----
    Subclasses provide algorithm construction. Evaluation, metrics,
    reporting, runtime setup, and checkpoint persistence are delegated to
    focused collaborators.
    """
    algorithm_name: str | None = None

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._algo: Any = None
        self._iteration: int = 0
        self._imitation_result: Any = None
        self._episodes_total: int = 0
        self._output_dir: Path = Path(config.training.output_dir)
        self._checkpoints = CheckpointManager(
            self._output_dir,
            world_contract(config.env.to_runtime_dict()),
        )
        self._lifecycle = TrainingLifecycle(self._output_dir)
        from theseo_anysearch.experiments.custom_metrics import (
            load_metric_providers,
            write_metric_manifest,
        )

        metric_config_path = self._output_dir.joinpath("experiment.yaml")
        self._metric_providers = load_metric_providers(
            metric_config_path if metric_config_path.is_file() else None
        )
        write_metric_manifest(self._metric_providers, self._output_dir)
        from theseo_anysearch.experiments.custom_rewards import (
            load_reward_provider,
            write_reward_manifest,
        )

        reward_source = self._output_dir.joinpath("rewards.py")
        reward_provider = load_reward_provider(
            reward_source if reward_source.is_file() else None,
            config.env.rewards.provider.name if config.env.rewards.provider else None,
        )
        write_reward_manifest(reward_provider, self._output_dir)
        from theseo_anysearch.experiments.native_extensions import NativeExtension

        native_manifest = self._output_dir.joinpath("native_extension", "extension.json")
        self._native_extension = NativeExtension.load(
            native_manifest if native_manifest.is_file() else None
        )
        self._curriculum = None
        if config.env.waypoint_curriculum.enabled:
            from theseo_anysearch.rllib.trainer.curriculum.waypoint import (
                CurriculumController,
            )

            self._curriculum = CurriculumController(config.env, config.evaluation)

    @classmethod
    def from_settings(cls, config: Settings) -> "Trainer":
        """Construct the matching trainer for validated settings.

        Parameters
        ----------
        config : Settings
            Validated experiment settings.

        Returns
        -------
        Trainer
            Algorithm-specific trainer instance.

        Raises
        ------
        NotImplementedError
            If no trainer is registered for the configured algorithm.
        """
        if cls is not Trainer:
            return cls(config)

        from theseo_anysearch.rllib.algorithms.registry import get_trainer_class

        trainer_cls = get_trainer_class(config.training.algorithm)
        return trainer_cls(config)

    # ------------------------------------------------------------------
    # Abstract
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_algorithm(self) -> Any:
        """
        Construct the RLlib Algorithm (or compatible duck-typed object).
        Called once at the start of train() or restore().
        """

    # ------------------------------------------------------------------
    # Optional hook
    # ------------------------------------------------------------------

    def on_iteration_end(self, result: TrainResult) -> None:
        """Handle a completed training iteration.

        Parameters
        ----------
        result : TrainResult
            Completed normalized training result.
        """

    def should_stop_training(self, result: TrainResult) -> bool:
        """Return whether an external lifecycle controller requests a stop."""
        return False

    def _env_config_dict(self) -> dict:
        """Build the runtime environment configuration.

        Returns
        -------
        dict
            Configuration passed to the registered environment.
        """
        env = self._config.env
        runtime = env.to_runtime_dict()
        runtime["geometry_pool"] = _resolve_pool_dir(env.geometry.pool)
        native_manifest = self._output_dir.joinpath("native_extension", "extension.json")
        from theseo_anysearch.experiments.native_extensions import (
            CAP_OUTCOME,
            CAP_PREDICATE,
            CAP_REWARD,
            CAP_SCENARIO,
        )

        if (
            native_manifest.is_file()
            and self._native_extension is not None
            and self._native_extension.capabilities & (CAP_REWARD | CAP_PREDICATE | CAP_OUTCOME | CAP_SCENARIO)
        ):
            runtime["native_extension_manifest"] = str(native_manifest.resolve())
        else:
            reward_source = self._output_dir.joinpath("rewards.py")
            if reward_source.is_file():
                runtime["reward_module_path"] = str(reward_source.resolve())
        scenario_source = self._output_dir.joinpath("scenarios.py")
        if scenario_source.is_file():
            runtime["scenario_module_path"] = str(scenario_source.resolve())
        return runtime

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self) -> list[TrainResult]:
        """Run the configured training loop.

        Returns
        -------
        list[TrainResult]
            One normalized result per completed training iteration.
        """
        self._algo = self._lifecycle.ensure_algorithm(
            self._algo,
            self._build_algorithm,
        )

        if self._config.imitation.enabled and self._iteration == 0:
            from theseo_anysearch.imitation.pretraining import (
                run_imitation_pretraining,
            )

            self._imitation_result = run_imitation_pretraining(
                self._algo,
                self._env_config_dict(),
                self._config.imitation,
                self._output_dir,
            )

        training = self._config.training
        evaluation = self._config.evaluation
        results: list[TrainResult] = []
        tb_writer = _TensorBoardRunWriter(self._output_dir)
        if self._imitation_result is not None:
            tb_writer.log_scalars(
                0,
                {
                    "imitation/cache_hit": float(self._imitation_result.cache_hit),
                    "imitation/validation_accuracy": self._imitation_result.validation_accuracy,
                    "imitation/validation_loss": self._imitation_result.best_validation_loss,
                    "imitation/pre_rl_success_rate": (
                        self._imitation_result.pre_rl_success_rate or 0.0
                    ),
                },
            )

        # --- Deterministic evaluation batch and trajectory writer setup ---
        traj_every = training.trajectory_every
        best_traj = training.best_trajectory
        _is_multi = training.algorithm == "multi_agent_voxel_ppo"
        _env_cfg = self._env_config_dict()

        from theseo_anysearch.experiments.output import OutputStore
        training_metrics = TrainingMetricCoordinator(
            self._metric_providers,
            self._native_extension,
            _env_cfg,
        )
        _store = OutputStore(self._output_dir)
        if self._curriculum is not None:
            self._curriculum.initialize(self._algo, _store, _env_cfg)
        from theseo_anysearch.rllib.trainer.early_stop import (
            EarlyStopState,
            TrainingEarlyStopController,
        )
        early_stop_config = training.early_stop
        early_stop_state = (
            EarlyStopState.model_validate(_store.read_json("early_stop_state.json"))
            if early_stop_config.enabled and _store.exists("early_stop_state.json")
            else EarlyStopState()
        )
        early_stop_controller = TrainingEarlyStopController(
            early_stop_config, early_stop_state
        )
        trajectory_reporter = TrajectoryReporter.create(
            _store,
            trajectory_every=traj_every,
            best_trajectory=best_traj,
            multi_agent=_is_multi,
            enabled=early_stop_config.enabled,
        )
        evaluation_coordinator = EvaluationCoordinator(
            evaluation=evaluation,
            early_stop_config=early_stop_config,
            early_stop_controller=early_stop_controller,
            metric_providers=self._metric_providers,
            native_extension=self._native_extension,
            env_config=_env_cfg,
            output_dir=self._output_dir,
            output_store=_store,
            tensorboard_writer=tb_writer,
            trajectory_reporter=trajectory_reporter,
            multi_agent=_is_multi,
            experiment_name=self._output_dir.parent.name,
            run_id=self._output_dir.name,
        )

        try:
            for i in range(self._iteration, training.iterations):
                execution = self._lifecycle.run_iteration(
                    self._algo,
                    i + 1,
                    training.iterations,
                )
                rllib_result = execution.result
                elapsed = execution.duration_s
                self._iteration = i + 1
                parsed = RllibTrainResult.from_raw(rllib_result)
                parsed_episodes_total = parsed.parse_episodes_total()
                if parsed_episodes_total is not None:
                    self._episodes_total = parsed_episodes_total

                result = TrainResult.from_rllib(
                    self._iteration,
                    rllib_result,
                    elapsed,
                )
                if self._curriculum is not None:
                    result.extra.update(self._curriculum.stage_metric())
                evaluation_due = self._iteration % evaluation.frequency == 0
                rllib_evaluation_episodes = getattr(
                    self._algo,
                    "_anysearch_evaluation_episodes",
                    None,
                )
                if evaluation_due and rllib_evaluation_episodes is None:
                    raise RuntimeError(
                        "RLlib evaluation completed without AnySearch "
                        "evaluation episodes"
                    )
                if rllib_evaluation_episodes is not None:
                    delattr(self._algo, "_anysearch_evaluation_episodes")

                _is_last_iter = self._iteration == training.iterations
                _checkpointed_for_best = False
                early_stop_triggered = False
                early_stop_decision = None
                evaluation_started = time.perf_counter()
                try:
                    if rllib_evaluation_episodes is not None:
                        evaluation_outcome = evaluation_coordinator.evaluate(
                            self._iteration,
                            result,
                            is_last_iteration=_is_last_iter,
                            episodes=rllib_evaluation_episodes,
                        )
                        result = evaluation_outcome.result
                        early_stop_triggered = evaluation_outcome.early_stop_triggered
                        early_stop_decision = evaluation_outcome.early_stop_decision
                        if evaluation_outcome.best_trajectory_written:
                            checkpoint_started = time.perf_counter()
                            self.checkpoint()
                            result.timings.anysearch_checkpoint_s += (
                                time.perf_counter() - checkpoint_started
                            )
                            _checkpointed_for_best = True
                    if self._curriculum is not None:
                        curriculum_metrics = self._curriculum.evaluate(
                            self._algo, self._iteration, _env_cfg, _store
                        )
                        result.extra.update(curriculum_metrics)
                        tb_writer.log_scalars(self._iteration, curriculum_metrics)
                finally:
                    result.timings.anysearch_evaluation_s = max(
                        time.perf_counter() - evaluation_started
                        - result.timings.anysearch_checkpoint_s,
                        0.0,
                    )

                reporting_started = time.perf_counter()
                result, training_scalars = training_metrics.apply(
                    self._iteration,
                    result,
                    rllib_result,
                    elapsed,
                )
                if training_scalars:
                    tb_writer.log_scalars(self._iteration, training_scalars)
                results.append(result)
                self.on_iteration_end(result)
                result.timings.anysearch_reporting_s = (
                    time.perf_counter() - reporting_started
                )

                if (
                    self._iteration % training.checkpoint_interval == 0
                    and not _checkpointed_for_best
                ):
                    checkpoint_started = time.perf_counter()
                    self.checkpoint()
                    result.timings.anysearch_checkpoint_s += (
                        time.perf_counter() - checkpoint_started
                    )
                    _checkpointed_for_best = True
                if early_stop_triggered and early_stop_decision is not None:
                    if not _checkpointed_for_best:
                        checkpoint_started = time.perf_counter()
                        self.checkpoint()
                        result.timings.anysearch_checkpoint_s += (
                            time.perf_counter() - checkpoint_started
                        )
                    tb_writer.log_iteration(result)
                    _store.write_json("early_stop.json", early_stop_decision.model_dump())
                    _append_trainer_stage_log(
                        self._output_dir,
                        f"Early stopping at iteration {self._iteration}: "
                        f"{early_stop_decision.mode}={early_stop_decision.value} "
                        f">= {early_stop_decision.threshold}",
                    )
                    break
                tb_writer.log_iteration(result)
                if self.should_stop_training(result):
                    break
        finally:
            tb_writer.close()

        return results

    def checkpoint(self) -> Path:
        """Save RLlib and project state for the current iteration.

        Returns
        -------
        pathlib.Path
            Created project checkpoint directory.
        """
        return self._checkpoints.save(
            self._algo,
            CheckpointState(
                iteration=self._iteration,
                episodes_total=self._episodes_total,
                rllib_path="",
            ),
        )

    def restore(self, checkpoint_dir: Path) -> None:
        """Restore algorithm weights and project state.

        Parameters
        ----------
        checkpoint_dir : pathlib.Path
            Project checkpoint directory to restore.
        """
        if self._algo is None:
            self._algo = self._build_algorithm()
        state = self._checkpoints.restore(self._algo, checkpoint_dir)
        self._iteration = state.iteration
        self._episodes_total = state.episodes_total

    def resume(self) -> bool:
        """Restore the latest checkpoint when available.

        Returns
        -------
        bool
            True when a checkpoint was restored, otherwise False.
        """
        checkpoint_dir = self._checkpoints.latest()
        if checkpoint_dir is None:
            return False
        self.restore(checkpoint_dir)
        return True
