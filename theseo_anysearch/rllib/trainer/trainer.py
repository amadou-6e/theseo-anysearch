"""Concrete orchestration for training, evaluation, and checkpointing."""

from __future__ import annotations

import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

from theseo_anysearch.models import Settings
from theseo_anysearch.rllib.trainer.base import BaseTrainer
from theseo_anysearch.rllib.trainer.checkpointing import (
    CheckpointManager,
    CheckpointState,
)
from theseo_anysearch.rllib.trainer.evaluation_coordinator import EvaluationCoordinator
from theseo_anysearch.rllib.trainer.metrics import TrainingMetricCoordinator
from theseo_anysearch.rllib.trainer.reporting import _TensorBoardRunWriter
from theseo_anysearch.rllib.trainer.results import RllibTrainResult, TrainResult
from theseo_anysearch.rllib.trainer.runtime import (
    _append_trainer_stage_log,
    _detect_num_gpus,
    _log_trainer_stage,
    _resolve_pool_dir,
)


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
        self._episodes_total: int = 0
        self._output_dir: Path = Path(config.training.output_dir)
        self._checkpoints = CheckpointManager(self._output_dir)
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
            config.env.rewards.custom.name if config.env.rewards.custom else None,
        )
        write_reward_manifest(reward_provider, self._output_dir)
        from theseo_anysearch.experiments.native_extensions import NativeExtension

        native_manifest = self._output_dir.joinpath("native_extension", "extension.json")
        self._native_extension = NativeExtension.load(
            native_manifest if native_manifest.is_file() else None
        )

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
        from theseo_anysearch.experiments.native_extensions import CAP_REWARD

        if (
            native_manifest.is_file()
            and self._native_extension is not None
            and self._native_extension.capabilities & CAP_REWARD
        ):
            runtime["native_extension_manifest"] = str(native_manifest.resolve())
        else:
            reward_source = self._output_dir.joinpath("rewards.py")
            if reward_source.is_file():
                runtime["reward_module_path"] = str(reward_source.resolve())
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
        import warnings

        if self._algo is None:
            _log_trainer_stage("Building algorithm instance")
            _append_trainer_stage_log(self._output_dir, "Building algorithm instance")
            self._algo = self._build_algorithm()
            _log_trainer_stage("Algorithm instance ready")
            _append_trainer_stage_log(self._output_dir, "Algorithm instance ready")

        training = self._config.training
        evaluation = self._config.evaluation
        results: list[TrainResult] = []
        tb_writer = _TensorBoardRunWriter(self._output_dir)

        # --- Deterministic evaluation batch and trajectory writer setup ---
        traj_every = training.trajectory_every
        best_traj = training.best_trajectory
        _traj_writer = None
        _is_multi = training.algorithm == "multi_agent_voxel_ppo"
        _env_cfg = self._env_config_dict()

        from theseo_anysearch.experiments.output import OutputStore
        from theseo_anysearch.experiments.custom_metrics import (
            CustomMetricError,
        )

        training_metrics = TrainingMetricCoordinator(
            self._metric_providers,
            self._native_extension,
            _env_cfg,
        )
        _store = OutputStore(self._output_dir)
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
        if traj_every or best_traj or early_stop_config.enabled:
            from theseo_anysearch.experiments.trajectory import (
                MultiTrajectoryWriter,
                TrajectoryWriter,
            )

            _Writer = MultiTrajectoryWriter if _is_multi else TrajectoryWriter
            _traj_writer = _Writer(_store, traj_every, best_traj)

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
            trajectory_writer=_traj_writer,
            checkpoint=self.checkpoint,
            multi_agent=_is_multi,
            experiment_name=self._output_dir.parent.name,
            run_id=self._output_dir.name,
        )

        try:
            for i in range(self._iteration, training.iterations):
                _log_trainer_stage(
                    f"Starting train iteration {i + 1}/{training.iterations}"
                )
                _append_trainer_stage_log(
                    self._output_dir,
                    f"Starting train iteration {i + 1}/{training.iterations}",
                )
                t0 = time.perf_counter()
                rllib_result = self._algo.train()
                elapsed = time.perf_counter() - t0
                _log_trainer_stage(
                    f"Finished train iteration {i + 1}/{training.iterations} in {elapsed:.2f}s"
                )
                _append_trainer_stage_log(
                    self._output_dir,
                    f"Finished train iteration {i + 1}/{training.iterations} in {elapsed:.2f}s",
                )

                self._iteration = i + 1
                parsed = RllibTrainResult.from_raw(rllib_result)
                self._episodes_total = parsed.parse_episodes_total(
                ) or self._episodes_total

                result = TrainResult.from_rllib(
                    self._iteration,
                    rllib_result,
                    elapsed,
                )

                _is_last_iter = self._iteration == training.iterations
                _checkpointed_for_best = False
                early_stop_triggered = False
                early_stop_decision = None
                try:
                    evaluation_outcome = evaluation_coordinator.evaluate(
                        self._algo,
                        self._iteration,
                        result,
                        is_last_iteration=_is_last_iter,
                    )
                    result = evaluation_outcome.result
                    early_stop_triggered = evaluation_outcome.early_stop_triggered
                    early_stop_decision = evaluation_outcome.early_stop_decision
                    _checkpointed_for_best = (
                        evaluation_outcome.checkpointed_for_best
                    )
                except CustomMetricError:
                    raise
                except Exception as exc:
                    warnings.warn(
                        f"evaluation collection failed at iter {self._iteration}: {exc}",
                        stacklevel=2,
                    )

                result, training_scalars = training_metrics.apply(
                    self._iteration,
                    result,
                    rllib_result,
                    elapsed,
                )
                if training_scalars:
                    tb_writer.log_scalars(self._iteration, training_scalars)
                results.append(result)
                tb_writer.log_iteration(result)
                self.on_iteration_end(result)

                if (
                    self._iteration % training.checkpoint_interval == 0
                    and not _checkpointed_for_best
                ):
                    self.checkpoint()
                    _checkpointed_for_best = True
                if early_stop_triggered and early_stop_decision is not None:
                    if not _checkpointed_for_best:
                        self.checkpoint()
                    _store.write_json("early_stop.json", early_stop_decision.model_dump())
                    _append_trainer_stage_log(
                        self._output_dir,
                        f"Early stopping at iteration {self._iteration}: "
                        f"{early_stop_decision.mode}={early_stop_decision.value} "
                        f">= {early_stop_decision.threshold}",
                    )
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
