"""Concrete orchestration for training, evaluation, and checkpointing."""

from __future__ import annotations

import json
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import ClassVar

from theseo_anysearch.models import Settings
from theseo_anysearch.rllib.trainer.base import BaseTrainer
from theseo_anysearch.rllib.trainer.reporting import _TensorBoardRunWriter
from theseo_anysearch.rllib.trainer.results import RllibTrainResult, TrainResult
from theseo_anysearch.rllib.trainer.runtime import (
    _append_trainer_stage_log,
    _detect_num_gpus,
    _log_trainer_stage,
    _resolve_pool_dir,
)


class Trainer(BaseTrainer):
    """
    Project-level wrapper around a Ray RLlib Algorithm.

    Subclasses implement _build_algorithm() to construct and return a
    configured ray.rllib.algorithms.Algorithm instance.  This base class
    handles the iteration loop, checkpointing, restore, and resume on top
    of whatever RLlib trainer is provided.

    For unit tests, _build_algorithm() can return any object that exposes
    .train() → dict, .save(path: str) → Any, .restore(path: str) → None.
    """

    algorithm_name: ClassVar[str | None] = None
    _registry: ClassVar[dict[str, type["Trainer"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.algorithm_name:
            cls._registry[cls.algorithm_name] = cls

    def __init__(self, config: Settings) -> None:
        self._config = config
        self._algo: Any = None
        self._iteration: int = 0
        self._episodes_total: int = 0
        self._output_dir: Path = Path(config.training.output_dir)
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

        trainer_cls = cls._registry.get(config.training.algorithm.lower())
        if trainer_cls is None:
            raise NotImplementedError(
                f"No Trainer registered for algorithm '{config.training.algorithm}'."
            )
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
        """Called after every training iteration. Override to add callbacks."""

    def _env_config_dict(self) -> dict:
        """Build the env config dict passed to VoxelEnv / MultiVoxelEnv."""
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
        """
        Run the training loop for config.training.iterations steps.
        Returns a list of TrainResult, one per iteration.
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
        evaluation_episodes = evaluation.episodes
        _traj_writer = None
        _is_multi = training.algorithm == "multi_agent_voxel_ppo"
        _env_cfg = self._env_config_dict()

        from theseo_anysearch.experiments.output import OutputStore
        from theseo_anysearch.experiments.trajectory import EpisodeRunMetrics
        from theseo_anysearch.rllib.trainer.evaluation import EvaluationMetrics
        from theseo_anysearch.experiments.custom_metrics import (
            CustomMetricError,
            EvaluationContext,
            TrainingContext,
            compute_custom_metrics,
            merge_custom_metrics,
        )

        _store = OutputStore(self._output_dir)
        from theseo_anysearch.rllib.trainer.early_stop import (
            EarlyStopState,
            TrainingEarlyStopController,
            heuristic_action_accuracy,
            heuristic_action_distance,
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
            _run_id = self._output_dir.name
            _exp_name = self._output_dir.parent.name

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
                    _log_trainer_stage(
                        f"Collecting {evaluation_episodes} deterministic evaluation "
                        f"episodes for iteration {self._iteration}"
                    )
                    _append_trainer_stage_log(
                        self._output_dir,
                        f"Collecting deterministic evaluation batch for iteration "
                        f"{self._iteration}",
                    )
                    evaluation_seed = evaluation.seed
                    from theseo_anysearch.rllib.trainer.parallel_evaluation import (
                        collect_rllib_evaluation_episodes,
                    )

                    episodes = collect_rllib_evaluation_episodes(
                        self._algo,
                        _env_cfg,
                        evaluation_episodes,
                        seed=evaluation_seed,
                        multi_agent=_is_multi,
                        num_envs_per_env_runner=evaluation.num_envs_per_env_runner,
                    )
                    metrics_factory = (
                        EpisodeRunMetrics.from_multi_voxel_episodes
                        if _is_multi
                        else EpisodeRunMetrics.from_voxel_episodes
                    )
                    metrics = metrics_factory(episodes)

                    evaluation_reward_mean = sum(
                        episode.total_reward for episode in episodes
                    ) / len(episodes)
                    evaluation_len_mean = sum(
                        len(episode.steps) for episode in episodes
                    ) / len(episodes)
                    evaluation_factory = (
                        EvaluationMetrics.from_multi_voxel_episodes
                        if _is_multi
                        else EvaluationMetrics.from_voxel_episodes
                    )
                    success_metrics = evaluation_factory(
                        episodes,
                        _env_cfg,
                        min_success_rate=evaluation.min_success_rate,
                    )
                    standardized = success_metrics.scalar_metrics()
                    evaluation_context = EvaluationContext(
                        iteration=self._iteration,
                        episodes=tuple(episodes),
                        standard_metrics={
                            **result.standard_metrics(), **standardized,
                            **metrics.as_scalar_dict(),
                            "evaluation_reward_mean": evaluation_reward_mean,
                            "evaluation_len_mean": evaluation_len_mean,
                        },
                        env_config=dict(_env_cfg),
                        final_infos=tuple(
                            dict(getattr(episode, "final_info", None) or {})
                            for episode in episodes
                        ),
                    )
                    evaluation_reserved = (
                        set(result.standard_metrics()) | set(standardized)
                        | set(metrics.as_scalar_dict())
                        | {"evaluation_reward_mean", "evaluation_len_mean"}
                    )
                    from theseo_anysearch.experiments.native_extensions import (
                        CAP_EVALUATION_METRICS,
                        validate_native_metrics,
                    )

                    native_has_evaluation = (
                        self._native_extension is not None
                        and self._native_extension.capabilities & CAP_EVALUATION_METRICS
                    )
                    native_raw = (
                        self._native_extension.compute_metrics(
                            "evaluation",
                            {
                                "iteration": evaluation_context.iteration,
                                "standard_metrics": evaluation_context.standard_metrics,
                                "env_config": evaluation_context.env_config,
                                "final_infos": evaluation_context.final_infos,
                            },
                        )
                        if native_has_evaluation else {}
                    )
                    python_evaluation_custom = compute_custom_metrics(
                        self._metric_providers.evaluation, evaluation_context,
                        reserved_names=evaluation_reserved,
                    )
                    native_evaluation_custom = (
                        validate_native_metrics(
                            "evaluation", native_raw,
                            reserved_names=evaluation_reserved,
                        )
                        if native_has_evaluation else {}
                    )
                    evaluation_custom = merge_custom_metrics(
                        python_evaluation_custom, native_evaluation_custom
                    )
                    heuristic_accuracy = None
                    heuristic_distance = None
                    heuristic_compared_states = 0
                    if early_stop_config.enabled and early_stop_config.mode in {
                        "heuristic_accuracy", "heuristic_distance"
                    }:
                        from theseo_anysearch.experiments.trajectory import collect_heuristic_episode

                        heuristic_episodes = [
                            collect_heuristic_episode(
                                _env_cfg,
                                early_stop_config.heuristic_type,
                                weight=early_stop_config.heuristic_weight,
                                seed=evaluation_seed + episode_index,
                            )
                            for episode_index in range(evaluation_episodes)
                        ]
                        if early_stop_config.mode == "heuristic_accuracy":
                            heuristic_accuracy, heuristic_compared_states = heuristic_action_accuracy(
                                episodes, heuristic_episodes
                            )
                        else:
                            heuristic_distance, heuristic_compared_states = heuristic_action_distance(
                                episodes,
                                heuristic_episodes,
                                metric=early_stop_config.heuristic_distance_metric,
                            )
                    early_stop_decision = early_stop_controller.evaluate(
                        self._iteration,
                        reward_mean=evaluation_reward_mean,
                        goal_finishes=metrics.finish_count,
                        heuristic_accuracy=heuristic_accuracy,
                        heuristic_distance=heuristic_distance,
                    )
                    early_stop_triggered = early_stop_decision.triggered
                    if early_stop_config.enabled:
                        _store.write_json(
                            "early_stop_state.json",
                            early_stop_controller.state.model_dump(),
                        )
                    result = result.model_copy(
                        update={
                            "evaluation_episodes": len(episodes),
                            "evaluation_goals_reached": metrics.finish_count,
                            "evaluation_success_rate": metrics.finish_rate,
                            "evaluation_status": success_metrics.status,

                            "extra": {
                                **result.extra,
                                "evaluation_reward_mean": evaluation_reward_mean,
                                "evaluation_len_mean": evaluation_len_mean,
                                **standardized,
                                **metrics.as_scalar_dict(),
                                **evaluation_custom,
                                "evaluation_heuristic_accuracy": heuristic_accuracy,
                                "evaluation_heuristic_distance": heuristic_distance,
                                "evaluation_heuristic_compared_states": heuristic_compared_states,
                                "early_stop_consecutive_matches": early_stop_decision.consecutive_matches,
                                "early_stop_triggered": early_stop_triggered,
                            },
                        }
                    )
                    scalar_metrics = {
                        **metrics.as_scalar_dict(),
                        **success_metrics.tensorboard_metrics(),
                        **{
                            f"eval/custom/{key.removeprefix('evaluation_')}": value
                            for key, value in evaluation_custom.items()
                        },

                        "eval/reward_mean": evaluation_reward_mean,
                        "eval/episode_len_mean": evaluation_len_mean,
                    }
                    tb_writer.log_scalars(self._iteration, scalar_metrics)
                    _store.write_json(
                        f"evaluation/iter_{self._iteration:06d}.json",
                        {
                            "iteration": self._iteration,
                            "seed_start": evaluation_seed,
                            "episode_count": len(episodes),
                            "num_env_runners": evaluation.num_env_runners,
                            "num_envs_per_env_runner": evaluation.num_envs_per_env_runner,
                            "max_evaluation_concurrency": min(
                                evaluation_episodes,
                                max(evaluation.num_env_runners, 1)
                                * evaluation.num_envs_per_env_runner,
                            ),
                            "goals_reached": metrics.finish_count,
                            "success_rate": metrics.finish_rate,
                            "reward_mean": evaluation_reward_mean,
                            "episode_len_mean": evaluation_len_mean,
                            "status": result.evaluation_status,
                            "minimum_success_rate": evaluation.min_success_rate,
                            "summary": success_metrics.model_dump(),

                            "metrics": scalar_metrics,
                            "early_stop": (
                                early_stop_decision.model_dump()
                                if early_stop_config.enabled
                                else None
                            ),
                            "episodes": [
                                {
                                    "seed": evaluation_seed + episode_index,
                                    "success": bool(episode.success),
                                    "total_reward": float(episode.total_reward),
                                    "steps": len(episode.steps),
                                }
                                for episode_index, episode in enumerate(episodes)
                            ],
                        },
                    )

                    if _traj_writer is not None:
                        for episode in episodes:
                            _traj_writer.record(episode)
                        written = _traj_writer.on_iteration_end(
                            self._iteration,
                            evaluation_reward_mean,
                            _exp_name,
                            _run_id,
                            force=_is_last_iter or early_stop_triggered,
                        )
                        if "trajectories/best.json" in written:
                            self.checkpoint()
                            _checkpointed_for_best = True
                    _append_trainer_stage_log(
                        self._output_dir,
                        f"Evaluation batch completed for iteration {self._iteration}: "
                        f"{metrics.finish_count} goals reached",
                    )
                except CustomMetricError:
                    raise
                except Exception as exc:
                    warnings.warn(
                        f"evaluation collection failed at iter {self._iteration}: {exc}",
                        stacklevel=2,
                    )

                training_context = TrainingContext(
                    iteration=self._iteration,
                    standard_metrics=result.standard_metrics(),
                    rllib_result=rllib_result,
                    environment_steps_total=result.environment_steps_total,
                    duration_s=elapsed,
                    env_config=dict(_env_cfg),
                )
                from theseo_anysearch.experiments.native_extensions import (
                    CAP_TRAINING_METRICS,
                    validate_native_metrics,
                )

                native_has_training = (
                    self._native_extension is not None
                    and self._native_extension.capabilities & CAP_TRAINING_METRICS
                )
                native_raw = (
                    self._native_extension.compute_metrics(
                        "training",
                        {
                            "iteration": training_context.iteration,
                            "standard_metrics": training_context.standard_metrics,
                            "environment_steps_total": training_context.environment_steps_total,
                            "duration_s": training_context.duration_s,
                            "env_config": training_context.env_config,
                        },
                    )
                    if native_has_training else {}
                )
                training_reserved = set(result.standard_metrics())
                python_training_custom = compute_custom_metrics(
                    self._metric_providers.training, training_context,
                    reserved_names=training_reserved,
                )
                native_training_custom = (
                    validate_native_metrics(
                        "training", native_raw,
                        reserved_names=training_reserved,
                    )
                    if native_has_training else {}
                )
                training_custom = merge_custom_metrics(
                    python_training_custom, native_training_custom
                )
                if training_custom:
                    result = result.model_copy(
                        update={"extra": {**result.extra, **training_custom}}
                    )
                    tb_writer.log_scalars(
                        self._iteration,
                        {
                            f"train/custom/{key.removeprefix('training_')}": value
                            for key, value in training_custom.items()
                        },
                    )
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
        """
        Save a project checkpoint for the current iteration.
        Calls algo.save() for RLlib state, then writes state.json + latest.json.
        Returns the checkpoint directory path.
        """
        ckpt_dir = self._output_dir / "checkpoints" / f"iter_{self._iteration:06d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # RLlib legacy save() may return a subdirectory path (e.g. ckpt_dir/checkpoint_000001).
        # Store the returned path so restore() uses the exact location.
        rllib_path = str(ckpt_dir)
        if self._algo is not None:
            returned = self._algo.save(str(ckpt_dir))
            if isinstance(returned, str) and returned:
                rllib_path = returned

        self._write_state(ckpt_dir, rllib_path=rllib_path)
        self._write_latest_pointer(ckpt_dir)
        return ckpt_dir

    def restore(self, checkpoint_dir: Path) -> None:
        """Restore algorithm weights and project state from a checkpoint directory."""
        if self._algo is None:
            self._algo = self._build_algorithm()

        state_file = checkpoint_dir / "state.json"
        rllib_path = str(checkpoint_dir)
        if state_file.exists():
            state = json.loads(state_file.read_text())
            self._iteration = state["iteration"]
            self._episodes_total = state.get("episodes_total", 0)
            rllib_path = state.get("rllib_path", str(checkpoint_dir))

        self._algo.restore(rllib_path)

    def resume(self) -> bool:
        """
        Restore from the latest checkpoint if one exists.
        Returns True when a checkpoint was found and loaded, False otherwise.
        """
        latest_ptr = self._output_dir / "checkpoints" / "latest.json"
        if not latest_ptr.exists():
            return False
        info = json.loads(latest_ptr.read_text())
        self.restore(Path(info["path"]))
        return True

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_state(self,
                     checkpoint_dir: Path,
                     rllib_path: str | None = None) -> None:
        state = {
            "iteration": self._iteration,
            "episodes_total": self._episodes_total,
            "rllib_path": rllib_path or str(checkpoint_dir),
        }
        (checkpoint_dir / "state.json").write_text(json.dumps(state, indent=2))

    def _write_latest_pointer(self, checkpoint_dir: Path) -> None:
        ptr = {"path": str(checkpoint_dir), "iteration": self._iteration}
        (self._output_dir / "checkpoints" / "latest.json").write_text(
            json.dumps(ptr, indent=2))
