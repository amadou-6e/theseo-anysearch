"""Deterministic evaluation orchestration for training iterations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict

from theseo_anysearch.rllib.trainer.results import TrainResult
from theseo_anysearch.rllib.trainer.runtime import (
    _append_trainer_stage_log,
    _log_trainer_stage,
)


class EvaluationOutcome(BaseModel):
    """Result of deterministic evaluation for one training iteration.

    Parameters
    ----------
    result : TrainResult
        Training result enriched with evaluation metrics.
    early_stop_triggered : bool
        Whether evaluation activated the configured early-stop condition.
    early_stop_decision : Any
        Detailed early-stop decision, when evaluated.
    checkpointed_for_best : bool
        Whether recording a new best trajectory created a checkpoint.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    result: TrainResult
    early_stop_triggered: bool
    early_stop_decision: Any
    checkpointed_for_best: bool


class EvaluationCoordinator:
    """Coordinate deterministic evaluation and its dependent concerns.

    Parameters
    ----------
    evaluation : Any
        Validated evaluation settings.
    early_stop_config : Any
        Validated training early-stop settings.
    early_stop_controller : Any
        Stateful early-stop evaluator.
    metric_providers : Any
        Loaded Python metric providers.
    native_extension : Any
        Loaded native extension, when configured.
    env_config : dict[str, Any]
        Runtime environment configuration.
    output_dir : pathlib.Path
        Training run output directory.
    output_store : Any
        Artifact output store.
    tensorboard_writer : Any
        Run-local TensorBoard writer.
    trajectory_writer : Any
        Optional trajectory writer.
    checkpoint : Callable[[], pathlib.Path]
        Callback used when a new best trajectory is recorded.
    multi_agent : bool
        Whether evaluation targets a multi-agent policy.
    experiment_name : str
        Experiment identifier stored with trajectories.
    run_id : str
        Run identifier stored with trajectories.
    """

    def __init__(
        self,
        *,
        evaluation: Any,
        early_stop_config: Any,
        early_stop_controller: Any,
        metric_providers: Any,
        native_extension: Any,
        env_config: dict[str, Any],
        output_dir: Path,
        output_store: Any,
        tensorboard_writer: Any,
        trajectory_writer: Any,
        checkpoint: Callable[[], Path],
        multi_agent: bool,
        experiment_name: str,
        run_id: str,
    ) -> None:
        self._evaluation = evaluation
        self._early_stop_config = early_stop_config
        self._early_stop_controller = early_stop_controller
        self._metric_providers = metric_providers
        self._native_extension = native_extension
        self._env_config = dict(env_config)
        self._output_dir = output_dir
        self._store = output_store
        self._tensorboard = tensorboard_writer
        self._trajectory_writer = trajectory_writer
        self._checkpoint = checkpoint
        self._multi_agent = multi_agent
        self._experiment_name = experiment_name
        self._run_id = run_id

    def evaluate(
        self,
        algorithm: Any,
        iteration: int,
        result: TrainResult,
        *,
        is_last_iteration: bool,
    ) -> EvaluationOutcome:
        """Evaluate a policy and enrich its current training result.

        Parameters
        ----------
        algorithm : Any
            RLlib algorithm or compatible policy implementation.
        iteration : int
            Current training iteration.
        result : TrainResult
            Normalized training result to enrich.
        is_last_iteration : bool
            Whether this is the configured final iteration.

        Returns
        -------
        EvaluationOutcome
            Enriched result and evaluation side-effect decisions.
        """
        from theseo_anysearch.experiments.custom_metrics import (
            EvaluationContext,
            compute_custom_metrics,
            merge_custom_metrics,
        )
        from theseo_anysearch.experiments.output import OutputStore
        from theseo_anysearch.experiments.trajectory import EpisodeRunMetrics
        from theseo_anysearch.rllib.trainer.early_stop import (
            heuristic_action_accuracy,
            heuristic_action_distance,
        )
        from theseo_anysearch.rllib.trainer.evaluation import EvaluationMetrics

        evaluation = self._evaluation
        evaluation_episodes = evaluation.episodes
        early_stop_config = self._early_stop_config
        early_stop_controller = self._early_stop_controller
        _env_cfg = self._env_config
        _store: OutputStore = self._store
        tb_writer = self._tensorboard
        _traj_writer = self._trajectory_writer
        _is_multi = self._multi_agent
        _is_last_iter = is_last_iteration
        _exp_name = self._experiment_name
        _run_id = self._run_id
        _checkpointed_for_best = False
        early_stop_triggered = False
        early_stop_decision = None
        algorithm = algorithm
        iteration = iteration
        _log_trainer_stage(
            f"Collecting {evaluation_episodes} deterministic evaluation "
            f"episodes for iteration {iteration}"
        )
        _append_trainer_stage_log(
            self._output_dir,
            f"Collecting deterministic evaluation batch for iteration "
            f"{iteration}",
        )
        evaluation_seed = evaluation.seed
        from theseo_anysearch.rllib.trainer.parallel_evaluation import (
            collect_rllib_evaluation_episodes,
        )

        episodes = collect_rllib_evaluation_episodes(
            algorithm,
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
            iteration=iteration,
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
            iteration,
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
        tb_writer.log_scalars(iteration, scalar_metrics)
        _store.write_json(
            f"evaluation/iter_{iteration:06d}.json",
            {
                "iteration": iteration,
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
                iteration,
                evaluation_reward_mean,
                _exp_name,
                _run_id,
                force=_is_last_iter or early_stop_triggered,
            )
            if "trajectories/best.json" in written:
                self._checkpoint()
                _checkpointed_for_best = True
        _append_trainer_stage_log(
            self._output_dir,
            f"Evaluation batch completed for iteration {iteration}: "
            f"{metrics.finish_count} goals reached",
        )
        return EvaluationOutcome(
            result=result,
            early_stop_triggered=early_stop_triggered,
            early_stop_decision=early_stop_decision,
            checkpointed_for_best=_checkpointed_for_best,
        )
