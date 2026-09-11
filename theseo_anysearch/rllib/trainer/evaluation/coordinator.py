"""Deterministic evaluation orchestration for training iterations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from theseo_anysearch.rllib.trainer.results import TrainResult
from theseo_anysearch.rllib.trainer.runtime import (
    _append_trainer_stage_log,
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
    best_trajectory_written : bool
        Whether evaluation wrote a new best trajectory.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    result: TrainResult
    early_stop_triggered: bool
    early_stop_decision: Any
    best_trajectory_written: bool


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
    trajectory_reporter : Any
        Optional trajectory artifact reporter.
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
        trajectory_reporter: Any,
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
        self._trajectory_reporter = trajectory_reporter
        self._multi_agent = multi_agent
        self._experiment_name = experiment_name
        self._run_id = run_id

    def evaluate(
        self,
        iteration: int,
        result: TrainResult,
        *,
        is_last_iteration: bool,
        episodes: list[Any],
    ) -> EvaluationOutcome:
        """Evaluate a policy and enrich its current training result.

        Parameters
        ----------
        iteration : int
            Current training iteration.
        result : TrainResult
            Normalized training result to enrich.
        is_last_iteration : bool
            Whether this is the configured final iteration.
        episodes : list[Any]
            Episodes collected by RLlib's custom evaluation function.

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
        from theseo_anysearch.rllib.trainer.evaluation.evaluator import EvaluationMetrics

        evaluation = self._evaluation
        evaluation_episodes = evaluation.episodes
        early_stop_config = self._early_stop_config
        early_stop_controller = self._early_stop_controller
        _env_cfg = self._env_config
        _store: OutputStore = self._store
        tb_writer = self._tensorboard
        trajectory_reporter = self._trajectory_reporter
        _is_multi = self._multi_agent
        _is_last_iter = is_last_iteration
        _exp_name = self._experiment_name
        _run_id = self._run_id
        best_trajectory_written = False
        early_stop_triggered = False
        early_stop_decision = None
        iteration = iteration
        evaluation_seed = evaluation.seed
        metrics_factory = (
            EpisodeRunMetrics.from_multi_voxel_episodes
            if _is_multi
            else EpisodeRunMetrics.from_voxel_episodes
        )
        metrics = metrics_factory(episodes)
        if not _is_multi:
            from theseo_anysearch.environments.task_identity import (
                AcceptedTaskManifest,
                build_evaluation_suite,
                publish_or_load_evaluation_suite,
            )

            accepted_tasks = [
                AcceptedTaskManifest.model_validate(episode.accepted_task)
                for episode in episodes
                if getattr(episode, "accepted_task", None) is not None
            ]
            if accepted_tasks:
                suite = publish_or_load_evaluation_suite(
                    Path(self._output_dir, "evaluation", "suite.json"),
                    build_evaluation_suite(accepted_tasks),
                )
                result = result.model_copy(
                    update={
                        "extra": {
                            **result.extra,
                            "evaluation_suite_identity": suite.identity_sha256,
                        }
                    }
                )

        episode_rewards = [
            sum(episode.total_rewards) if _is_multi else episode.total_reward
            for episode in episodes
        ]
        evaluation_reward_mean = sum(episode_rewards) / len(episode_rewards)
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
                (
                    "eval/task/waypoint/" + key.removeprefix("evaluation_")
                    if "waypoint" in key
                    else "eval/custom/" + key.removeprefix("evaluation_")
                ): value
                for key, value in evaluation_custom.items()
            },
            "eval/task/episode_len_mean": evaluation_len_mean,
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
                        "success": (
                            bool(sum(episode.total_rewards) > 0.0)
                            if _is_multi
                            else bool(episode.success)
                        ),
                        "total_reward": float(episode_rewards[episode_index]),
                        "steps": len(episode.steps),
                        "accepted_task_identity": (
                            (getattr(episode, "accepted_task", None) or {}).get(
                                "identity_sha256"
                            )
                        ),
                    }
                    for episode_index, episode in enumerate(episodes)
                ],
            },
        )

        if trajectory_reporter is not None:
            best_trajectory_written = trajectory_reporter.record(
                episodes,
                iteration=iteration,
                reward_mean=evaluation_reward_mean,
                experiment_name=_exp_name,
                run_id=_run_id,
                force=_is_last_iter or early_stop_triggered,
            )
        _append_trainer_stage_log(
            self._output_dir,
            f"Evaluation batch completed for iteration {iteration}: "
            f"{metrics.finish_count} goals reached",
        )
        return EvaluationOutcome(
            result=result,
            early_stop_triggered=early_stop_triggered,
            early_stop_decision=early_stop_decision,
            best_trajectory_written=best_trajectory_written,
        )
