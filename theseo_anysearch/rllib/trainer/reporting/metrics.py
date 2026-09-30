"""Custom training metric computation for trainer iterations."""

from __future__ import annotations

from typing import Any, Mapping

from theseo_anysearch.rllib.trainer.results import TrainResult


def _numeric_leaves(value: Any) -> dict[str, float]:
    """Flatten numeric RLlib leaves by their final field name."""
    leaves: dict[str, float] = {}
    if not isinstance(value, Mapping):
        return leaves
    for key, item in value.items():
        if isinstance(item, Mapping):
            leaves.update(_numeric_leaves(item))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            leaves[str(key)] = float(item)
    return leaves


def canonical_rllib_metrics(rllib_result: Mapping[str, Any]) -> dict[str, float]:
    """Map available RLlib task and optimizer signals into stable namespaces."""
    optimizer_root = (
        rllib_result.get("learners")
        or rllib_result.get("learner_results")
        or (
            rllib_result.get("info", {}).get("learner", {})
            if isinstance(rllib_result.get("info"), Mapping)
            else {}
        )
    )
    all_leaves = _numeric_leaves(optimizer_root)
    optimizer_aliases = {
        "policy_loss": "policy_loss",
        "vf_loss": "value_loss",
        "value_loss": "value_loss",
        "entropy": "entropy",
        "mean_kl_loss": "approx_kl",
        "kl": "approx_kl",
        "clip_fraction": "clip_fraction",
        "vf_explained_var": "explained_variance",
        "explained_variance": "explained_variance",
        "cur_lr": "learning_rate",
        "learning_rate": "learning_rate",
        "grad_norm": "gradient_norm",
        "td_error": "td_loss",
        "td_loss": "td_loss",
        "q_value_mean": "q_value_mean",
        "epsilon": "exploration_epsilon",
        "num_entries": "replay_buffer_size",
    }
    metrics = {
        f"train/optimization/{canonical}": all_leaves[source]
        for source, canonical in optimizer_aliases.items()
        if source in all_leaves
    }
    env_leaves = _numeric_leaves(rllib_result.get("env_runners", {}))
    task_aliases = {
        "success_rate": "success_rate",
        "route_success_rate": "waypoint/route_success_rate",
        "waypoint_completion_fraction_mean": "waypoint/completion_fraction_mean",
        "waypoints_reached_mean": "waypoint/waypoints_reached_mean",
        "route_efficiency_mean": "waypoint/route_efficiency_mean",
        "collision_rate": "collision_rate",
        "invalid_action_rate": "invalid_action_rate",
        "truncation_rate": "truncation_rate",
    }
    metrics.update({
        f"train/task/{canonical}": env_leaves[source]
        for source, canonical in task_aliases.items()
        if source in env_leaves
    })
    return metrics


class TrainingMetricCoordinator:
    """Compute and merge Python and native training metrics.

    Parameters
    ----------
    providers : Any
        Loaded Python metric provider collection.
    native_extension : Any
        Loaded native extension, when configured.
    env_config : Mapping[str, Any]
        Runtime environment configuration exposed to metric providers.
    """

    def __init__(
        self,
        providers: Any,
        native_extension: Any,
        env_config: Mapping[str, Any],
    ) -> None:
        self._providers = providers
        self._native_extension = native_extension
        self._env_config = dict(env_config)

    def apply(
        self,
        iteration: int,
        result: TrainResult,
        rllib_result: dict[str, Any],
        duration_s: float,
    ) -> tuple[TrainResult, dict[str, float]]:
        """Apply configured custom metrics to an iteration result.

        Parameters
        ----------
        iteration : int
            Current training iteration.
        result : TrainResult
            Normalized result before custom metrics.
        rllib_result : dict[str, Any]
            Raw result returned by RLlib.
        duration_s : float
            Training iteration duration in seconds.

        Returns
        -------
        tuple[TrainResult, dict[str, float]]
            Updated result and TensorBoard-tagged custom scalar metrics.
        """
        from theseo_anysearch.experiments.custom_metrics import (
            TrainingContext,
            compute_custom_metrics,
            merge_custom_metrics,
        )
        from theseo_anysearch.experiments.native_extensions import (
            CAP_TRAINING_METRICS,
            validate_native_metrics,
        )

        context = TrainingContext(
            iteration=iteration,
            standard_metrics=result.standard_metrics(),
            rllib_result=rllib_result,
            environment_steps_total=result.environment_steps_total,
            duration_s=duration_s,
            env_config=dict(self._env_config),
        )
        native_enabled = (
            self._native_extension is not None
            and self._native_extension.capabilities & CAP_TRAINING_METRICS
        )
        native_raw = (
            self._native_extension.compute_metrics(
                "training",
                {
                    "iteration": context.iteration,
                    "standard_metrics": context.standard_metrics,
                    "environment_steps_total": context.environment_steps_total,
                    "duration_s": context.duration_s,
                    "env_config": context.env_config,
                },
            )
            if native_enabled
            else {}
        )
        reserved_names = set(result.standard_metrics())
        python_metrics = compute_custom_metrics(
            self._providers.training,
            context,
            reserved_names=reserved_names,
        )
        native_metrics = (
            validate_native_metrics(
                "training",
                native_raw,
                reserved_names=reserved_names,
            )
            if native_enabled
            else {}
        )
        custom_metrics = merge_custom_metrics(python_metrics, native_metrics)
        canonical = canonical_rllib_metrics(rllib_result)
        if not custom_metrics and not canonical:
            return result, {}

        updated = result.model_copy(
            update={"extra": {**result.extra, **custom_metrics, **canonical}}
        )
        tensorboard_scalars = {
            f"train/custom/{key.removeprefix('training_')}": value
            for key, value in custom_metrics.items()
        }
        tensorboard_scalars.update(canonical)
        return updated, tensorboard_scalars
