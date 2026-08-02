"""Custom training metric computation for trainer iterations."""

from __future__ import annotations

from typing import Any, Mapping

from theseo_anysearch.rllib.trainer.results import TrainResult


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
        if not custom_metrics:
            return result, {}

        updated = result.model_copy(
            update={"extra": {**result.extra, **custom_metrics}}
        )
        tensorboard_scalars = {
            f"train/custom/{key.removeprefix('training_')}": value
            for key, value in custom_metrics.items()
        }
        return updated, tensorboard_scalars
