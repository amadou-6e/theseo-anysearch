"""Normalized result models produced by RLlib training."""

from __future__ import annotations

from numbers import Real
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IterationTimings(BaseModel):
    """Timing breakdown for one training iteration, in seconds."""

    model_config = ConfigDict(extra="forbid")

    rllib_iteration_ema_s: float | None = None
    rllib_training_step_ema_s: float | None = None
    training_step_calls: int | None = None
    sampling_ema_s: float | None = None
    learner_update_ema_s: float | None = None
    sync_weights_ema_s: float | None = None
    replay_add_ema_s: float | None = None
    replay_sample_ema_s: float | None = None
    replay_update_priorities_ema_s: float | None = None
    env_step_ema_s: float | None = None
    inference_ema_s: float | None = None
    env_to_module_connector_ema_s: float | None = None
    module_to_env_connector_ema_s: float | None = None
    anysearch_evaluation_s: float = 0.0
    anysearch_checkpoint_s: float = 0.0
    anysearch_reporting_s: float = 0.0

    @classmethod
    def from_rllib(cls, result: dict[str, Any]) -> "IterationTimings":
        """Extract stable timing fields from a raw RLlib result."""
        timers = result.get("timers") or {}
        env_runners = result.get("env_runners") or {}

        def seconds(source: dict[str, Any], key: str) -> float | None:
            value = source.get(key)
            return float(value) if isinstance(value, Real) else None

        return cls(
            rllib_iteration_ema_s=seconds(timers, "training_iteration"),
            rllib_training_step_ema_s=seconds(timers, "training_step"),
            training_step_calls=(
                int(result["num_training_step_calls_per_iteration"])
                if isinstance(
                    result.get("num_training_step_calls_per_iteration"),
                    Real,
                )
                else None
            ),
            sampling_ema_s=seconds(timers, "env_runner_sampling_timer"),
            learner_update_ema_s=seconds(timers, "learner_update_timer"),
            sync_weights_ema_s=seconds(timers, "synch_weights"),
            replay_add_ema_s=seconds(timers, "replay_buffer_add_data_timer"),
            replay_sample_ema_s=seconds(timers, "replay_buffer_sampling_timer"),
            replay_update_priorities_ema_s=seconds(
                timers, "replay_buffer_update_prios_timer"
            ),
            env_step_ema_s=seconds(env_runners, "env_step_timer"),
            inference_ema_s=seconds(env_runners, "rlmodule_inference_timer"),
            env_to_module_connector_ema_s=seconds(
                env_runners, "env_to_module_connector"
            ),
            module_to_env_connector_ema_s=seconds(
                env_runners, "module_to_env_connector"
            ),
        )

    def tensorboard_scalars(self, elapsed_s: float) -> dict[str, float]:
        """Return present timing values under stable TensorBoard tags."""
        values = self.model_dump(exclude_none=True)
        scalars = {f"performance/{name}": value for name, value in values.items()}
        scalars["performance/rllib_wall_time_s"] = elapsed_s
        if (
            self.rllib_training_step_ema_s is not None
            and self.training_step_calls is not None
        ):
            estimated_total = (
                self.rllib_training_step_ema_s * self.training_step_calls
            )
            scalars["performance/rllib_training_step_estimated_total_s"] = (
                estimated_total
            )
            scalars["performance/rllib_estimated_residual_s"] = max(
                elapsed_s - estimated_total,
                0.0,
            )
        return scalars


class RllibTrainResult(BaseModel):
    """Normalized view over RLlib training results.

    Parameters
    ----------
    env_runners : dict[str, Any]
        New API stack result block for env runner metrics.
    episode_reward_mean : float | None
        Legacy top-level mean episode reward.
    episode_len_mean : float | None
        Legacy top-level mean episode length.
    episodes_total : int | None
        Legacy total number of episodes seen.
    training_iteration : int | None
        RLlib-reported training iteration.
    time_this_iter_s : float | None
        Duration of the iteration in seconds.
    """
    model_config = ConfigDict(extra="allow")

    env_runners: dict[str, Any] = Field(default_factory=dict)
    episode_reward_mean: float | None = None
    episode_len_mean: float | None = None
    episodes_total: int | None = None
    training_iteration: int | None = None
    time_this_iter_s: float | None = None
    timesteps_total: int | None = None
    timers: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_raw(cls, result: dict[str, Any]) -> "RllibTrainResult":
        """Validate an unstructured RLlib result mapping.

        Parameters
        ----------
        result : dict[str, Any]
            Raw mapping returned by RLlib.

        Returns
        -------
        RllibTrainResult
            Validated result model.
        """
        return cls.model_validate(result)

    def parse_episode_return(self) -> float | None:
        """Return the mean episode return across RLlib API versions.

        Returns
        -------
        float | None
            Mean episode return, or ``None`` when unavailable.
        """
        value = self.env_runners.get("episode_return_mean")
        if value is not None:
            return float(value)
        return (
            float(self.episode_reward_mean)
            if self.episode_reward_mean is not None
            else None
        )

    def parse_episode_len(self) -> float | None:
        """Return the mean episode length across RLlib API versions.

        Returns
        -------
        float | None
            Mean episode length, or ``None`` when unavailable.
        """
        value = self.env_runners.get("episode_len_mean")
        if value is not None:
            return float(value)
        return (
            float(self.episode_len_mean)
            if self.episode_len_mean is not None
            else None
        )

    def parse_episodes_total(self) -> int | None:
        """Return the lifetime episode count across RLlib API versions.

        Returns
        -------
        int | None
            Lifetime episode count, or ``None`` when unavailable.
        """
        value = self.env_runners.get("num_episodes_lifetime")
        if value is not None:
            return int(value)
        return int(self.episodes_total) if self.episodes_total is not None else None

    def parse_environment_steps_total(self) -> int | None:
        """Return sampled environment steps across RLlib API versions.

        Returns
        -------
        int | None
            Lifetime sampled environment-step count, or ``None`` when unavailable.
        """
        value = self.env_runners.get("num_env_steps_sampled_lifetime")
        if value is not None:
            return int(value)
        return int(self.timesteps_total) if self.timesteps_total is not None else None


class TrainResult(BaseModel):
    """Project-level summary of one RLlib training iteration."""
    model_config = ConfigDict(extra="forbid")

    iteration: int
    episode_reward_mean: float
    episode_len_mean: float
    episodes_total: int
    elapsed_s: float
    environment_steps_total: int = 0
    episodes_this_iter: int = 0
    goals_reached_this_iter: int = 0
    goals_reached_total: int = 0
    training_success_rate: float = 0.0
    evaluation_episodes: int = 0
    evaluation_goals_reached: int = 0
    evaluation_success_rate: float = 0.0
    evaluation_status: str = "not_evaluated"
    extra: dict[str, Any] = Field(default_factory=dict)
    timings: IterationTimings = Field(default_factory=IterationTimings)

    def standard_metrics(self) -> dict[str, float]:
        """Return the shared numeric metric contract for every reporter."""
        metrics = {
            "episode_reward_mean": self.episode_reward_mean,
            "episode_len_mean": self.episode_len_mean,
            "episodes_total": float(self.episodes_total),
            "episodes_this_iter": float(self.episodes_this_iter),
            "goals_reached_this_iter": float(self.goals_reached_this_iter),
            "goals_reached_total": float(self.goals_reached_total),
            "training_success_rate": self.training_success_rate,
            "evaluation_episodes": float(self.evaluation_episodes),
            "evaluation_goals_reached": float(self.evaluation_goals_reached),
            "evaluation_success_rate": self.evaluation_success_rate,
            "elapsed_s": self.elapsed_s,
            "environment_steps_total": float(self.environment_steps_total),
        }
        metrics.update({
            key: float(value)
            for key, value in self.extra.items()
            if key.startswith(("evaluation_", "training_"))
            and isinstance(value, (int, float))
        })
        return metrics

    @classmethod
    def from_rllib(
        cls,
        iteration: int,
        rllib_result: dict[str, Any],
        elapsed_s: float,
    ) -> "TrainResult":
        """Build from the dict returned by ray.rllib.algorithms.Algorithm.train()."""
        parsed = RllibTrainResult.from_raw(rllib_result)
        required_metrics = {
            "episode reward mean": (
                parsed.parse_episode_return(),
                "env_runners.episode_return_mean or episode_reward_mean",
            ),
            "episode length mean": (
                parsed.parse_episode_len(),
                "env_runners.episode_len_mean or episode_len_mean",
            ),
            "episode count": (
                parsed.parse_episodes_total(),
                "env_runners.num_episodes_lifetime or episodes_total",
            ),
            "environment step count": (
                parsed.parse_environment_steps_total(),
                "env_runners.num_env_steps_sampled_lifetime or timesteps_total",
            ),
        }
        missing = [
            f"{name} ({keys})"
            for name, (value, keys) in required_metrics.items()
            if value is None
        ]
        if missing:
            raise ValueError(
                "RLlib training result is missing required metrics: "
                + "; ".join(missing)
            )

        return cls(
            iteration=iteration,
            episode_reward_mean=required_metrics["episode reward mean"][0],
            episode_len_mean=required_metrics["episode length mean"][0],
            episodes_total=required_metrics["episode count"][0],
            elapsed_s=elapsed_s,
            environment_steps_total=required_metrics["environment step count"][0],
            timings=IterationTimings.from_rllib(rllib_result),
            extra={
                "training_iteration": parsed.training_iteration or iteration,
                "time_this_iter_s": parsed.time_this_iter_s,
            },
        )
