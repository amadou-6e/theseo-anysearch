"""Normalized result models produced by RLlib training."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
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

    # TODO! make these properties e.g. @property def mean_reward(self) → float
    def parse_episode_return(self) -> float:
        """Return the mean episode return across RLlib API versions.

        Returns
        -------
        float
            Mean episode return, or zero when unavailable.
        """
        value = self.env_runners.get("episode_return_mean")
        if value is not None:
            return float(value)
        return float(self.episode_reward_mean
                     ) if self.episode_reward_mean is not None else 0.0

    def parse_episode_len(self) -> float:
        """Return the mean episode length across RLlib API versions.

        Returns
        -------
        float
            Mean episode length, or zero when unavailable.
        """
        value = self.env_runners.get("episode_len_mean")
        if value is not None:
            return float(value)
        return float(self.episode_len_mean
                     ) if self.episode_len_mean is not None else 0.0

    def parse_episodes_total(self) -> int:
        """Return the lifetime episode count across RLlib API versions.

        Returns
        -------
        int
            Lifetime episode count.
        """
        value = self.env_runners.get("num_episodes_lifetime")
        if value is not None:
            return int(value)
        return int(
            self.episodes_total) if self.episodes_total is not None else 0

    def parse_environment_steps_total(self) -> int:
        """Return sampled environment steps across RLlib API versions.

        Returns
        -------
        int
            Lifetime sampled environment-step count.
        """
        value = self.env_runners.get("num_env_steps_sampled_lifetime")
        if value is not None:
            return int(value)
        return int(self.timesteps_total) if self.timesteps_total is not None else 0


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
        return cls(
            iteration=iteration,
            episode_reward_mean=parsed.parse_episode_return(),
            episode_len_mean=parsed.parse_episode_len(),
            episodes_total=parsed.parse_episodes_total(),
            elapsed_s=elapsed_s,
            environment_steps_total=parsed.parse_environment_steps_total(),
            extra={
                "training_iteration": parsed.training_iteration or iteration,
                "time_this_iter_s": parsed.time_this_iter_s,
            },
        )
