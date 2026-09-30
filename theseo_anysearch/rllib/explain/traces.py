"""Episode observation traces consumed by policy explainers."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from theseo_anysearch.rllib.explain.scoring import PolicyScorer


class ObservationTraceStep(BaseModel):
    """One pre-action observation and its transition metadata.

    Parameters
    ----------
    step : int
        Zero-based step index.
    observation : Mapping[str, np.ndarray]
        Pre-action observation dictionary.
    action : int
        Action taken from the observation.
    reward : float
        Reward received after the action.
    cursor_before : tuple[float, float, float]
        Cursor position before the action.
    cursor_after : tuple[float, float, float]
        Cursor position after the action.
    done : bool
        Whether the episode ended after this action.
    collision : bool | None, optional
        Explicit collision metadata if available.
    info : Mapping[str, object] | None, optional
        Extra transition metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    step: int
    observation: Mapping[str, np.ndarray]
    action: int
    reward: float
    cursor_before: tuple[float, float, float]
    cursor_after: tuple[float, float, float]
    done: bool
    collision: bool | None = None
    info: Mapping[str, object] | None = None

    def is_collision(self) -> bool:
        """Return whether this step is a collision."""

        if self.collision is not None:
            return self.collision
        return not self.done and self.cursor_before == self.cursor_after


class ObservationTrace:
    """Sequence of pre-action observations and transition metadata.

    Parameters
    ----------
    steps : Sequence[ObservationTraceStep]
        Trace steps in episode order.
    algorithm : str, default="mock"
        Algorithm name associated with the trace.
    """

    def __init__(self, steps: Sequence[ObservationTraceStep], algorithm: str = "mock") -> None:
        self._steps = tuple(steps)
        self.algorithm = algorithm

    def __len__(self) -> int:
        """Return trace length."""

        return len(self._steps)

    def __iter__(self):
        """Iterate over trace steps."""

        return iter(self._steps)

    def step(self, index: int) -> ObservationTraceStep:
        """Return one trace step."""

        return self._steps[index]

    def observations(self) -> list[Mapping[str, np.ndarray]]:
        """Return pre-action observations for every step."""

        return [step.observation for step in self._steps]


class PolicyEvaluationTraceCollector:
    """Collect an observation trace by running a policy in an environment.

    Parameters
    ----------
    env : object
        Gymnasium-like environment exposing ``reset`` and ``step``.
    scorer : PolicyScorer
        Policy scorer used to select deterministic actions.
    algorithm : str, default="dqn"
        Algorithm name to attach to the collected trace.
    """

    def __init__(self, env: object, scorer: PolicyScorer, algorithm: str = "dqn") -> None:
        self._env = env
        self._scorer = scorer
        self._algorithm = algorithm

    def collect(self, seed: int | None = None, max_steps: int | None = None) -> ObservationTrace:
        """Run one deterministic policy episode and return its trace.

        Parameters
        ----------
        seed : int | None, optional
            Environment reset seed.
        max_steps : int | None, optional
            Optional cap below the environment time limit.

        Returns
        -------
        ObservationTrace
            Collected pre-action observation trace.
        """

        observation, _ = self._env.reset(seed=seed)
        steps: list[ObservationTraceStep] = []
        step_index = 0
        while max_steps is None or step_index < max_steps:
            action = self._scorer.select_action(observation)
            cursor_before = self._cursor(observation)
            next_observation, reward, terminated, truncated, info = self._env.step(action)
            done = bool(terminated or truncated)
            cursor_after = self._cursor(next_observation)
            collision = (cursor_before == cursor_after) and not done
            steps.append(
                ObservationTraceStep(
                    step=step_index,
                    observation=self._copy_observation(observation),
                    action=action,
                    reward=float(reward),
                    cursor_before=cursor_before,
                    cursor_after=cursor_after,
                    done=done,
                    collision=collision,
                    info=info,
                )
            )
            step_index += 1
            if done:
                break
            observation = next_observation
        return ObservationTrace(steps, algorithm=self._algorithm)

    def _cursor(self, observation: Mapping[str, np.ndarray]) -> tuple[float, float, float]:
        """Return normalized cursor position from an observation."""

        cursor = np.asarray(observation["cursor_pos"], dtype=np.float32)
        return tuple(float(value) for value in cursor)

    def _copy_observation(self, observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return a detached copy of an observation dictionary."""

        return {
            name: np.asarray(value, dtype=np.float32).copy()
            for name, value in observation.items()
        }
