"""Step selection strategies for explanation traces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from theseo_anysearch.rllib.explain.traces import ObservationTrace


class StepSelector(ABC):
    """Base class for trace step selectors."""

    @abstractmethod
    def select(self, trace: ObservationTrace, max_steps: int | None = None) -> list[int]:
        """Return selected step indices."""


class CollisionStepSelector(StepSelector):
    """Select trace steps that are collisions."""

    def select(self, trace: ObservationTrace, max_steps: int | None = None) -> list[int]:
        """Return collision step indices."""

        indices = [step.step for step in trace if step.is_collision()]
        return indices[:max_steps] if max_steps is not None else indices


class ExplicitStepSelector(StepSelector):
    """Select explicit step indices provided by the caller.

    Parameters
    ----------
    indices : tuple[int, ...]
        Requested trace step indices.
    """

    def __init__(self, indices: tuple[int, ...]) -> None:
        self._indices = indices

    def select(self, trace: ObservationTrace, max_steps: int | None = None) -> list[int]:
        """Return validated explicit step indices."""

        trace_len = len(trace)
        invalid = [index for index in self._indices if index < 0 or index >= trace_len]
        if invalid:
            raise ValueError(f"explicit explanation steps are outside trace length {trace_len}: {invalid}")
        indices = list(self._indices)
        return indices[:max_steps] if max_steps is not None else indices
