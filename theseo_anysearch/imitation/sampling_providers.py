"""Named batch-sampling providers for imitation pretraining."""

from __future__ import annotations

from typing import Any, Callable, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from theseo_anysearch.imitation.dataset import DemonstrationDataset


class SamplingProviderError(ValueError):
    """Raised when a sampling provider name is unknown or misbehaves."""


class EpisodeSamplingContext(BaseModel):
    """Pure inputs handed to a batch-sampling provider for one epoch."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    dataset: DemonstrationDataset
    split: Literal["train", "validation"]
    batch_size: int
    seed: int
    parameters: dict[str, Any] = Field(default_factory=dict)


SamplingProvider = Callable[[EpisodeSamplingContext], list[np.ndarray]]


def _split_size(dataset: DemonstrationDataset, split: str) -> int:
    return len(dataset.train_actions) if split == "train" else len(dataset.validation_actions)


def _split_episode_ids(dataset: DemonstrationDataset, split: str) -> np.ndarray:
    return (
        dataset.train_episode_ids if split == "train" else dataset.validation_episode_ids
    )


def uniform_transition(context: EpisodeSamplingContext) -> list[np.ndarray]:
    """Shuffle all transitions in the split and slice into fixed-size batches."""
    if context.parameters:
        raise SamplingProviderError(
            "sampling provider 'uniform_transition' does not accept parameters"
        )
    count = _split_size(context.dataset, context.split)
    rng = np.random.default_rng(context.seed)
    indices = rng.permutation(count)
    return [
        indices[offset:offset + context.batch_size]
        for offset in range(0, len(indices), context.batch_size)
    ]


def uniform_episode(context: EpisodeSamplingContext) -> list[np.ndarray]:
    """Shuffle whole episodes and slice into batches of their transitions.

    Episodes are never split across batches, so a returned batch may exceed
    ``batch_size`` by up to one episode's transition count -- ``batch_size``
    is a soft target under this sampler, not a hard bound.
    """
    if context.parameters:
        raise SamplingProviderError(
            "sampling provider 'uniform_episode' does not accept parameters"
        )
    episode_ids = _split_episode_ids(context.dataset, context.split)
    unique_episodes = np.unique(episode_ids)
    rng = np.random.default_rng(context.seed)
    rng.shuffle(unique_episodes)
    batches: list[np.ndarray] = []
    current: list[int] = []
    for episode_id in unique_episodes:
        current.extend(np.flatnonzero(episode_ids == episode_id).tolist())
        if len(current) >= context.batch_size:
            batches.append(np.asarray(current, dtype=np.int64))
            current = []
    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    return batches


BUILT_IN_SAMPLING_PROVIDERS: dict[str, SamplingProvider] = {
    "uniform_transition": uniform_transition,
    "uniform_episode": uniform_episode,
}


def resolve_sampling_provider(
    name: str,
    *,
    python_provider: Any | None = None,
) -> SamplingProvider:
    """Resolve a sampling provider by name, built-in first, then Python."""
    built_in = BUILT_IN_SAMPLING_PROVIDERS.get(name)
    if built_in is not None:
        return built_in
    if python_provider is not None and getattr(python_provider, "name", None) == name:
        return python_provider.sample
    raise SamplingProviderError(f"unknown sampling provider: {name!r}")
