"""Tests for built-in batch-sampling providers."""

import numpy as np
import pytest

from theseo_anysearch.imitation.dataset import DemonstrationDataset
from theseo_anysearch.imitation.models import DemonstrationManifest
from theseo_anysearch.imitation.sampling_providers import (
    EpisodeSamplingContext,
    SamplingProviderError,
    resolve_sampling_provider,
)


def _dataset() -> DemonstrationDataset:
    manifest = DemonstrationManifest(
        fingerprint="abc",
        generation_provider_name="astar",
        generation_provider_parameters={},
        requested_episodes=3,
        successful_episodes=3,
        accepted_episodes=3,
        attempted_episodes=3,
        training_episodes=3,
        validation_episodes=0,
        training_samples=6,
        validation_samples=0,
        observation_size=1,
        action_count=2,
        seeds=[1, 2, 3],
    )
    return DemonstrationDataset(
        train_observations=np.zeros((6, 1), dtype=np.float32),
        train_actions=np.zeros(6, dtype=np.int64),
        train_episode_ids=np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64),
        validation_observations=np.zeros((0, 1), dtype=np.float32),
        validation_actions=np.zeros(0, dtype=np.int64),
        validation_episode_ids=np.zeros(0, dtype=np.int64),
        manifest=manifest,
    )


def test_resolve_sampling_provider_rejects_unknown_name():
    with pytest.raises(SamplingProviderError, match="unknown sampling provider"):
        resolve_sampling_provider("not_a_real_provider")


def test_uniform_transition_covers_every_row_exactly_once():
    provider = resolve_sampling_provider("uniform_transition")
    batches = provider(
        EpisodeSamplingContext(dataset=_dataset(), split="train", batch_size=4, seed=1)
    )
    covered = np.sort(np.concatenate(batches))
    assert covered.tolist() == [0, 1, 2, 3, 4, 5]


def test_uniform_episode_never_splits_an_episode_across_batches():
    provider = resolve_sampling_provider("uniform_episode")
    dataset = _dataset()
    batches = provider(
        EpisodeSamplingContext(dataset=dataset, split="train", batch_size=2, seed=1)
    )
    for batch in batches:
        episode_ids_in_batch = set(dataset.train_episode_ids[batch].tolist())
        assert len(episode_ids_in_batch) == 1
    covered = np.sort(np.concatenate(batches))
    assert covered.tolist() == [0, 1, 2, 3, 4, 5]


def _dataset_with_oversized_episode() -> DemonstrationDataset:
    manifest = DemonstrationManifest(
        fingerprint="abc",
        generation_provider_name="astar",
        generation_provider_parameters={},
        requested_episodes=1,
        successful_episodes=1,
        accepted_episodes=1,
        attempted_episodes=1,
        training_episodes=1,
        validation_episodes=0,
        training_samples=5,
        validation_samples=0,
        observation_size=1,
        action_count=2,
        seeds=[1],
    )
    return DemonstrationDataset(
        train_observations=np.zeros((5, 1), dtype=np.float32),
        train_actions=np.zeros(5, dtype=np.int64),
        train_episode_ids=np.zeros(5, dtype=np.int64),
        validation_observations=np.zeros((0, 1), dtype=np.float32),
        validation_actions=np.zeros(0, dtype=np.int64),
        validation_episode_ids=np.zeros(0, dtype=np.int64),
        manifest=manifest,
    )


def test_uniform_episode_batch_may_legitimately_exceed_batch_size():
    provider = resolve_sampling_provider("uniform_episode")
    dataset = _dataset_with_oversized_episode()
    batches = provider(
        EpisodeSamplingContext(dataset=dataset, split="train", batch_size=3, seed=1)
    )
    assert len(batches) == 1
    assert sorted(batches[0].tolist()) == [0, 1, 2, 3, 4]


def test_uniform_transition_rejects_unexpected_parameters():
    provider = resolve_sampling_provider("uniform_transition")
    with pytest.raises(SamplingProviderError, match="does not accept parameters"):
        provider(
            EpisodeSamplingContext(
                dataset=_dataset(),
                split="train",
                batch_size=4,
                seed=1,
                parameters={"bogus_key": 1},
            )
        )


def test_uniform_episode_rejects_unexpected_parameters():
    provider = resolve_sampling_provider("uniform_episode")
    with pytest.raises(SamplingProviderError, match="does not accept parameters"):
        provider(
            EpisodeSamplingContext(
                dataset=_dataset(),
                split="train",
                batch_size=4,
                seed=1,
                parameters={"bogus_key": 1},
            )
        )
