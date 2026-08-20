"""Tests for built-in episode-generation providers."""

import json

import pytest

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.imitation.generation_providers import (
    BUILT_IN_GENERATION_PROVIDERS,
    EpisodeGenerationContext,
    GenerationProviderError,
    resolve_generation_provider,
)


def _tiny_env_config(tmp_path) -> dict:
    waypoints = tmp_path.joinpath("waypoints.json")
    waypoints.write_text(
        json.dumps({"start": [4, 4, 4], "goal": [4, 4, 6]}),
        encoding="utf-8",
    )
    return {
        "waypoints_file": str(waypoints),
        "grid_size": 8,
        "max_steps": 10,
        "agent_count": 1,
        "obs_mode": "radial",
        "ray_max_len": 10,
        "trail_mode": False,
    }


def test_resolve_generation_provider_returns_built_in():
    provider = resolve_generation_provider("astar")
    assert provider is BUILT_IN_GENERATION_PROVIDERS["astar"]


def test_resolve_generation_provider_rejects_unknown_name():
    with pytest.raises(GenerationProviderError, match="unknown generation provider"):
        resolve_generation_provider("not_a_real_provider")


def test_astar_provider_records_raw_observations_and_reaches_goal(tmp_path):
    env = VoxelEnv(_tiny_env_config(tmp_path))
    observation, _ = env.reset(seed=10)
    provider = resolve_generation_provider("astar")

    episode = provider(
        EpisodeGenerationContext(env=env, observation=observation, seed=10, attempt=0)
    )

    assert episode.success is True
    assert len(episode.observations) == len(episode.actions)
    assert episode.seed == 10
    env.close()


def test_weighted_astar_provider_rejects_non_positive_weight(tmp_path):
    env = VoxelEnv(_tiny_env_config(tmp_path))
    observation, _ = env.reset(seed=10)
    provider = resolve_generation_provider("weighted_astar")

    with pytest.raises(GenerationProviderError, match="parameters.weight"):
        provider(
            EpisodeGenerationContext(
                env=env,
                observation=observation,
                seed=10,
                attempt=0,
                parameters={"weight": -1.0},
            )
        )
    env.close()
