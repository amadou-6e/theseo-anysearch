"""Shared fixtures for environment integration tests."""

from __future__ import annotations

import pytest

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv


VOXEL_CONFIG = {
    "max_steps": 20,
    "seed": 42,
}


@pytest.fixture()
def env():
    """Return a default integration voxel environment."""

    return VoxelEnv(VOXEL_CONFIG)
