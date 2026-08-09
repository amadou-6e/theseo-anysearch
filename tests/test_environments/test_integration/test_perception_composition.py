"""Integration coverage for registry-driven voxel observations."""

from __future__ import annotations

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv


def test_composed_spatial_sensors_match_declared_space():
    env = VoxelEnv(
        {
            "max_steps": 5,
            "seed": 7,
            "perceptions": [
                {"name": "local_box", "parameters": {"radius": 1}},
                {"name": "radial", "parameters": {"max_len": 4}},
            ],
        }
    )
    observation, _ = env.reset()

    assert set(observation) == {"local_grid", "ray_hits", "ray_hit_types"}
    assert env.observation_space.contains(observation)
    assert observation["local_grid"].shape == (27,)


def test_perception_metadata_is_available_in_runtime_settings():
    from theseo_anysearch.settings import EnvConfig

    config = EnvConfig(
        agent_count=1,
        observation={
            "sensors": [
                {"name": "local_box", "parameters": {"radius": 1}},
                {"name": "radial", "parameters": {"max_len": 4}},
            ]
        },
    )
    runtime = config.to_runtime_dict()

    assert [item["name"] for item in runtime["perceptions"]] == ["local_box", "radial"]
    assert runtime["perception_metadata"][0]["estimated_compute"] == "O((2r+1)^3)"
