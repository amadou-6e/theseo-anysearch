"""Tests for sibling-`imitation.py` generation-provider discovery."""

import json

import pytest

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.experiments.custom_imitation import (
    CustomGenerationError,
    available_python_generation_names,
    discover_generation_source,
    load_generation_provider,
)
from theseo_anysearch.imitation.generation_providers import EpisodeGenerationContext


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


IMITATION_SOURCE = '''
def straight_line_generator(context):
    return {
        "observations": [context.observation],
        "actions": [0],
        "success": True,
        "seed": context.seed,
    }
'''


def test_discover_generation_source_finds_sibling_file(tmp_path):
    config_path = tmp_path.joinpath("experiment.yaml")
    config_path.write_text("imitation:\n  generation:\n    provider: straight_line_generator\n")
    tmp_path.joinpath("imitation.py").write_text(IMITATION_SOURCE, encoding="utf-8")

    source = discover_generation_source(config_path, "straight_line_generator")

    assert source == tmp_path.joinpath("imitation.py")


def test_discover_generation_source_returns_none_without_sibling_file(tmp_path):
    config_path = tmp_path.joinpath("experiment.yaml")
    config_path.write_text("imitation:\n  generation:\n    provider: straight_line_generator\n")

    assert discover_generation_source(config_path, "straight_line_generator") is None


def test_available_python_generation_names_probes_exports(tmp_path):
    source = tmp_path.joinpath("imitation.py")
    source.write_text(IMITATION_SOURCE, encoding="utf-8")

    assert available_python_generation_names(
        source, ("straight_line_generator", "not_defined")
    ) == ("straight_line_generator",)


def test_load_generation_provider_runs_the_python_function(tmp_path):
    source = tmp_path.joinpath("imitation.py")
    source.write_text(IMITATION_SOURCE, encoding="utf-8")
    record = load_generation_provider(source, "straight_line_generator")

    env = VoxelEnv(_tiny_env_config(tmp_path))
    observation, _ = env.reset(seed=7)
    episode = record.generate(
        EpisodeGenerationContext(env=env, observation=observation, seed=7, attempt=0)
    )
    env.close()

    assert episode.success is True
    assert episode.seed == 7


def test_load_generation_provider_rejects_wrong_arity(tmp_path):
    source = tmp_path.joinpath("imitation.py")
    source.write_text("def bad(context, extra):\n    return {}\n", encoding="utf-8")

    with pytest.raises(CustomGenerationError, match="exactly one argument"):
        load_generation_provider(source, "bad")
