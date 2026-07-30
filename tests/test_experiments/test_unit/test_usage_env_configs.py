from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from theseo_anysearch.experiments.loader import load_experiment

LEGACY_FIELDS = {
    "stl_path",
    "stl_paths",
    "scale",
    "scale_range",
    "grid_size",
    "geometry_boxes",
    "geometry_pool",
    "obs_mode",
    "box_radius",
    "box_radii",
    "ray_max_len",
    "action_mode",
    "step_cost",
    "collision_cost",
    "goal_reward",
    "distance_shaping",
    "distance_reward_mode",
    "zone_reward_min",
    "zone_reward_max",
    "zone_reward_curve",
}


def environment_mappings(value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            if key == "env" and isinstance(nested, dict):
                yield nested
            yield from environment_mappings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from environment_mappings(nested)


def test_usage_experiments_do_not_use_flattened_environment_fields() -> None:
    repository = Path(__file__).resolve().parents[3]
    experiments = Path(repository, "usage", "experiments")
    paths = [
        path
        for path in experiments.rglob("*.yaml")
        if "runtime" not in path.parts
    ]
    mappings = []
    for path in paths:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        mappings.extend(environment_mappings(document))

    assert len(paths) == 31
    assert mappings
    for environment in mappings:
        assert LEGACY_FIELDS.isdisjoint(environment)


def test_representative_nested_experiments_load() -> None:
    repository = Path(__file__).resolve().parents[3]
    for parts in (
        ("usage", "experiments", "showcase", "quick_demo.yaml"),
        ("usage", "experiments", "heuristics", "dijkstra", "run.yaml"),
        ("usage", "experiments", "train", "ppo_maps_vector_zones.yaml"),
    ):
        config = load_experiment(Path(repository, *parts))
        dumped = config.env.model_dump(mode="python")
        assert {"geometry", "observation", "action", "rewards"} <= dumped.keys()
