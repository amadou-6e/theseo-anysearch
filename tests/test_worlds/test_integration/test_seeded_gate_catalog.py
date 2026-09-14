"""End-to-end validity checks for seed-selected staggered gate worlds."""

import json
from pathlib import Path

import pytest

from theseo_anysearch.environments.gymnasium.voxel_env import VoxelEnv
from theseo_anysearch.experiments.trajectory import _VoxelEpisodeState
from theseo_anysearch.imitation.dataset import collect_demonstrations
from theseo_anysearch.imitation.models import ImitationConfig
from theseo_anysearch.rllib.trainer.waypoint_curriculum import WaypointCurriculum
from theseo_anysearch.settings.environment import EnvConfig
from theseo_anysearch.settings import load_experiment
from theseo_anysearch.worlds import world_contract
from theseo_anysearch.worlds.seeded_catalog import load_catalog
from usage.experiments.train.large_world_obstacle_pilot.curriculum import (
    MAX_EPISODE_STEPS, STAGE_LENGTHS, gate_curriculum_settings,
)
from usage.experiments.train.large_world_obstacle_pilot.seeded_portals.generate import (
    build_catalog, staggered_centers,
)
from usage.experiments.train.large_world_obstacle_pilot.seeded_portals.preflight import (
    preflight,
)


@pytest.fixture(scope="module")
def catalog_path(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("seeded-gates")
    result = build_catalog(root, variant_count=2)
    return Path(result["path"])


def _env_config(catalog_path: Path) -> dict:
    return EnvConfig.model_validate({
        "agent_count": 1, "max_steps": MAX_EPISODE_STEPS, "trail_mode": False,
        "geometry": {"extent": [4096, 2048, 512],
                     "compiled_world_catalog_path": str(catalog_path)},
        "observation": {"mode": "box", "box_radius": 1},
        "action": {"mode": "discrete_18"},
        "waypoint_curriculum": gate_curriculum_settings(),
    }).to_runtime_dict()


def test_staggered_centers_are_deterministic_and_alternate() -> None:
    first = staggered_centers(437_000)
    assert first == staggered_centers(437_000)
    assert first != staggered_centers(437_001)
    assert all(96 <= abs(y - 1024) <= 144 and z == 256 for y, z in first)
    assert all((first[index][0] - 1024) * (first[index + 1][0] - 1024) < 0
               for index in range(5))


def test_candidate_training_config_has_distinct_world_and_run_identity() -> None:
    candidate = load_experiment(Path(
        "usage/experiments/train/large_world_obstacle_pilot/"
        "seeded_portals/experiment.yaml"
    ))
    assert candidate.experiment.name == "gate-seeded-portals-100"
    assert candidate.training.iterations == 100
    assert candidate.imitation.pretraining.epochs == 20
    assert candidate.imitation.generation.episodes == 128
    assert candidate.env.geometry.compiled_world_catalog_path is not None
    assert candidate.env.geometry.compiled_world_path is None


def test_catalog_contract_does_not_change_legacy_world_fingerprints() -> None:
    assert "catalog_identity_sha256" not in world_contract({
        "extent": [4096, 2048, 512],
        "world_identity_sha256": "legacy-world",
    })


def test_catalog_integrity_seed_mapping_and_runtime_routes(catalog_path: Path) -> None:
    catalog = load_catalog(catalog_path)
    assert catalog.for_seed(0).identity_sha256 != catalog.for_seed(1).identity_sha256
    assert catalog.for_seed(2).identity_sha256 == catalog.for_seed(0).identity_sha256
    assert [
        sum(1 for _ in route.waypoints) for route in
        (catalog.route_for_stage(stage, 0) for stage in range(12))
    ][-1] == 8
    config = _env_config(catalog_path)
    assert world_contract(config)["catalog_identity_sha256"] == catalog.identity_sha256
    env = VoxelEnv(config)
    try:
        env.set_waypoint_curriculum([{"seeded_catalog_stage": 11}], [1.0])
        first_info = env.reset(seed=0)[1]
        assert first_info["world_identity_sha256"] == catalog.for_seed(0).identity_sha256
        assert env._config["waypoint_route"]["waypoints"][0][1] == (
            catalog.for_seed(0).portal_centers[0][0]
        )
        second_info = env.reset(seed=1)[1]
        assert second_info["world_identity_sha256"] == catalog.for_seed(1).identity_sha256
        assert env._config["waypoint_route"]["waypoints"][0][1] == (
            catalog.for_seed(1).portal_centers[0][0]
        )
        replay_info = env.reset(seed=0)[1]
        assert replay_info["world_identity_sha256"] == first_info["world_identity_sha256"]
        assert replay_info["world_layout_seed"] == first_info["world_layout_seed"]
        assert env._config["waypoint_route"]["waypoints"][0][1] == (
            catalog.for_seed(0).portal_centers[0][0]
        )
    finally:
        env.close()


def test_evaluation_route_seed_matches_reset_world(catalog_path: Path) -> None:
    settings = EnvConfig.model_validate({
        "agent_count": 1, "max_steps": MAX_EPISODE_STEPS, "trail_mode": False,
        "geometry": {"extent": [4096, 2048, 512],
                     "compiled_world_catalog_path": str(catalog_path)},
        "observation": {"mode": "box", "box_radius": 1},
        "action": {"mode": "discrete_18"},
        "waypoint_curriculum": gate_curriculum_settings(),
    })
    runtime = settings.to_runtime_dict()
    curriculum = WaypointCurriculum(settings.waypoint_curriculum, runtime)
    catalog = load_catalog(catalog_path)
    for seed in (142, 143):
        route = curriculum.route_for_stage(runtime, 11, seed=seed)
        assert route == catalog.route_for_stage(
            11, seed, variation_radius=2, action_mode="discrete_18"
        )
        route_env = dict(runtime)
        route_env["waypoint_route"] = route.model_dump(mode="python")
        route_env["waypoint_curriculum"] = {"enabled": False}
        env = VoxelEnv(route_env)
        try:
            info = env.reset(seed=seed)[1]
            assert info["world_identity_sha256"] == catalog.for_seed(seed).identity_sha256
            assert info["world_layout_seed"] == seed
        finally:
            env.close()


def test_astar_crosses_all_seeded_gates_at_exact_lengths(catalog_path: Path) -> None:
    report = preflight(catalog_path, [0, 1])
    assert report["checked_episodes"] == 8
    final = [result for result in report["results"] if result["stage"] == 11]
    assert all(result["actions"] == STAGE_LENGTHS[-1] for result in final)
    assert all(result["lateral_actions"] >= 600 for result in final)
    assert all(result["straight_x_fraction"] < 0.8 for result in final)
    assert final[0]["wall_crossings"] != final[1]["wall_crossings"]


def test_replay_episode_references_the_selected_world(catalog_path: Path) -> None:
    config = _env_config(catalog_path)
    catalog = load_catalog(catalog_path)
    for seed in (0, 1):
        episode = _VoxelEpisodeState.create(config, seed=seed)
        try:
            assert episode.world is not None
            assert episode.world.identity_sha256 == catalog.for_seed(seed).identity_sha256
            assert episode.world.source_root == catalog.for_seed(seed).root
        finally:
            episode.close()


def test_collector_records_seed_to_world_identity(catalog_path: Path) -> None:
    config = _env_config(catalog_path)
    dataset = collect_demonstrations(
        config,
        ImitationConfig.model_validate({
            "enabled": True,
            "generation": {"provider": "astar", "episodes": 12,
                           "max_attempts": 24, "require_success": True},
            "collection": {"seed_start": 1000, "validation_fraction": 0.1,
                           "curriculum_stages": "all"},
        }),
    )
    assert dataset.manifest.accepted_episodes == 12
    assert dataset.manifest.stage_episode_counts == [1] * 12
    assert dataset.manifest.world_catalog_sha256 == load_catalog(catalog_path).identity_sha256
    assert dataset.manifest.episode_world_identities == [
        load_catalog(catalog_path).for_seed(seed).identity_sha256
        for seed in dataset.manifest.seeds
    ]
    assert (dataset.manifest.training_samples + dataset.manifest.validation_samples
            == sum(STAGE_LENGTHS))


def test_catalog_checksum_rejects_modified_layout(catalog_path: Path, tmp_path: Path) -> None:
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    payload["variants"][0]["portal_centers"][0][0] += 1
    bad = tmp_path / "catalog.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_catalog(bad)
