from pathlib import Path

import pytest

from theseo_anysearch.worlds.compiler import BoxSource, compile_world
from theseo_anysearch.worlds.manifest import WorldExtent
from theseo_anysearch.worlds.residency import (
    WorldResidencySettings,
    resolve_worker_world,
    stage_compiled_world,
)
from theseo_anysearch.settings.environment.environment import EnvConfig


def test_node_staging_is_content_addressed_and_reused(tmp_path: Path) -> None:
    world = compile_world(
        [BoxSource(minimum=(0, 0, 0), maximum_inclusive=(1, 1, 1))],
        WorldExtent(x=4, y=4, z=4),
        tmp_path.joinpath("source"),
    )

    first = stage_compiled_world(world, tmp_path.joinpath("node"))
    second = stage_compiled_world(world, tmp_path.joinpath("node"))

    assert first.root == second.root
    assert first.manifest.identity_sha256 == world.manifest.identity_sha256
    assert first.root.joinpath("staging.json").is_file()
    assert all(path.is_file() for path in first.root.iterdir())


def test_worker_world_resolution_stages_shared_pack(tmp_path: Path) -> None:
    world = compile_world(
        [BoxSource(minimum=(0, 0, 0), maximum_inclusive=(0, 0, 0))],
        WorldExtent(x=4, y=4, z=4),
        tmp_path.joinpath("shared"),
    )

    resolved = resolve_worker_world(world.root, tmp_path.joinpath("worker"))

    assert resolved.root.parent == tmp_path.joinpath("worker").resolve()
    assert resolved.manifest.identity_sha256 == world.manifest.identity_sha256


def test_node_staging_rebuilds_a_corrupt_local_copy(tmp_path: Path) -> None:
    world = compile_world(
        [BoxSource(minimum=(0, 0, 0), maximum_inclusive=(1, 1, 1))],
        WorldExtent(x=4, y=4, z=4),
        tmp_path.joinpath("source"),
    )
    node_cache = tmp_path.joinpath("node")
    staged = stage_compiled_world(world, node_cache)
    staged.pack_path.write_bytes(b"corrupt")

    rebuilt = stage_compiled_world(world, node_cache)

    assert rebuilt.manifest.identity_sha256 == world.manifest.identity_sha256
    assert rebuilt.pack_path.read_bytes() == world.pack_path.read_bytes()


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"maximum_decoded_bytes": 0}, "maximum_decoded_bytes"),
        ({"prefetch_margin": -1}, "prefetch_margin"),
        ({"lock_timeout_seconds": 0}, "lock_timeout_seconds"),
    ],
)
def test_residency_settings_reject_invalid_limits(
    updates: dict[str, int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        WorldResidencySettings(**updates)


def test_nested_environment_settings_translate_residency_fields() -> None:
    config = EnvConfig.model_validate(
        {
            "geometry": {
                "compiled_world_path": "runtime/worlds/example",
                "node_cache": "runtime/worlds/node-cache",
                "maximum_decoded_bytes": 4096,
                "prefetch_margin": 5,
            }
        }
    ).to_runtime_dict()

    assert Path(config["compiled_world_path"]).parts[-3:] == (
        "runtime",
        "worlds",
        "example",
    )
    assert config["world_maximum_decoded_bytes"] == 4096
    assert config["world_prefetch_margin"] == 5
    assert Path(config["compiled_world_node_cache"]).parts[-3:] == (
        "runtime",
        "worlds",
        "node-cache",
    )


def test_native_environment_uses_prefetched_pack_without_hot_step_reads(
    tmp_path: Path,
) -> None:
    theseo_core = pytest.importorskip("theseo_core")
    world = compile_world(
        [BoxSource(minimum=(1, 1, 1), maximum_inclusive=(1, 1, 1))],
        WorldExtent(x=4, y=4, z=4),
        tmp_path.joinpath("source"),
    )
    env = theseo_core.PyVoxelEnv(
        max_steps=4,
        trail_mode=False,
        grid_size=4,
    )
    env.set_compiled_world(str(world.root), 1024)
    env.set_world_residency_radius(4)
    env.set_waypoints((2, 2, 2), (3, 3, 3), 1)
    env.prefetch_world_region((0, 0, 0), (4, 4, 4))
    reads_before = env.world_cache_metrics()["pack_reads"]

    env.reset(1)
    env.step(26)

    metrics = env.world_cache_metrics()
    assert env.world_occupied((1, 1, 1)) is True
    assert metrics["pack_reads"] == reads_before
    assert metrics["cache_hits"] > 0
