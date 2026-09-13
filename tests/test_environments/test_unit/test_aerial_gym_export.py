"""No-network Aerial Gym adapter tests using hand-written collision URDFs."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from theseo_anysearch.environments.aerial_gym_export import (
    BOUNDS_MIN,
    Box,
    Instance,
    _rotation,
    export_scene,
    parse_collision_boxes,
    rasterize_boxes,
    scene_instances,
)
from theseo_anysearch.environments.routing_manifests import (
    ConversionRecord,
    RoutingTaskRecord,
    RoutingWorldRecord,
    SourceRecord,
    read_sidecar,
)


def _urdf(size: tuple[float, float, float], *, origin: str = "0 0 0", rpy: str = "0 0 0") -> str:
    dims = " ".join(str(value) for value in size)
    return (
        '<robot name="synthetic"><link name="base_link">'
        '<visual><geometry><sphere radius="8"/></geometry></visual>'
        f'<collision><origin xyz="{origin}" rpy="{rpy}"/>'
        f'<geometry><box size="{dims}"/></geometry></collision>'
        '</link></robot>'
    )


@pytest.fixture
def synthetic_source(tmp_path):
    root = tmp_path / "source"
    for name in (
        "aerial_gym/config/env_config/env_with_lidar_nav_obstacles.py",
        "aerial_gym/config/asset_config/lidar_nav_env_config.py",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic config for adapter tests\n", encoding="utf-8")
    for instance in scene_instances(0, "altitude"):
        path = root / instance.urdf
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.name.endswith("_wall.urdf") and "objects" not in str(path):
            if path.name in ("front_wall.urdf", "back_wall.urdf"):
                size = (0.2, 20.0, 20.0)
            elif path.name in ("top_wall.urdf", "bottom_wall.urdf"):
                size = (20.0, 20.0, 0.2)
            else:
                size = (20.0, 0.2, 20.0)
        else:
            size = (0.1, 1.0, 1.0)
        path.write_text(_urdf(size), encoding="utf-8")
    return root


def test_collision_origin_and_rotation_are_composed_and_visual_ignored(tmp_path):
    path = tmp_path / "body.urdf"
    path.write_text(
        _urdf((2.0, 0.4, 0.5), origin="1 0 0", rpy="0 0 1.5707963267948966"),
        encoding="utf-8",
    )
    box, = parse_collision_boxes(
        path, Instance("body.urdf", (4.0, 5.0, 6.0), (0.0, 0.0, np.pi / 2))
    )
    assert box.center_m == pytest.approx((4.0, 6.0, 6.0))
    assert box.contains(np.array([[4.0, 6.0, 6.0], [4.0, 6.0, 7.0]])).tolist() == [True, False]
    assert np.allclose(box.rotation, _rotation((0, 0, np.pi)))


@pytest.mark.parametrize("geometry", ['<sphere radius="1"/>', '<mesh filename="missing.obj"/>'])
def test_unsupported_collision_geometry_is_rejected(tmp_path, geometry):
    path = tmp_path / "bad.urdf"
    path.write_text(
        f'<robot><link name="base"><collision><geometry>{geometry}'
        '</geometry></collision></link></robot>',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="collision boxes"):
        parse_collision_boxes(path, Instance("bad.urdf", (0.0, 0.0, 0.0)))


def test_rasterization_tracks_rotated_collision_interiors():
    box = Box((0.3, 2.0, 1.0), (0.0, 0.0, -1.0), _rotation((0.0, 0.0, np.pi / 4)))
    grid = rasterize_boxes((box,), 0.25)
    assert grid.dtype == np.uint8
    assert grid.shape == (40, 40, 24)
    centers = np.indices(grid.shape).reshape(3, -1).T
    positions = np.asarray(BOUNDS_MIN) + (centers + 0.5) * 0.25
    source_hit = box.contains(positions).reshape(grid.shape)
    assert np.any(source_hit)
    assert not np.any(source_hit & ~grid.astype(bool))
    assert not grid[0, 0, 0]


@pytest.mark.parametrize("layout", ["detour", "altitude"])
def test_export_is_hash_identical_and_tasks_have_required_topology(
    tmp_path, synthetic_source, layout
):
    first = tmp_path / "first"
    second = tmp_path / "second"
    a = export_scene(synthetic_source, first, revision="test-revision", seed=3, layout=layout)
    b = export_scene(synthetic_source, second, revision="test-revision", seed=3, layout=layout)
    assert a == b
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    assert a["route_cells_6_axis"] > a["straight_cells"]
    if layout == "altitude":
        assert a["same_altitude_route_cells"] is None
    source = read_sidecar(first / "source.json", SourceRecord)
    conversion = read_sidecar(first / "conversion.json", ConversionRecord)
    world = read_sidecar(first / "world.json", RoutingWorldRecord)
    task = read_sidecar(first / "task.json", RoutingTaskRecord)
    assert source.rights.allowed_uses == ()
    assert conversion.parameters["scene_instances_sha256"] == a["scene_instances_sha256"]
    assert task.provenance == "derived"
    assert task.world_identity_sha256 == world.identity_sha256
    assert np.load(first / "occupancy.npy", allow_pickle=False).shape == world.extent.as_tuple()
    assert len(json.loads((first / "scene-instances.json").read_text())) == a["collision_boxes"]


def test_source_file_change_changes_export_identity(tmp_path, synthetic_source):
    first = export_scene(
        synthetic_source, tmp_path / "first", revision="test-revision", seed=3,
        layout="detour",
    )
    config = synthetic_source / "aerial_gym/config/env_config/env_with_lidar_nav_obstacles.py"
    config.write_text("# modified synthetic config\n", encoding="utf-8")
    second = export_scene(
        synthetic_source, tmp_path / "second", revision="test-revision", seed=3,
        layout="detour",
    )
    assert first["source_content_identity_sha256"] != second["source_content_identity_sha256"]
    assert first["world_identity_sha256"] != second["world_identity_sha256"]


def test_git_checkout_revision_and_selected_file_cleanliness(
    tmp_path, synthetic_source, monkeypatch
):
    (synthetic_source / ".git").mkdir()
    state = {"head": "correct", "dirty": ""}

    def fake_run(args, **_kwargs):
        return SimpleNamespace(stdout=state["head"] if "rev-parse" in args else state["dirty"])

    monkeypatch.setattr("theseo_anysearch.environments.aerial_gym_export.subprocess.run", fake_run)
    with pytest.raises(ValueError, match="revision differs"):
        export_scene(
            synthetic_source, tmp_path / "bad-head", revision="wrong", seed=0,
            layout="detour",
        )
    assert not (tmp_path / "bad-head").exists()
    state["dirty"] = " M resources/models/environment_assets/walls/top_wall.urdf"
    with pytest.raises(ValueError, match="selected source files are modified"):
        export_scene(
            synthetic_source, tmp_path / "dirty", revision="correct", seed=0,
            layout="detour",
        )
    assert not (tmp_path / "dirty").exists()


def test_invalid_layout_and_existing_output_are_rejected(tmp_path, synthetic_source):
    with pytest.raises(ValueError, match="layout"):
        scene_instances(0, "unknown")
    with pytest.raises(ValueError, match="seed"):
        scene_instances(-1, "detour")
    output = tmp_path / "result"
    output.mkdir()
    with pytest.raises(FileExistsError):
        export_scene(synthetic_source, output, revision="test-revision", seed=0, layout="detour")
