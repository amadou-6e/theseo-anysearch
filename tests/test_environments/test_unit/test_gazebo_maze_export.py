"""Offline tests for SDF include resolution and finite-body maze validity."""

import io
import tarfile

import numpy as np
import pytest

from theseo_anysearch.environments.gazebo_maze_export import (
    Collision,
    _clearance_mask,
    _route,
    _rotation,
    collision_parity_census,
    over_wall_census,
    rasterize,
    read_source_collisions,
    replay_route,
)


def _archive(path, members):
    with tarfile.open(path, "w:xz") as archive:
        for name, text in members.items():
            payload = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def _model(name, body, *, filename="model.sdf"):
    return {
        f"{name}/model.config": f"<model><sdf version='1.5'>{filename}</sdf></model>",
        f"{name}/{filename}": f"<sdf version='1.5'>{body}</sdf>",
    }


def _fixture_archives(tmp_path, *, missing=False):
    maze = {
        "3d_maze/easy_maze.world": (
            "<sdf version='1.5'><world name='default'>"
            "<include><uri>model://easy_maze_3d</uri><pose>1 2 0 0 0 1.5707963267948966</pose></include>"
            "<include><uri>model://sun_2</uri></include>"
            "<include><uri>model://home</uri><pose>5 6 0 0 0 0</pose></include>"
            "</world></sdf>"
        ),
    }
    maze.update(_model("3d_maze/easy_maze_3d", (
        "<model name='easy_maze_3d'><include><name>w1</name><uri>model://maze_wall</uri>"
        "<pose>2 0 0 0 0 0</pose></include></model>"
    )))
    if not missing:
        maze.update(_model("3d_maze/maze_wall", (
            "<model name='maze_wall'><link name='wall'><pose>0 0 1 0 0 0</pose>"
            "<collision name='solid'><pose>1 0 0 0 0 0</pose><geometry><box><size>2 0.8 2</size>"
            "</box></geometry></collision><visual><geometry><sphere><radius>99</radius>"
            "</sphere></geometry></visual></link></model>"
        )))
    common = {}
    common.update(_model("common_models/home", (
        "<model name='home'><link name='base'><collision name='pad'><geometry>"
        "<cylinder><radius>1</radius><length>0.08</length></cylinder>"
        "</geometry></collision></link></model>"
    ), filename="home.sdf"))
    common.update(_model("common_models/sun_2", "<light name='sun_2' type='directional'/>") )
    _archive(tmp_path / "3d_maze.tar.xz", maze)
    _archive(tmp_path / "common_models.tar.xz", common)


def test_nested_sdf_poses_and_collision_geometry_only(tmp_path):
    _fixture_archives(tmp_path)
    collisions, used = read_source_collisions(tmp_path, verify_hashes=False)
    assert len(collisions) == 2
    wall, home = collisions
    assert wall.name == "easy_maze_3d/w1/wall/solid"
    assert wall.center_m == pytest.approx((1, 5, 1))
    assert wall.contains(np.array([[1, 5, 1], [1, 5, 5]])).tolist() == [True, False]
    assert home.kind == "cylinder"
    assert home.center_m == pytest.approx((5, 6, 0))
    assert any("easy_maze.world" in name for name in used)
    assert any("home.sdf" in name for name in used)


def test_missing_include_and_unpinned_archive_fail_closed(tmp_path):
    _fixture_archives(tmp_path, missing=True)
    with pytest.raises(ValueError, match="missing SDF dependency"):
        read_source_collisions(tmp_path, verify_hashes=False)
    with pytest.raises(ValueError, match="source archive hash differs"):
        read_source_collisions(tmp_path)


def test_rotated_box_and_cylinder_conservatively_rasterize():
    collisions = (
        Collision("wall", "box", (0, 0, 4), _rotation((0, 0, np.pi / 4)), size_m=(6, 0.8, 8)),
        Collision("home", "cylinder", (3, 3, 0.05), np.eye(3), radius_m=1, length_m=0.08),
    )
    occupied = rasterize(collisions)
    assert occupied.dtype == np.uint8
    assert occupied.shape == (184, 184, 18)
    strata = collision_parity_census(occupied, collisions, voxel_m=0.5)
    assert set(strata) == {"random_centers", "interiors", "surfaces", "openings", "borders", "diagonals"}
    assert all(row["source_hit_voxel_free"] == 0 for row in strata.values())


def test_roof_closes_over_wall_and_planar_control_remains():
    wall = Collision("wall", "box", (0, 0, 4), np.eye(3), size_m=(10, 0.8, 8))
    roof = Collision("roof", "box", (0, 0, 8.5), np.eye(3), size_m=(92, 92, 1))
    occupied = rasterize((wall, roof))
    passable, _ = _clearance_mask(occupied, 0.25, 0.5)
    assert over_wall_census(passable, (wall,), voxel_m=0.5)["body_valid_over_wall_cells"] == 0
    with pytest.raises(ValueError, match="vertical bypass"):
        over_wall_census(np.ones_like(passable, dtype=bool), (wall,), voxel_m=0.5)
    # A route on one side of the barrier is available at fixed altitude.
    start, goal = (78, 82, 2), (82, 82, 2)
    route = _route(~passable, start, goal, planar=True)
    assert route is not None
    replay_route(route, (wall, roof), voxel_m=0.5, radius_m=0.25)


def test_replay_rejects_body_collision_and_diagonal_hop():
    wall = Collision("wall", "box", (0, 0, 1.25), np.eye(3), size_m=(0.2, 4, 2))
    with pytest.raises(ValueError, match="collides"):
        replay_route(((91, 92, 2), (92, 92, 2)), (wall,), voxel_m=0.5, radius_m=0.25)
    with pytest.raises(ValueError, match="non-axis-adjacent"):
        replay_route(((91, 92, 2), (92, 93, 3)), (), voxel_m=0.5, radius_m=0.25)
