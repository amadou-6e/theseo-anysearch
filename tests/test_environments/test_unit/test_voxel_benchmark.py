"""Strict source parsing and independently replayed point-path diagnostics."""

from __future__ import annotations

import zipfile

import numpy as np
import networkx as nx
import pytest

from theseo_anysearch.environments.voxel_benchmark import (
    ScenarioQuery,
    load_voxel_map,
    parse_scenarios,
    plan_point_path,
    peak_process_memory_bytes,
    replay_point_path,
    run_diagnostic_slice,
)


def _map_zip(tmp_path, body: str, name: str = "plant01.3dmap"):
    path = tmp_path / "plant01.3dmap.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"plants/{name}", body)
    return path


def _scen(tmp_path, rows: str):
    path = tmp_path / "plant01.3dscen"
    path.write_text("version 2\nplant01.3dmap\n" + rows, encoding="ascii")
    return path


@pytest.mark.parametrize(
    "body,reason",
    [
        ("voxel 4 4 4\n1 2\n", "exactly three"),
        ("voxel 4 4 4\n1 2 3 4\n", "exactly three"),
        ("voxel 4 4 4\n4 0 0\n", "out-of-bounds"),
        ("voxel 0 4 4\n", "extent"),
        ("junk 4 4 4\n", "header"),
    ],
)
def test_strict_map_rejects_invalid_lines(tmp_path, body, reason):
    with pytest.raises(ValueError, match=reason):
        load_voxel_map(_map_zip(tmp_path, body))


def test_rev_voxel_is_vectorized_and_not_free_by_default(tmp_path):
    grid, member, repeated = load_voxel_map(
        _map_zip(tmp_path, "rev_voxel 4 3 2\n1 1 1\n1 1 1\n")
    )
    assert member == "plants/plant01.3dmap"
    assert grid.shape == (4, 3, 2)
    assert grid.sum() == 23
    assert grid[1, 1, 1] == 0
    assert repeated == 1


def test_scenario_accounts_for_every_source_row(tmp_path):
    grid = np.zeros((4, 4, 4), dtype=np.uint8)
    grid[2, 2, 2] = 1
    scen = _scen(
        tmp_path,
        "0 0 0 1 1 1 1.73 1 1\n"
        "0 0 0 9 1 1 1 1 1\n"
        "0 0 0 2 2 2 1 1 1\n"
        "malformed\n"
        "0 0 0 1 1 1 nan 1 1\n",
    )
    accepted, rejected = parse_scenarios(scen, "plant01.3dmap", grid)
    assert len(accepted) == 1
    assert accepted[0].source_line == 3
    assert accepted[0].opaque_values == (1.73, 1.0, 1.0)
    assert [item.reason for item in rejected] == [
        "endpoint out of bounds",
        "occupied endpoint",
        "expected nine columns",
        "non-finite opaque value",
    ]


def test_scenario_header_must_match_map(tmp_path):
    scen = _scen(tmp_path, "0 0 0 1 1 1 1 1 1\n")
    with pytest.raises(ValueError, match="does not match"):
        parse_scenarios(scen, "other.3dmap", np.zeros((4, 4, 4), dtype=np.uint8))


def test_networkx_point_path_and_independent_replay():
    grid = np.zeros((5, 5, 3), dtype=np.uint8)
    grid[2, :, 0] = 1
    query = ScenarioQuery(0, 3, (0, 2, 0), (4, 2, 0), (0, 0, 0))
    path, cost, expansions = plan_point_path(grid, query)
    assert path[0] == query.start and path[-1] == query.goal
    assert any(position[2] > 0 for position in path)
    assert expansions > 0
    assert cost == pytest.approx(replay_point_path(grid, query, path))
    with pytest.raises(ValueError, match="occupied"):
        replay_point_path(grid, query, [query.start, (2, 2, 0), query.goal])


def test_diagonal_cannot_cut_occupied_corner():
    grid = np.zeros((2, 2, 1), dtype=np.uint8)
    grid[1, 0, 0] = 1
    grid[0, 1, 0] = 1
    query = ScenarioQuery(0, 3, (0, 0, 0), (1, 1, 0), (0, 0, 0))
    with pytest.raises(nx.NetworkXNoPath):
        plan_point_path(grid, query)
    with pytest.raises(ValueError, match="corner"):
        replay_point_path(grid, query, [query.start, query.goal])


def test_end_to_end_compiles_world_and_records_noncomparable_run(tmp_path):
    map_zip = _map_zip(tmp_path, "voxel 6 6 3\n2 2 0\n")
    scen = _scen(tmp_path, "0 0 0 4 4 0 5 1 1\n2 2 0 1 1 0 1 1 1\n")
    report = run_diagnostic_slice(map_zip, scen, tmp_path / "output", query_limit=1)
    assert report["accepted_queries"] == 1
    assert len(report["rejected_queries"]) == 1
    assert report["evaluated_queries"][0]["status"] == "valid_path"
    assert report["world_pack_identity"]
    assert report["rights_status"] == "unreviewed"
    assert report["status"].startswith("technical_diagnostic")
    assert report["governing_spec_sha"] == "1dc8397ce7a8d4d9ac5def3e2ea0cdc472a2489b"
    assert report["peak_process_memory_bytes"] > 0
    assert report["evaluated_queries"][0]["task_identity_sha256"]


def test_process_memory_telemetry_is_positive():
    assert peak_process_memory_bytes() > 0
