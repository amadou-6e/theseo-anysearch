"""Independent continuous replay against occupied voxel cubes."""

import numpy as np
import pytest

from theseo_anysearch.environments.voxel_route_replay import (
    axis_segment_clear,
    replay_six_axis_route,
    shortest_six_axis_route,
)


def test_axis_segment_catches_blocked_corner_and_bounds():
    truth = np.zeros((9, 9, 9), dtype=np.uint8)
    truth[4, 4, 4] = 1
    assert not axis_segment_clear(truth, (2, 4, 4), (6, 4, 4), body_radius_voxels=0)
    assert axis_segment_clear(truth, (2, 5, 4), (6, 5, 4), body_radius_voxels=0)
    assert not axis_segment_clear(truth, (2, 5, 4), (6, 5, 4), body_radius_voxels=0.5)
    assert axis_segment_clear(truth, (2, 5, 5), (6, 5, 5), body_radius_voxels=0.7)
    assert not axis_segment_clear(truth, (2, 5, 5), (6, 5, 5), body_radius_voxels=2**-0.5)
    assert not axis_segment_clear(truth, (0, 2, 2), (2, 2, 2), body_radius_voxels=0.5)
    assert not axis_segment_clear(truth, (2, 2, 2), (10, 2, 2), body_radius_voxels=0)
    with pytest.raises(ValueError, match="axis-aligned"):
        axis_segment_clear(truth, (2, 2, 2), (4, 4, 2), body_radius_voxels=0)


def test_shortest_route_is_deterministic_and_replayed_independently():
    truth = np.zeros((9, 9, 9), dtype=np.uint8)
    truth[4, 4, 4] = 1
    passable = truth == 0
    first = shortest_six_axis_route(passable, (2, 4, 4), (6, 4, 4))
    assert first == shortest_six_axis_route(passable, (2, 4, 4), (6, 4, 4))
    assert first is not None and len(first) > 5
    replay_six_axis_route(truth, first, body_radius_m=0, meters_per_voxel=1)
    with pytest.raises(ValueError, match="collides"):
        replay_six_axis_route(
            truth, ((2, 4, 4), (3, 4, 4), (4, 4, 4)),
            body_radius_m=0, meters_per_voxel=1,
        )
    with pytest.raises(ValueError, match="non-six-axis"):
        replay_six_axis_route(
            truth, ((2, 4, 4), (3, 5, 4)), body_radius_m=0, meters_per_voxel=1,
        )


def test_replay_rejects_tampered_or_nonbinary_truth():
    truth = np.zeros((5, 5, 5), dtype=np.uint8)
    route = ((1, 1, 1), (2, 1, 1))
    replay_six_axis_route(truth, route, body_radius_m=0.1, meters_per_voxel=1)
    truth[2, 1, 1] = 1
    with pytest.raises(ValueError, match="collides"):
        replay_six_axis_route(truth, route, body_radius_m=0.1, meters_per_voxel=1)
    truth[2, 1, 1] = 2
    with pytest.raises(ValueError, match="binary"):
        replay_six_axis_route(truth, route, body_radius_m=0.1, meters_per_voxel=1)
