"""Exact sphere sweeps against occupied voxel cubes and solid bounds."""

import numpy as np
import pytest

from theseo_anysearch.environments.shape_collision import (
    AxisHeading,
    Sphere,
    swept_voxel_segment_clear,
)


@pytest.mark.parametrize("axis", range(3))
@pytest.mark.parametrize("direction", (-1, 1))
def test_sphere_sweep_covers_all_six_travel_directions(axis: int, direction: int) -> None:
    truth = np.zeros((9, 9, 9), dtype=np.uint8)
    truth[4, 4, 4] = 1
    start = [4, 4, 4]
    end = [4, 4, 4]
    start[axis] -= 2 * direction
    end[axis] += 2 * direction
    assert not swept_voxel_segment_clear(truth, tuple(start), tuple(end), shape=Sphere(0))

    clear_start = start.copy()
    clear_end = end.copy()
    transverse = (axis + 1) % 3
    clear_start[transverse] += 1
    clear_end[transverse] += 1
    segment = (tuple(clear_start), tuple(clear_end))
    assert swept_voxel_segment_clear(truth, *segment, shape=Sphere(0.49))
    assert not swept_voxel_segment_clear(truth, *segment, shape=Sphere(0.5))


def test_sphere_heading_is_explicit_and_invariant() -> None:
    truth = np.zeros((9, 9, 9), dtype=np.uint8)
    truth[4, 4, 4] = 1
    segment = ((2, 5, 5), (6, 5, 5))
    for heading in AxisHeading:
        assert swept_voxel_segment_clear(
            truth, *segment, shape=Sphere(0.7), heading=heading
        )
        assert not swept_voxel_segment_clear(
            truth, *segment, shape=Sphere(2**-0.5), heading=heading
        )


def test_sphere_sweep_treats_world_bounds_as_solid() -> None:
    truth = np.zeros((7, 7, 7), dtype=np.uint8)
    assert swept_voxel_segment_clear(truth, (0, 2, 2), (1, 2, 2), shape=Sphere(0))
    assert not swept_voxel_segment_clear(truth, (0, 2, 2), (1, 2, 2), shape=Sphere(0.5))
    assert not swept_voxel_segment_clear(truth, (5, 2, 2), (7, 2, 2), shape=Sphere(0))


def test_unsupported_shapes_and_invalid_inputs_fail_explicitly() -> None:
    truth = np.zeros((7, 7, 7), dtype=np.uint8)
    segment = ((2, 2, 2), (3, 2, 2))
    with pytest.raises(TypeError, match="only Sphere"):
        swept_voxel_segment_clear(truth, *segment, shape=object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="heading"):
        swept_voxel_segment_clear(truth, *segment, shape=Sphere(0), heading="+x")  # type: ignore[arg-type]
    for radius in (-1, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="sphere radius"):
            Sphere(radius)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="axis-aligned"):
        swept_voxel_segment_clear(truth, (2, 2, 2), (3, 3, 2), shape=Sphere(0))
    truth[3, 2, 2] = 2
    with pytest.raises(ValueError, match="binary"):
        swept_voxel_segment_clear(truth, *segment, shape=Sphere(0))
