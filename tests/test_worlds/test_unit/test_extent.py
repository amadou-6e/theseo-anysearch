import math

import pytest

from theseo_anysearch.worlds.extent import (
    contains_task_coordinate,
    maximum_euclidean,
    maximum_manhattan,
    resolve_extent,
    resolve_task_extent,
    task_center,
)


def test_cubic_shorthand_retains_legacy_contract() -> None:
    assert resolve_extent({"grid_size": 8}) == (8, 8, 8)
    assert task_center((8, 8, 8)) == (4, 4, 4)
    assert maximum_manhattan((8, 8, 8)) == 21
    assert maximum_euclidean((8, 8, 8)) == pytest.approx(math.sqrt(147))


def test_non_cubic_extent_uses_one_based_per_axis_bounds() -> None:
    extent = resolve_extent({"grid_size": None, "extent": [5, 9, 3]})
    assert extent == (5, 9, 3)
    assert task_center(extent) == (3, 5, 2)
    assert contains_task_coordinate(extent, (5, 9, 3))
    assert not contains_task_coordinate(extent, (6, 9, 3))
    assert maximum_manhattan(extent) == 14


def test_conflicting_scalar_and_extent_are_rejected() -> None:
    with pytest.raises(ValueError, match="different world bounds"):
        resolve_extent({"grid_size": 8, "extent": [8, 9, 8]})


def test_live_task_extent_rejects_coordinate_truncation() -> None:
    with pytest.raises(ValueError, match="up to 65535"):
        resolve_task_extent({"grid_size": None, "extent": [65536, 2, 2]})
