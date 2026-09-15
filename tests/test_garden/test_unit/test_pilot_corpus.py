"""Tests for deterministic pilot observation materialization."""
from __future__ import annotations

import numpy as np

from theseo_anysearch.garden.pilots.corpus import (
    make_pilot_observation,
    proper_cube_rotation,
)
from theseo_anysearch.garden.splits import GeometryDescriptor


def _descriptor(family: str = "topology", band: str = "medium") -> GeometryDescriptor:
    return GeometryDescriptor(
        geometry_id=f"fixture-{family}-{band}",
        family=family,
        occupancy_band=band,
        source="unit-test",
    )


def test_observations_are_reproducible_and_radius_specific() -> None:
    first = make_pilot_observation(_descriptor(), 7, radius=8)
    second = make_pilot_observation(_descriptor(), 7, radius=8)
    wider = make_pilot_observation(_descriptor(), 7, radius=16)
    assert first.identity_sha256 == second.identity_sha256
    assert np.array_equal(first.occupancy, second.occupancy)
    assert first.occupancy.shape == (17, 17, 17)
    assert wider.occupancy.shape == (33, 33, 33)
    assert first.identity_sha256 != wider.identity_sha256


def test_all_geometry_families_and_density_bands_are_nontrivial() -> None:
    means: dict[str, list[float]] = {band: [] for band in ("low", "medium", "high")}
    for family in ("open", "thin_obstacle", "topology", "imported"):
        for band in means:
            observation = make_pilot_observation(
                _descriptor(family, band), 1, radius=16
            )
            assert observation.occupancy.any()
            assert not observation.occupancy.all()
            means[band].append(float(observation.occupancy.mean()))
    assert np.mean(means["low"]) < np.mean(means["medium"])
    assert np.mean(means["medium"]) < np.mean(means["high"])


def test_unknown_boundary_is_removed_from_occupancy() -> None:
    observation = make_pilot_observation(_descriptor(), 0, radius=8)
    assert observation.unknown_mask.any()
    assert not np.any(observation.occupancy & observation.unknown_mask)


def test_cube_rotation_enumerates_24_unique_proper_orientations() -> None:
    volume = np.arange(27).reshape(3, 3, 3)
    rotations = [proper_cube_rotation(volume, index) for index in range(24)]
    assert len({rotation.tobytes() for rotation in rotations}) == 24
    assert all(
        np.array_equal(np.sort(rotation, axis=None), np.arange(27))
        for rotation in rotations
    )
