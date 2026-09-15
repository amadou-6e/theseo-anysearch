import pytest

from theseo_anysearch.worlds.artifacts import publish_eager_geometry
from theseo_anysearch.worlds.manifest import WorldExtent
from theseo_anysearch.worlds.transformations import (
    SparseTransformedRead,
    generate_box_transform,
    measure_region_residency,
    transformed_artifact_metadata,
)


def test_fixed_input_produces_fixed_identity_and_boxes() -> None:
    args = ("a" * 64, WorldExtent(x=32, y=24, z=16))
    first = generate_box_transform(*args, seed=17, count=8)
    second = generate_box_transform(*args, seed=17, count=8)
    assert first == second


def test_bounded_sparse_read_matches_small_world_equivalent() -> None:
    base = {(2, 2, 2), (9, 9, 9)}
    transform = generate_box_transform(
        "b" * 64, WorldExtent(x=16, y=16, z=16), seed=4, count=3,
        minimum_size=(2, 2, 2), maximum_size=(2, 2, 2),
    )
    reader = SparseTransformedRead(
        lambda coordinate: coordinate in base,
        lambda minimum, maximum: (
            item for item in base if all(minimum[a] <= item[a] < maximum[a] for a in range(3))
        ),
        transform,
    )
    expected = set(base)
    for box in transform.boxes:
        expected.update(box.coordinates_in_region((1, 1, 1), (17, 17, 17)))
    assert set(reader.occupied_in_region((1, 1, 1), (17, 17, 17))) == expected
    assert all(reader.occupied(item) for item in expected)


def test_overlay_instances_are_reset_isolated() -> None:
    transform = generate_box_transform("c" * 64, WorldExtent(x=8, y=8, z=8), seed=2, count=1)
    first = SparseTransformedRead(lambda _: False, lambda _a, _b: (), transform)
    second = SparseTransformedRead(lambda _: False, lambda _a, _b: (), transform)
    first.occupied((1, 1, 1))
    assert first.point_queries == 1
    assert second.point_queries == 0


def test_cold_and_hot_region_reads_are_measured() -> None:
    transform = generate_box_transform("d" * 64, WorldExtent(x=8, y=8, z=8), seed=3, count=1)
    reader = SparseTransformedRead(lambda _: False, lambda _a, _b: (), transform)
    metrics = measure_region_residency(reader, (1, 1, 1), (9, 9, 9))
    assert metrics["occupied_voxels"] > 0
    assert reader.region_queries == 2


def test_transformed_metadata_invalidates_base_derivatives(tmp_path) -> None:
    base = publish_eager_geometry(
        ((1, 1, 1),),
        WorldExtent(x=8, y=8, z=8),
        tmp_path,
        transformations=({"type": "existing"},),
        validation={"valid": True},
        difficulty={"detour_ratio": 1.5},
    ).manifest
    transform = generate_box_transform(
        base.identity_sha256,
        base.extent,
        seed=5,
        count=1,
    )

    metadata = transformed_artifact_metadata(base, transform)

    assert metadata["derivatives_invalidated"] is True
    assert metadata["candidates"] is None
    assert metadata["validation"] == {}
    assert metadata["difficulty"] == {}
    assert metadata["overview"] is None
    assert metadata["transformations"][0] == {"type": "existing"}
    assert metadata["transformations"][1]["identity_sha256"] == transform.identity_sha256


def test_transformed_metadata_rejects_the_wrong_base(tmp_path) -> None:
    base = publish_eager_geometry(
        ((1, 1, 1),), WorldExtent(x=8, y=8, z=8), tmp_path
    ).manifest
    transform = generate_box_transform(
        "f" * 64, base.extent, seed=5, count=1
    )

    with pytest.raises(ValueError, match="base identity does not match"):
        transformed_artifact_metadata(base, transform)


@pytest.mark.parametrize(
    ("minimum_size", "maximum_size", "message"),
    [
        ((0, 1, 1), (2, 2, 2), "positive"),
        ((3, 1, 1), (2, 2, 2), "minimum box size"),
        ((9, 1, 1), (10, 2, 2), "world extent"),
    ],
)
def test_invalid_box_size_ranges_fail_actionably(
    minimum_size, maximum_size, message
) -> None:
    with pytest.raises(ValueError, match=message):
        generate_box_transform(
            "a" * 64,
            WorldExtent(x=8, y=8, z=8),
            seed=1,
            count=1,
            minimum_size=minimum_size,
            maximum_size=maximum_size,
        )
