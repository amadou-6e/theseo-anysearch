from theseo_anysearch.worlds.manifest import WorldExtent
from theseo_anysearch.worlds.transformations import (
    SparseTransformedRead,
    generate_box_transform,
    measure_region_residency,
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
