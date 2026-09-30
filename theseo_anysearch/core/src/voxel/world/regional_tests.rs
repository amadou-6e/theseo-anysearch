use super::*;

fn extent() -> WorldExtent {
    WorldExtent { x: 9, y: 5, z: 3 }
}

fn region(minimum: StorageCoord, maximum_exclusive: StorageCoord) -> BoundedRegion {
    BoundedRegion::new(minimum, maximum_exclusive, extent()).unwrap()
}

fn populate<W: WorldMutation>(world: &mut W) {
    for (coordinate, kind) in [
        (StorageCoord { x: 0, y: 0, z: 0 }, 1),
        (StorageCoord { x: 3, y: 1, z: 1 }, 2),
        (StorageCoord { x: 4, y: 1, z: 1 }, 3),
        (StorageCoord { x: 8, y: 4, z: 2 }, 4),
    ] {
        world
            .set_block_value(
                coordinate,
                Block {
                    kind,
                    ..Block::default()
                },
            )
            .unwrap();
    }
}

#[test]
fn hashmap_and_chunked_queries_match_across_boundaries() {
    let mut oracle = HashMapWorld::new(extent());
    let mut chunked = ChunkedWorld::new(extent(), WorldExtent { x: 4, y: 2, z: 2 }).unwrap();
    populate(&mut oracle);
    populate(&mut chunked);

    for coordinate in [
        StorageCoord { x: 0, y: 0, z: 0 },
        StorageCoord { x: 3, y: 1, z: 1 },
        StorageCoord { x: 4, y: 1, z: 1 },
        StorageCoord { x: 8, y: 4, z: 2 },
    ] {
        assert_eq!(
            oracle.get_block_value(coordinate),
            chunked.get_block_value(coordinate)
        );
    }
    let bounded = region(
        StorageCoord { x: 3, y: 0, z: 0 },
        StorageCoord { x: 9, y: 5, z: 3 },
    );
    assert_eq!(
        oracle.blocks_in_region(bounded).unwrap(),
        chunked.blocks_in_region(bounded).unwrap()
    );
    assert_eq!(oracle.block_count(), chunked.block_count());
}

#[test]
fn empty_uniform_and_dense_logical_chunks_have_exact_counts() {
    let extent = WorldExtent { x: 4, y: 2, z: 2 };
    let chunk_shape = WorldExtent { x: 2, y: 2, z: 2 };
    let mut world = ChunkedWorld::new(extent, chunk_shape).unwrap();
    assert_eq!(world.block_count(), 0);
    assert_eq!(world.resident_chunk_count(), 0);

    for z in 0..2 {
        for y in 0..2 {
            for x in 0..2 {
                world
                    .set_block_value(StorageCoord { x, y, z }, Block::default())
                    .unwrap();
            }
        }
    }
    assert_eq!(world.block_count(), 8);
    assert_eq!(world.resident_chunk_count(), 1);

    for z in 0..2 {
        for y in 0..2 {
            for x in 2..4 {
                world
                    .set_block_value(
                        StorageCoord { x, y, z },
                        Block {
                            kind: (x + y + z) as u8,
                            ..Block::default()
                        },
                    )
                    .unwrap();
            }
        }
    }
    assert_eq!(world.block_count(), 16);
    assert_eq!(world.resident_chunk_count(), 2);
}

#[test]
fn bounded_region_validation_is_per_axis() {
    let valid = region(
        StorageCoord { x: 8, y: 4, z: 2 },
        StorageCoord { x: 9, y: 5, z: 3 },
    );
    assert!(valid.contains(StorageCoord { x: 8, y: 4, z: 2 }));
    assert!(matches!(
        BoundedRegion::new(
            StorageCoord { x: 0, y: 0, z: 0 },
            StorageCoord { x: 9, y: 6, z: 3 },
            extent()
        ),
        Err(WorldAccessError::InvalidRegion { .. })
    ));
}

#[test]
fn ray_queries_match_and_stop_at_world_edge() {
    let mut oracle = HashMapWorld::new(extent());
    let mut chunked = ChunkedWorld::new(extent(), WorldExtent { x: 4, y: 2, z: 2 }).unwrap();
    let target = StorageCoord { x: 8, y: 4, z: 2 };
    oracle.set_block_value(target, Block::default()).unwrap();
    chunked.set_block_value(target, Block::default()).unwrap();
    let ray = GridRay::new(StorageCoord { x: 4, y: 4, z: 2 }, [1, 0, 0], 10).unwrap();
    assert_eq!(oracle.raycast(ray).unwrap(), chunked.raycast(ray).unwrap());
    assert_eq!(chunked.raycast(ray).unwrap().unwrap().steps, 4);

    let miss = GridRay::new(StorageCoord { x: 0, y: 0, z: 0 }, [-1, 0, 0], 10).unwrap();
    assert_eq!(chunked.raycast(miss).unwrap(), None);
}

#[test]
fn in_memory_residency_guard_scopes_the_requested_region() {
    let world = ChunkedWorld::new(extent(), WorldExtent { x: 4, y: 2, z: 2 }).unwrap();
    let requested = region(
        StorageCoord { x: 1, y: 1, z: 1 },
        StorageCoord { x: 5, y: 4, z: 3 },
    );
    assert!(world.is_region_resident(requested));
    let guard = world.pin_region(requested).unwrap();
    assert_eq!(guard.region(), requested);
}

#[test]
fn removing_last_block_releases_empty_chunk() {
    let mut world = ChunkedWorld::new(extent(), WorldExtent { x: 4, y: 2, z: 2 }).unwrap();
    let coordinate = StorageCoord { x: 4, y: 2, z: 2 };
    world.set_block_value(coordinate, Block::default()).unwrap();
    assert_eq!(world.resident_chunk_count(), 1);
    assert_eq!(
        world.remove_block_value(coordinate).unwrap(),
        Some(Block::default())
    );
    assert_eq!(world.resident_chunk_count(), 0);
}

#[test]
fn maximum_u32_coordinate_does_not_require_a_flat_global_index() {
    let extent = WorldExtent {
        x: u32::MAX,
        y: 2,
        z: 2,
    };
    let coordinate = StorageCoord {
        x: u32::MAX - 1,
        y: 1,
        z: 1,
    };
    let mut world = ChunkedWorld::new(extent, WorldExtent { x: 32, y: 2, z: 2 }).unwrap();

    world.set_block_value(coordinate, Block::default()).unwrap();
    assert_eq!(
        world.get_block_value(coordinate).unwrap(),
        Some(Block::default())
    );
    assert_eq!(world.block_count(), 1);
}
