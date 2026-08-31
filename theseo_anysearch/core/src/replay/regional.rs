use std::{
    collections::HashMap,
    path::Path,
    time::{Duration, Instant},
};

use crate::voxel::world::{
    BoundedRegion, DiskCacheMetrics, StorageCoord, WorldAccessError, WorldRead, WorldState,
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplayMutation {
    pub coordinate: StorageCoord,
    pub occupied: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RegionalReplayFrame {
    pub region: BoundedRegion,
    pub render_origin: StorageCoord,
    pub occupied: Vec<StorageCoord>,
    pub cache_metrics: Option<DiskCacheMetrics>,
    pub load_time: Duration,
}

#[derive(Clone, Debug)]
pub struct RegionalReplaySource {
    world: WorldState,
}

impl RegionalReplaySource {
    pub fn open(root: &Path, maximum_decoded_bytes: usize) -> Result<Self, WorldAccessError> {
        Ok(Self {
            world: WorldState::from_compiled_pack(root, maximum_decoded_bytes)?,
        })
    }

    pub const fn from_world(world: WorldState) -> Self {
        Self { world }
    }

    pub fn load_agent_region(
        &self,
        center: StorageCoord,
        radius: u32,
        mutations: &[ReplayMutation],
    ) -> Result<RegionalReplayFrame, WorldAccessError> {
        let started = Instant::now();
        let region = agent_region(center, radius, self.world.extent())?;
        self.world.prefetch_region(region)?;
        let mut occupied = self
            .world
            .blocks_in_region(region)?
            .into_iter()
            .map(|(coordinate, _)| (coordinate, true))
            .collect::<HashMap<_, _>>();
        for mutation in mutations
            .iter()
            .filter(|mutation| region.contains(mutation.coordinate))
        {
            if mutation.occupied {
                occupied.insert(mutation.coordinate, true);
            } else {
                occupied.remove(&mutation.coordinate);
            }
        }
        let mut occupied = occupied.into_keys().collect::<Vec<_>>();
        occupied.sort_by_key(|coordinate| coordinate.global_key());
        Ok(RegionalReplayFrame {
            region,
            render_origin: region.minimum,
            occupied,
            cache_metrics: self.world.disk_cache_metrics(),
            load_time: started.elapsed(),
        })
    }
}

pub fn agent_region(
    center: StorageCoord,
    radius: u32,
    extent: crate::voxel::world::WorldExtent,
) -> Result<BoundedRegion, WorldAccessError> {
    let minimum = StorageCoord {
        x: center.x.saturating_sub(radius),
        y: center.y.saturating_sub(radius),
        z: center.z.saturating_sub(radius),
    };
    let maximum_exclusive = StorageCoord {
        x: center
            .x
            .saturating_add(radius)
            .saturating_add(1)
            .min(extent.x),
        y: center
            .y
            .saturating_add(radius)
            .saturating_add(1)
            .min(extent.y),
        z: center
            .z
            .saturating_add(radius)
            .saturating_add(1)
            .min(extent.z),
    };
    BoundedRegion::new(minimum, maximum_exclusive, extent)
}

pub fn camera_relative(coordinate: StorageCoord, render_origin: StorageCoord) -> (f32, f32, f32) {
    (
        (i64::from(coordinate.x) - i64::from(render_origin.x)) as f32,
        (i64::from(coordinate.y) - i64::from(render_origin.y)) as f32,
        (i64::from(coordinate.z) - i64::from(render_origin.z)) as f32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::world::{Block, WorldExtent};

    #[test]
    fn agent_region_clamps_at_non_cubic_boundaries() {
        let region = agent_region(
            StorageCoord { x: 1, y: 47, z: 31 },
            4,
            WorldExtent {
                x: 64,
                y: 48,
                z: 32,
            },
        )
        .unwrap();
        assert_eq!(region.minimum, StorageCoord { x: 0, y: 43, z: 27 });
        assert_eq!(
            region.maximum_exclusive,
            StorageCoord { x: 6, y: 48, z: 32 }
        );
    }

    #[test]
    fn regional_frame_resolves_base_additions_and_tombstones() {
        let mut world = WorldState::new_chunked(16);
        world.replace_base_blocks([
            ((30, 30, 30), Block::default()),
            ((31, 30, 30), Block::default()),
        ]);
        let source = RegionalReplaySource::from_world(world);
        let frame = source
            .load_agent_region(
                StorageCoord {
                    x: 30,
                    y: 30,
                    z: 30,
                },
                2,
                &[
                    ReplayMutation {
                        coordinate: StorageCoord {
                            x: 30,
                            y: 30,
                            z: 30,
                        },
                        occupied: false,
                    },
                    ReplayMutation {
                        coordinate: StorageCoord {
                            x: 32,
                            y: 30,
                            z: 30,
                        },
                        occupied: true,
                    },
                ],
            )
            .unwrap();
        assert_eq!(
            frame.occupied,
            vec![
                StorageCoord {
                    x: 31,
                    y: 30,
                    z: 30
                },
                StorageCoord {
                    x: 32,
                    y: 30,
                    z: 30
                },
            ]
        );
    }

    #[test]
    fn regional_frame_reads_across_chunk_boundaries() {
        let mut world = WorldState::new_chunked(16);
        world.replace_base_blocks([
            ((15, 8, 8), Block::default()),
            ((16, 8, 8), Block::default()),
            ((17, 8, 8), Block::default()),
        ]);
        let frame = RegionalReplaySource::from_world(world)
            .load_agent_region(StorageCoord { x: 16, y: 8, z: 8 }, 1, &[])
            .unwrap();
        assert_eq!(frame.occupied.len(), 3);
        assert_eq!(frame.region.minimum.x, 15);
        assert_eq!(frame.region.maximum_exclusive.x, 18);
    }

    #[test]
    fn cold_teleport_replaces_the_visible_region_without_global_enumeration() {
        let mut world = WorldState::new_chunked(16);
        world.replace_base_blocks([
            ((2, 2, 2), Block::default()),
            ((900, 800, 700), Block::default()),
        ]);
        let source = RegionalReplaySource::from_world(world);
        let first = source
            .load_agent_region(StorageCoord { x: 2, y: 2, z: 2 }, 2, &[])
            .unwrap();
        let teleported = source
            .load_agent_region(
                StorageCoord {
                    x: 900,
                    y: 800,
                    z: 700,
                },
                2,
                &[],
            )
            .unwrap();
        assert_eq!(first.occupied, vec![StorageCoord { x: 2, y: 2, z: 2 }]);
        assert_eq!(
            teleported.occupied,
            vec![StorageCoord {
                x: 900,
                y: 800,
                z: 700
            }]
        );
        assert_ne!(first.render_origin, teleported.render_origin);
    }

    #[test]
    fn camera_relative_conversion_preserves_voxel_steps_near_native_limit() {
        let origin = StorageCoord {
            x: 59_984,
            y: 39_984,
            z: 19_984,
        };
        assert_eq!(
            camera_relative(
                StorageCoord {
                    x: 60_000,
                    y: 40_000,
                    z: 20_000
                },
                origin,
            ),
            (16.0, 16.0, 16.0)
        );
        assert_eq!(
            camera_relative(
                StorageCoord {
                    x: 59_999,
                    y: 39_999,
                    z: 19_999
                },
                origin,
            ),
            (15.0, 15.0, 15.0)
        );
    }
}
