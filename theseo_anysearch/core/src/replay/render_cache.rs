use std::collections::{HashMap, HashSet};

use crate::voxel::world::StorageCoord;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct ChunkCoord {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum FaceDirection {
    NegativeX,
    PositiveX,
    NegativeY,
    PositiveY,
    NegativeZ,
    PositiveZ,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExposedFace {
    pub voxel: StorageCoord,
    pub direction: FaceDirection,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct RenderCacheKey {
    pub world_identity: String,
    pub chunk: ChunkCoord,
    pub overlay_revision: u64,
    pub settings_revision: u64,
}

#[derive(Clone, Debug, Default)]
pub struct ChunkRenderData {
    pub faces: Vec<ExposedFace>,
}

#[derive(Default)]
pub struct ChunkRenderCache {
    entries: HashMap<RenderCacheKey, ChunkRenderData>,
    builds: u64,
    hits: u64,
}

impl ChunkRenderCache {
    pub fn get_or_build(
        &mut self,
        key: RenderCacheKey,
        occupied: &HashSet<StorageCoord>,
        chunk_edge: u32,
    ) -> &ChunkRenderData {
        if self.entries.contains_key(&key) {
            self.hits += 1;
        } else {
            self.builds += 1;
            let data = extract_exposed_faces(key.chunk, occupied, chunk_edge);
            self.entries.insert(key.clone(), data);
        }
        self.entries.get(&key).expect("render entry was inserted")
    }

    pub fn invalidate_mutation(&mut self, coordinate: StorageCoord, chunk_edge: u32) {
        let affected = affected_chunks(coordinate, chunk_edge);
        self.entries.retain(|key, _| !affected.contains(&key.chunk));
    }

    pub fn builds(&self) -> u64 {
        self.builds
    }
    pub fn hits(&self) -> u64 {
        self.hits
    }
}

pub fn extract_exposed_faces(
    chunk: ChunkCoord,
    occupied: &HashSet<StorageCoord>,
    chunk_edge: u32,
) -> ChunkRenderData {
    // Neighbors outside the loaded region are conservatively treated as empty;
    // their boundary faces self-correct when an adjacent region becomes visible.
    let minimum = StorageCoord {
        x: chunk.x * chunk_edge,
        y: chunk.y * chunk_edge,
        z: chunk.z * chunk_edge,
    };
    let maximum = StorageCoord {
        x: minimum.x.saturating_add(chunk_edge),
        y: minimum.y.saturating_add(chunk_edge),
        z: minimum.z.saturating_add(chunk_edge),
    };
    let mut faces = Vec::new();
    for &voxel in occupied.iter().filter(|voxel| {
        voxel.x >= minimum.x
            && voxel.x < maximum.x
            && voxel.y >= minimum.y
            && voxel.y < maximum.y
            && voxel.z >= minimum.z
            && voxel.z < maximum.z
    }) {
        for (direction, neighbor) in neighbors(voxel) {
            if neighbor
                .map(|coord| occupied.contains(&coord))
                .unwrap_or(false)
            {
                continue;
            }
            faces.push(ExposedFace { voxel, direction });
        }
    }
    faces.sort_by_key(|face| (face.voxel.global_key(), face.direction as u8));
    ChunkRenderData { faces }
}

/// Stable geometry revision for one chunk and its one-voxel neighbor halo.
///
/// Including the halo makes a boundary mutation revise both adjacent chunks,
/// while unrelated replay-step changes leave the key unchanged.
pub fn chunk_occupancy_revision(
    chunk: ChunkCoord,
    occupied: &HashSet<StorageCoord>,
    chunk_edge: u32,
) -> u64 {
    let minimum = StorageCoord {
        x: chunk.x.saturating_mul(chunk_edge).saturating_sub(1),
        y: chunk.y.saturating_mul(chunk_edge).saturating_sub(1),
        z: chunk.z.saturating_mul(chunk_edge).saturating_sub(1),
    };
    let maximum = StorageCoord {
        x: chunk
            .x
            .saturating_add(1)
            .saturating_mul(chunk_edge)
            .saturating_add(1),
        y: chunk
            .y
            .saturating_add(1)
            .saturating_mul(chunk_edge)
            .saturating_add(1),
        z: chunk
            .z
            .saturating_add(1)
            .saturating_mul(chunk_edge)
            .saturating_add(1),
    };
    let mut coordinates = occupied
        .iter()
        .filter(|coordinate| {
            coordinate.x >= minimum.x
                && coordinate.x < maximum.x
                && coordinate.y >= minimum.y
                && coordinate.y < maximum.y
                && coordinate.z >= minimum.z
                && coordinate.z < maximum.z
        })
        .copied()
        .collect::<Vec<_>>();
    coordinates.sort_by_key(|coordinate| coordinate.global_key());
    let mut revision = 0xcbf29ce484222325_u64;
    for coordinate in coordinates {
        for byte in coordinate
            .x
            .to_le_bytes()
            .into_iter()
            .chain(coordinate.y.to_le_bytes())
            .chain(coordinate.z.to_le_bytes())
        {
            revision ^= u64::from(byte);
            revision = revision.wrapping_mul(0x100000001b3);
        }
    }
    revision
}

fn neighbors(voxel: StorageCoord) -> [(FaceDirection, Option<StorageCoord>); 6] {
    [
        (
            FaceDirection::NegativeX,
            voxel.x.checked_sub(1).map(|x| StorageCoord { x, ..voxel }),
        ),
        (
            FaceDirection::PositiveX,
            voxel.x.checked_add(1).map(|x| StorageCoord { x, ..voxel }),
        ),
        (
            FaceDirection::NegativeY,
            voxel.y.checked_sub(1).map(|y| StorageCoord { y, ..voxel }),
        ),
        (
            FaceDirection::PositiveY,
            voxel.y.checked_add(1).map(|y| StorageCoord { y, ..voxel }),
        ),
        (
            FaceDirection::NegativeZ,
            voxel.z.checked_sub(1).map(|z| StorageCoord { z, ..voxel }),
        ),
        (
            FaceDirection::PositiveZ,
            voxel.z.checked_add(1).map(|z| StorageCoord { z, ..voxel }),
        ),
    ]
}

fn affected_chunks(coordinate: StorageCoord, chunk_edge: u32) -> HashSet<ChunkCoord> {
    let base = ChunkCoord {
        x: coordinate.x / chunk_edge,
        y: coordinate.y / chunk_edge,
        z: coordinate.z / chunk_edge,
    };
    let mut chunks = HashSet::from([base]);
    for (value, axis) in [(coordinate.x, 0), (coordinate.y, 1), (coordinate.z, 2)] {
        if value % chunk_edge == 0 && value > 0 {
            chunks.insert(offset(base, axis, -1));
        }
        if value % chunk_edge == chunk_edge - 1 {
            chunks.insert(offset(base, axis, 1));
        }
    }
    chunks
}

fn offset(mut chunk: ChunkCoord, axis: u8, delta: i32) -> ChunkCoord {
    let value = match axis {
        0 => &mut chunk.x,
        1 => &mut chunk.y,
        _ => &mut chunk.z,
    };
    *value = if delta < 0 {
        value.saturating_sub(1)
    } else {
        value.saturating_add(1)
    };
    chunk
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coordinate(x: u32, y: u32, z: u32) -> StorageCoord {
        StorageCoord { x, y, z }
    }

    #[test]
    fn adjacent_voxels_do_not_emit_their_shared_interior_faces() {
        let occupied = HashSet::from([coordinate(15, 2, 2), coordinate(16, 2, 2)]);
        let left = extract_exposed_faces(ChunkCoord { x: 0, y: 0, z: 0 }, &occupied, 16);
        let right = extract_exposed_faces(ChunkCoord { x: 1, y: 0, z: 0 }, &occupied, 16);
        assert_eq!(left.faces.len(), 5);
        assert_eq!(right.faces.len(), 5);
        assert!(!left
            .faces
            .iter()
            .any(|face| face.direction == FaceDirection::PositiveX));
        assert!(!right
            .faces
            .iter()
            .any(|face| face.direction == FaceDirection::NegativeX));
    }

    #[test]
    fn unchanged_key_reuses_cached_geometry_across_camera_motion() {
        let occupied = HashSet::from([coordinate(2, 2, 2)]);
        let key = RenderCacheKey {
            world_identity: "world".to_string(),
            chunk: ChunkCoord { x: 0, y: 0, z: 0 },
            overlay_revision: 7,
            settings_revision: 0,
        };
        let mut cache = ChunkRenderCache::default();
        cache.get_or_build(key.clone(), &occupied, 16);
        cache.get_or_build(key, &occupied, 16);
        assert_eq!(cache.builds(), 1);
        assert_eq!(cache.hits(), 1);
    }

    #[test]
    fn boundary_mutation_invalidates_both_neighboring_chunks() {
        let occupied = HashSet::from([coordinate(15, 2, 2), coordinate(16, 2, 2)]);
        let mut cache = ChunkRenderCache::default();
        for x in [0, 1] {
            cache.get_or_build(
                RenderCacheKey {
                    world_identity: "world".to_string(),
                    chunk: ChunkCoord { x, y: 0, z: 0 },
                    overlay_revision: 0,
                    settings_revision: 0,
                },
                &occupied,
                16,
            );
        }
        cache.invalidate_mutation(coordinate(15, 2, 2), 16);
        for x in [0, 1] {
            cache.get_or_build(
                RenderCacheKey {
                    world_identity: "world".to_string(),
                    chunk: ChunkCoord { x, y: 0, z: 0 },
                    overlay_revision: 0,
                    settings_revision: 0,
                },
                &occupied,
                16,
            );
        }
        assert_eq!(cache.builds(), 4);
    }

    #[test]
    fn per_chunk_revision_ignores_unrelated_changes_and_tracks_boundary_halo() {
        let base = HashSet::from([coordinate(15, 2, 2), coordinate(80, 2, 2)]);
        let mut unrelated = base.clone();
        unrelated.insert(coordinate(81, 2, 2));
        let left = ChunkCoord { x: 0, y: 0, z: 0 };
        assert_eq!(
            chunk_occupancy_revision(left, &base, 16),
            chunk_occupancy_revision(left, &unrelated, 16)
        );
        let mut boundary = base.clone();
        boundary.insert(coordinate(16, 2, 2));
        assert_ne!(
            chunk_occupancy_revision(left, &base, 16),
            chunk_occupancy_revision(left, &boundary, 16)
        );
    }
}
