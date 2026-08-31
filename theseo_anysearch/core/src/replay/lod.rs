use std::cmp::Ordering;
use std::collections::HashSet;

use super::render_cache::ChunkCoord;
use crate::voxel::world::{StorageCoord, WorldExtent};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraChunkView {
    pub center: [f64; 3],
    pub half_extent: [f64; 3],
    pub forward: [f64; 3],
    pub minimum_forward_dot: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ChunkBudgets {
    pub visible: usize,
    pub detailed: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ChunkSelection {
    pub detailed: Vec<ChunkCoord>,
    pub coarse: Vec<ChunkCoord>,
    pub considered: usize,
}

/// Expands a visible chunk set with indexed neighbors without inventing empty chunks.
pub fn expand_chunk_halo(
    visible: &[ChunkCoord],
    indexed_chunks: impl IntoIterator<Item = ChunkCoord>,
    rings: u32,
) -> Vec<ChunkCoord> {
    let visible = visible.iter().copied().collect::<HashSet<_>>();
    let mut resident = indexed_chunks
        .into_iter()
        .filter(|candidate| {
            visible.iter().any(|chunk| {
                chunk.x.abs_diff(candidate.x) <= rings
                    && chunk.y.abs_diff(candidate.y) <= rings
                    && chunk.z.abs_diff(candidate.z) <= rings
            })
        })
        .collect::<Vec<_>>();
    resident.sort_by_key(|chunk| (chunk.x, chunk.y, chunk.z));
    resident.dedup();
    resident
}

/// Selects every indexed chunk intersecting an agent-centered voxel box.
pub fn chunks_intersecting_box(
    indexed_chunks: impl IntoIterator<Item = ChunkCoord>,
    center: StorageCoord,
    radius: u32,
    shape: WorldExtent,
) -> Vec<ChunkCoord> {
    let minimum = StorageCoord {
        x: center.x.saturating_sub(radius),
        y: center.y.saturating_sub(radius),
        z: center.z.saturating_sub(radius),
    };
    let maximum = StorageCoord {
        x: center.x.saturating_add(radius),
        y: center.y.saturating_add(radius),
        z: center.z.saturating_add(radius),
    };
    let minimum_chunk = ChunkCoord {
        x: minimum.x / shape.x,
        y: minimum.y / shape.y,
        z: minimum.z / shape.z,
    };
    let maximum_chunk = ChunkCoord {
        x: maximum.x / shape.x,
        y: maximum.y / shape.y,
        z: maximum.z / shape.z,
    };
    let mut visible = indexed_chunks
        .into_iter()
        .filter(|chunk| {
            (minimum_chunk.x..=maximum_chunk.x).contains(&chunk.x)
                && (minimum_chunk.y..=maximum_chunk.y).contains(&chunk.y)
                && (minimum_chunk.z..=maximum_chunk.z).contains(&chunk.z)
        })
        .collect::<Vec<_>>();
    visible.sort_by_key(|chunk| (chunk.x, chunk.y, chunk.z));
    visible.dedup();
    visible
}

/// Conservatively selects chunks before any voxel payload is decoded.
pub fn select_chunks(
    indexed_chunks: impl IntoIterator<Item = ChunkCoord>,
    view: CameraChunkView,
    budgets: ChunkBudgets,
) -> ChunkSelection {
    let mut candidates = indexed_chunks
        .into_iter()
        .filter_map(|chunk| {
            let point = [
                chunk.x as f64 + 0.5,
                chunk.y as f64 + 0.5,
                chunk.z as f64 + 0.5,
            ];
            if (0..3)
                .any(|axis| (point[axis] - view.center[axis]).abs() > view.half_extent[axis] + 0.5)
            {
                return None;
            }
            let offset = [
                point[0] - view.center[0],
                point[1] - view.center[1],
                point[2] - view.center[2],
            ];
            let distance_sq = offset.iter().map(|value| value * value).sum::<f64>();
            if distance_sq > 0.0 {
                let forward_dot = (offset[0] * view.forward[0]
                    + offset[1] * view.forward[1]
                    + offset[2] * view.forward[2])
                    / distance_sq.sqrt();
                if forward_dot < view.minimum_forward_dot {
                    return None;
                }
            }
            Some((chunk, distance_sq))
        })
        .collect::<Vec<_>>();
    let considered = candidates.len();
    candidates.sort_by(|left, right| {
        left.1
            .partial_cmp(&right.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| (left.0.x, left.0.y, left.0.z).cmp(&(right.0.x, right.0.y, right.0.z)))
    });
    candidates.truncate(budgets.visible);
    let detailed_count = budgets.detailed.min(candidates.len());
    let detailed = candidates[..detailed_count]
        .iter()
        .map(|value| value.0)
        .collect();
    let coarse = candidates[detailed_count..]
        .iter()
        .map(|value| value.0)
        .collect();
    ChunkSelection {
        detailed,
        coarse,
        considered,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zoomed_out_selection_never_exceeds_hard_budgets() {
        let chunks = (0..10_000).map(|index| ChunkCoord {
            x: index % 100,
            y: index / 100,
            z: 0,
        });
        let selected = select_chunks(
            chunks,
            CameraChunkView {
                center: [50.0, 50.0, 0.0],
                half_extent: [10_000.0; 3],
                forward: [0.0, 0.0, 1.0],
                minimum_forward_dot: -1.0,
            },
            ChunkBudgets {
                visible: 64,
                detailed: 16,
            },
        );
        assert_eq!(selected.detailed.len(), 16);
        assert_eq!(selected.coarse.len(), 48);
        assert_eq!(selected.considered, 10_000);
    }

    #[test]
    fn view_filter_handles_non_cubic_boundary_regions() {
        let selected = select_chunks(
            [
                ChunkCoord { x: 0, y: 0, z: 0 },
                ChunkCoord { x: 7, y: 2, z: 1 },
                ChunkCoord { x: 8, y: 2, z: 1 },
            ],
            CameraChunkView {
                center: [7.5, 2.5, 1.5],
                // Stop before the adjacent chunk boundary. Selection is
                // conservative and includes every chunk volume intersecting
                // the view, not only chunks whose centers lie inside it.
                half_extent: [0.4, 0.4, 0.4],
                forward: [1.0, 0.0, 0.0],
                minimum_forward_dot: -1.0,
            },
            ChunkBudgets {
                visible: 8,
                detailed: 4,
            },
        );
        assert_eq!(selected.detailed, vec![ChunkCoord { x: 7, y: 2, z: 1 }]);
    }

    #[test]
    fn halo_contains_visible_chunks_and_one_indexed_neighbor_ring() {
        let resident = expand_chunk_halo(
            &[ChunkCoord { x: 4, y: 4, z: 4 }],
            [
                ChunkCoord { x: 3, y: 4, z: 4 },
                ChunkCoord { x: 4, y: 4, z: 4 },
                ChunkCoord { x: 5, y: 5, z: 5 },
                ChunkCoord { x: 6, y: 4, z: 4 },
            ],
            1,
        );
        assert_eq!(
            resident,
            vec![
                ChunkCoord { x: 3, y: 4, z: 4 },
                ChunkCoord { x: 4, y: 4, z: 4 },
                ChunkCoord { x: 5, y: 5, z: 5 },
            ]
        );
    }

    #[test]
    fn moving_box_selects_every_intersecting_chunk_before_crossing_it() {
        let indexed = [
            ChunkCoord { x: 0, y: 0, z: 0 },
            ChunkCoord { x: 1, y: 0, z: 0 },
            ChunkCoord { x: 2, y: 0, z: 0 },
        ];
        let shape = WorldExtent {
            x: 32,
            y: 32,
            z: 32,
        };
        assert_eq!(
            chunks_intersecting_box(
                indexed,
                StorageCoord {
                    x: 16,
                    y: 16,
                    z: 16
                },
                16,
                shape,
            ),
            vec![
                ChunkCoord { x: 0, y: 0, z: 0 },
                ChunkCoord { x: 1, y: 0, z: 0 },
            ]
        );
        assert_eq!(
            chunks_intersecting_box(
                indexed,
                StorageCoord {
                    x: 48,
                    y: 16,
                    z: 16
                },
                16,
                shape,
            ),
            vec![
                ChunkCoord { x: 1, y: 0, z: 0 },
                ChunkCoord { x: 2, y: 0, z: 0 },
            ]
        );
    }
}
