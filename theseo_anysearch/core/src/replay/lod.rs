use std::cmp::Ordering;

use super::render_cache::ChunkCoord;

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
                half_extent: [0.6, 0.6, 0.6],
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
}
