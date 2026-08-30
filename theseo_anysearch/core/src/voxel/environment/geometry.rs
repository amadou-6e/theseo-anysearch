use crate::voxel::world::Coord;

// ---------------------------------------------------------------------------
// Surface cell computation
// ---------------------------------------------------------------------------

pub(super) const DIRS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Returns the set of all free (non-geometry) cells reachable from the grid
/// boundary via 6-connectivity BFS.  Used to exclude enclosed cavities.
fn reachable_from_boundary(
    geo_set: &std::collections::HashSet<Coord>,
    extent: [u16; 3],
) -> std::collections::HashSet<Coord> {
    use std::collections::{HashSet, VecDeque};
    let mut reachable: HashSet<Coord> = HashSet::new();
    let mut queue: VecDeque<Coord> = VecDeque::new();

    for x in 1..=extent[0] {
        for y in 1..=extent[1] {
            for z in [1, extent[2]] {
                let c = (x, y, z);
                if !geo_set.contains(&c) && reachable.insert(c) {
                    queue.push_back(c);
                }
            }
        }
    }
    for x in 1..=extent[0] {
        for z in 1..=extent[2] {
            for y in [1, extent[1]] {
                let c = (x, y, z);
                if !geo_set.contains(&c) && reachable.insert(c) {
                    queue.push_back(c);
                }
            }
        }
    }
    for y in 1..=extent[1] {
        for z in 1..=extent[2] {
            for x in [1, extent[0]] {
                let c = (x, y, z);
                if !geo_set.contains(&c) && reachable.insert(c) {
                    queue.push_back(c);
                }
            }
        }
    }

    while let Some((x, y, z)) = queue.pop_front() {
        for (dx, dy, dz) in DIRS {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nz = z as i32 + dz;
            if nx >= 1
                && ny >= 1
                && nz >= 1
                && nx <= i32::from(extent[0])
                && ny <= i32::from(extent[1])
                && nz <= i32::from(extent[2])
            {
                let nc = (nx as u16, ny as u16, nz as u16);
                if !geo_set.contains(&nc) && reachable.insert(nc) {
                    queue.push_back(nc);
                }
            }
        }
    }
    reachable
}

/// Returns cells that are 6-adjacent to geometry, not in geometry, and
/// reachable from the grid boundary (i.e. not enclosed inside a cavity).
pub(super) fn compute_surface_cells(geometry: &[Coord], extent: [u16; 3]) -> Vec<Coord> {
    use std::collections::HashSet;
    let geo_set: HashSet<Coord> = geometry.iter().copied().collect();
    let reachable = reachable_from_boundary(&geo_set, extent);

    let mut surface: HashSet<Coord> = HashSet::new();
    for &(gx, gy, gz) in geometry {
        for (dx, dy, dz) in DIRS {
            let nx = gx as i32 + dx;
            let ny = gy as i32 + dy;
            let nz = gz as i32 + dz;
            if nx >= 1
                && ny >= 1
                && nz >= 1
                && nx <= i32::from(extent[0])
                && ny <= i32::from(extent[1])
                && nz <= i32::from(extent[2])
            {
                let nc = (nx as u16, ny as u16, nz as u16);
                if !geo_set.contains(&nc) && reachable.contains(&nc) {
                    surface.insert(nc);
                }
            }
        }
    }
    let mut cells: Vec<Coord> = surface.into_iter().collect();
    cells.sort(); // deterministic ordering
    cells
}

pub(super) fn coord_to_i32(coord: Coord) -> [i32; 3] {
    [i32::from(coord.0), i32::from(coord.1), i32::from(coord.2)]
}

/// L2 (Euclidean) distance between two coords.
pub(super) fn l2(a: Coord, b: Coord) -> f32 {
    let dx = a.0 as f32 - b.0 as f32;
    let dy = a.1 as f32 - b.1 as f32;
    let dz = a.2 as f32 - b.2 as f32;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Manhattan distance between two coords (used for the observation field only).
pub(super) fn manhattan(a: Coord, b: Coord) -> u32 {
    let dx = (a.0 as i32 - b.0 as i32).unsigned_abs();
    let dy = (a.1 as i32 - b.1 as i32).unsigned_abs();
    let dz = (a.2 as i32 - b.2 as i32).unsigned_abs();
    dx + dy + dz
}
