use std::collections::HashSet;

use crate::voxel::world::{Block, Coord, WorldState, WORLD_SIZE};

use super::StlMesh;

#[derive(Clone, Debug)]
pub struct BlockPlacement {
    pub coord: Coord,
    pub block: Block,
}

pub fn voxelize_mesh(mesh: &StlMesh, origin: Coord, scale: f32) -> Vec<BlockPlacement> {
    voxelize_mesh_f32(
        mesh,
        (origin.0 as f32, origin.1 as f32, origin.2 as f32),
        scale,
    )
}

/// Like `voxelize_mesh` but accepts a floating-point origin, allowing sub-voxel
/// placement adjustments (e.g. for padding-aware, normalised STL loading).
pub fn voxelize_mesh_f32(
    mesh: &StlMesh,
    origin: (f32, f32, f32),
    scale: f32,
) -> Vec<BlockPlacement> {
    let mut coords = HashSet::new();

    for tri in &mesh.triangles {
        for v in tri {
            insert_point(&mut coords, origin, scale, *v);
        }

        // Sample triangle edges to produce a connected surface graph for pathfinding.
        sample_edge(&mut coords, origin, scale, tri[0], tri[1]);
        sample_edge(&mut coords, origin, scale, tri[1], tri[2]);
        sample_edge(&mut coords, origin, scale, tri[2], tri[0]);
        sample_triangle(&mut coords, origin, scale, tri[0], tri[1], tri[2]);
    }

    coords = solid_fill(coords);

    coords
        .into_iter()
        .map(|coord| BlockPlacement {
            coord,
            block: Block::default(),
        })
        .collect()
}

fn solid_fill(surface: HashSet<Coord>) -> HashSet<Coord> {
    if surface.is_empty() {
        return surface;
    }

    let min_x = surface
        .iter()
        .map(|c| c.0)
        .min()
        .unwrap_or(0)
        .saturating_sub(1);
    let min_y = surface
        .iter()
        .map(|c| c.1)
        .min()
        .unwrap_or(0)
        .saturating_sub(1);
    let min_z = surface
        .iter()
        .map(|c| c.2)
        .min()
        .unwrap_or(0)
        .saturating_sub(1);
    let max_x = surface
        .iter()
        .map(|c| c.0)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
        .min(WORLD_SIZE - 1);
    let max_y = surface
        .iter()
        .map(|c| c.1)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
        .min(WORLD_SIZE - 1);
    let max_z = surface
        .iter()
        .map(|c| c.2)
        .max()
        .unwrap_or(0)
        .saturating_add(1)
        .min(WORLD_SIZE - 1);

    let mut exterior = HashSet::new();
    let mut queue = std::collections::VecDeque::new();

    for x in min_x..=max_x {
        for y in min_y..=max_y {
            for z in [min_z, max_z] {
                let c = (x, y, z);
                if !surface.contains(&c) && exterior.insert(c) {
                    queue.push_back(c);
                }
            }
        }
    }
    for x in min_x..=max_x {
        for z in min_z..=max_z {
            for y in [min_y, max_y] {
                let c = (x, y, z);
                if !surface.contains(&c) && exterior.insert(c) {
                    queue.push_back(c);
                }
            }
        }
    }
    for y in min_y..=max_y {
        for z in min_z..=max_z {
            for x in [min_x, max_x] {
                let c = (x, y, z);
                if !surface.contains(&c) && exterior.insert(c) {
                    queue.push_back(c);
                }
            }
        }
    }

    while let Some((x, y, z)) = queue.pop_front() {
        let neigh = [
            (x.saturating_add(1), y, z),
            (x.saturating_sub(1), y, z),
            (x, y.saturating_add(1), z),
            (x, y.saturating_sub(1), z),
            (x, y, z.saturating_add(1)),
            (x, y, z.saturating_sub(1)),
        ];
        for n in neigh {
            if n.0 < min_x
                || n.0 > max_x
                || n.1 < min_y
                || n.1 > max_y
                || n.2 < min_z
                || n.2 > max_z
            {
                continue;
            }
            if surface.contains(&n) || !exterior.insert(n) {
                continue;
            }
            queue.push_back(n);
        }
    }

    let mut filled = surface.clone();
    for x in min_x..=max_x {
        for y in min_y..=max_y {
            for z in min_z..=max_z {
                let c = (x, y, z);
                if !exterior.contains(&c) {
                    filled.insert(c);
                }
            }
        }
    }
    filled
}

fn sample_edge(
    coords: &mut HashSet<Coord>,
    origin: (f32, f32, f32),
    scale: f32,
    a: [f32; 3],
    b: [f32; 3],
) {
    let dx = (b[0] - a[0]) * scale;
    let dy = (b[1] - a[1]) * scale;
    let dz = (b[2] - a[2]) * scale;
    let length = (dx * dx + dy * dy + dz * dz).sqrt();
    let steps = length.ceil().max(1.0) as i32;

    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let p = [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ];
        insert_point(coords, origin, scale, p);
    }
}

fn sample_triangle(
    coords: &mut HashSet<Coord>,
    origin: (f32, f32, f32),
    scale: f32,
    a: [f32; 3],
    b: [f32; 3],
    c: [f32; 3],
) {
    let len_ab = edge_len_scaled(a, b, scale);
    let len_bc = edge_len_scaled(b, c, scale);
    let len_ca = edge_len_scaled(c, a, scale);
    let steps = len_ab.max(len_bc).max(len_ca).ceil().max(1.0) as i32;

    for i in 0..=steps {
        for j in 0..=(steps - i) {
            let u = i as f32 / steps as f32;
            let v = j as f32 / steps as f32;
            let w = 1.0 - u - v;
            let p = [
                a[0] * u + b[0] * v + c[0] * w,
                a[1] * u + b[1] * v + c[1] * w,
                a[2] * u + b[2] * v + c[2] * w,
            ];
            insert_point(coords, origin, scale, p);
        }
    }
}

fn edge_len_scaled(a: [f32; 3], b: [f32; 3], scale: f32) -> f32 {
    let dx = (b[0] - a[0]) * scale;
    let dy = (b[1] - a[1]) * scale;
    let dz = (b[2] - a[2]) * scale;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// A watertight unit cube (12 triangles) for solid-fill tests.
#[cfg(test)]
const CUBE_STL: &str = r#"solid cube
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 1 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 1 1 1
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 0 1 1
      vertex 1 1 1
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 1
      vertex 0 0 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 1 1 1
      vertex 0 1 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 1 1 0
      vertex 1 1 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 1 1
      vertex 0 1 0
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 0 1
      vertex 0 1 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 1 0
      vertex 1 1 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 1 1
      vertex 1 0 1
    endloop
  endfacet
endsolid cube"#;

fn insert_point(coords: &mut HashSet<Coord>, origin: (f32, f32, f32), scale: f32, point: [f32; 3]) {
    let x = origin.0 + point[0] * scale;
    let y = origin.1 + point[1] * scale;
    let z = origin.2 + point[2] * scale;

    if x < 0.0 || y < 0.0 || z < 0.0 {
        return;
    }

    let c = (x.floor() as u16, y.floor() as u16, z.floor() as u16);
    if WorldState::in_bounds(c) && c.0 < WORLD_SIZE && c.1 < WORLD_SIZE && c.2 < WORLD_SIZE {
        coords.insert(c);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::world::ingest::parse_ascii_stl;

    const TRIANGLE_STL: &str = "vertex 0 0 0\nvertex 1 0 0\nvertex 0 1 0\n";

    fn triangle_mesh() -> StlMesh {
        parse_ascii_stl(TRIANGLE_STL).unwrap()
    }

    fn cube_mesh() -> StlMesh {
        parse_ascii_stl(CUBE_STL).unwrap()
    }

    #[test]
    fn single_triangle_produces_voxels() {
        let placements = voxelize_mesh(&triangle_mesh(), (100, 100, 100), 1.0);
        assert!(!placements.is_empty());
    }

    #[test]
    fn scale_increases_voxel_count() {
        let small = voxelize_mesh(&triangle_mesh(), (100, 100, 100), 1.0);
        let large = voxelize_mesh(&triangle_mesh(), (100, 100, 100), 10.0);
        assert!(large.len() > small.len());
    }

    #[test]
    fn origin_shifts_placement() {
        let origin = (200u16, 300u16, 400u16);
        let placements = voxelize_mesh(&triangle_mesh(), origin, 1.0);
        for p in &placements {
            assert!(p.coord.0 >= origin.0);
            assert!(p.coord.1 >= origin.1);
            assert!(p.coord.2 >= origin.2);
        }
    }

    #[test]
    fn out_of_bounds_points_discarded() {
        // Place triangle near the world edge; negative-offset verts would go out of bounds
        let placements = voxelize_mesh(&triangle_mesh(), (0, 0, 0), 1.0);
        for p in &placements {
            assert!(WorldState::in_bounds(p.coord));
        }
    }

    #[test]
    fn solid_fill_fills_interior() {
        // A watertight cube must produce more voxels than a flat triangle at the same scale,
        // because solid_fill adds interior cells to the cube but not the open triangle.
        let cube = voxelize_mesh(&cube_mesh(), (100, 100, 100), 5.0);
        let tri = voxelize_mesh(&triangle_mesh(), (100, 100, 100), 5.0);
        assert!(cube.len() > tri.len());
        // A 1×1×1 cube at scale 5 produces a ≥ 5³ = 125 voxel solid
        assert!(cube.len() >= 125);
    }

    #[test]
    fn empty_mesh_returns_empty() {
        let empty = StlMesh::default();
        assert!(voxelize_mesh(&empty, (100, 100, 100), 1.0).is_empty());
    }

    #[test]
    fn no_duplicates() {
        let placements = voxelize_mesh(&cube_mesh(), (100, 100, 100), 5.0);
        let coords: HashSet<Coord> = placements.iter().map(|p| p.coord).collect();
        assert_eq!(coords.len(), placements.len());
    }
}
