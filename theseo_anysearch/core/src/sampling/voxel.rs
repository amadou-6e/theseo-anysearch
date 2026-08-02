use std::{fs::File, io::Read, path::Path};

use crate::world::{
    ingest::{parse_ascii_stl, voxelize_mesh, voxelize_mesh_f32},
    Block, World, WorldState,
};

/// Reusable voxel-geometry sampler without episode or Python state.
pub struct VoxelSampler {
    world: WorldState,
    grid_size: u16,
}

impl VoxelSampler {
    pub fn new(grid_size: u16) -> Self {
        Self {
            world: WorldState::new(),
            grid_size,
        }
    }

    pub fn load_stl(&mut self, path: &Path, scale: f32) -> Result<usize, String> {
        let contents = read_stl(path)?;
        let mesh = parse_ascii_stl(&contents).map_err(|error| format!("{error:?}"))?;
        let placements = voxelize_mesh(&mesh, (1, 1, 1), scale);
        Ok(self.replace_placements(placements))
    }

    pub fn load_stl_normalized(
        &mut self,
        path: &Path,
        scale: f32,
        origin: (f32, f32, f32),
    ) -> Result<usize, String> {
        let contents = read_stl(path)?;
        let mesh = parse_ascii_stl(&contents).map_err(|error| format!("{error:?}"))?;
        let minimum = mesh
            .triangles
            .iter()
            .flatten()
            .fold([f32::MAX; 3], |mut minimum, vertex| {
                for axis in 0..3 {
                    minimum[axis] = minimum[axis].min(vertex[axis]);
                }
                minimum
            });
        let minimum = if minimum[0] == f32::MAX {
            [0.0; 3]
        } else {
            minimum
        };
        let adjusted_origin = (
            origin.0 - minimum[0] * scale,
            origin.1 - minimum[1] * scale,
            origin.2 - minimum[2] * scale,
        );
        Ok(self.replace_placements(voxelize_mesh_f32(&mesh, adjusted_origin, scale)))
    }

    pub fn load_geometry_boxes(&mut self, boxes: &[Vec<u16>]) -> Result<usize, String> {
        self.world = WorldState::new();
        let mut count = 0;
        for bounds in boxes {
            if bounds.len() < 6 {
                return Err(
                    "Each box must have 6 values: [xmin,ymin,zmin,xmax,ymax,zmax]".to_owned(),
                );
            }
            for x in bounds[0]..=bounds[3] {
                for y in bounds[1]..=bounds[4] {
                    for z in bounds[2]..=bounds[5] {
                        self.world
                            .set_block((x, y, z), Block::default())
                            .map_err(|error| format!("{error:?}"))?;
                        count += 1;
                    }
                }
            }
        }
        Ok(count)
    }

    pub fn free_cells(&self) -> Vec<(u16, u16, u16)> {
        let mut cells = Vec::new();
        for x in 1..=self.grid_size {
            for y in 1..=self.grid_size {
                for z in 1..=self.grid_size {
                    if !self.world.is_filled((x, y, z)) {
                        cells.push((x, y, z));
                    }
                }
            }
        }
        cells
    }

    pub fn sample_box_observations(&self, positions: &[(u16, u16, u16)], radius: u32) -> Vec<f32> {
        let radius = radius as i32;
        let side = (2 * radius + 1) as usize;
        let mut result = Vec::with_capacity(positions.len() * side * side * side);
        for &(cx, cy, cz) in positions {
            for dx in -radius..=radius {
                for dy in -radius..=radius {
                    for dz in -radius..=radius {
                        let x = i32::from(cx) + dx;
                        let y = i32::from(cy) + dy;
                        let z = i32::from(cz) + dz;
                        let in_bounds = x >= 1
                            && y >= 1
                            && z >= 1
                            && x <= i32::from(self.grid_size)
                            && y <= i32::from(self.grid_size)
                            && z <= i32::from(self.grid_size);
                        result.push(if in_bounds {
                            self.world.is_filled((x as u16, y as u16, z as u16)) as u8 as f32
                        } else {
                            0.0
                        });
                    }
                }
            }
        }
        result
    }

    pub fn filled_count(&self) -> usize {
        self.world.iter_filled().count()
    }

    fn replace_placements(
        &mut self,
        placements: Vec<crate::world::ingest::BlockPlacement>,
    ) -> usize {
        self.world = WorldState::new();
        let mut count = 0;
        for placement in placements {
            let (x, y, z) = placement.coord;
            if (1..=self.grid_size).contains(&x)
                && (1..=self.grid_size).contains(&y)
                && (1..=self.grid_size).contains(&z)
            {
                let _ = self.world.set_block(placement.coord, placement.block);
                count += 1;
            }
        }
        count
    }
}

fn read_stl(path: &Path) -> Result<String, String> {
    let mut file = File::open(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)
        .map_err(|error| format!("{}: {error}", path.display()))?;
    Ok(contents)
}
