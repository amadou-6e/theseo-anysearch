use std::path::Path;

use pyo3::prelude::*;

use crate::sampling::VoxelSampler;

use super::errors;

/// Python adapter for the reusable Rust voxel sampler.
#[pyclass]
pub struct PyVoxelSampler {
    inner: VoxelSampler,
}

#[pymethods]
impl PyVoxelSampler {
    #[new]
    #[pyo3(signature = (grid_size=32))]
    pub fn new(grid_size: u16) -> Self {
        Self {
            inner: VoxelSampler::new(grid_size),
        }
    }

    pub fn load_stl(&mut self, path: String, scale: f32) -> PyResult<usize> {
        self.inner
            .load_stl(Path::new(&path), scale)
            .map_err(errors::io)
    }

    pub fn load_stl_normalized(
        &mut self,
        path: String,
        scale: f32,
        origin_x: f32,
        origin_y: f32,
        origin_z: f32,
    ) -> PyResult<usize> {
        self.inner
            .load_stl_normalized(Path::new(&path), scale, (origin_x, origin_y, origin_z))
            .map_err(errors::runtime)
    }

    pub fn load_geometry_boxes(&mut self, boxes: Vec<Vec<u16>>) -> PyResult<usize> {
        self.inner
            .load_geometry_boxes(&boxes)
            .map_err(errors::invalid_value)
    }

    pub fn free_cells(&self) -> Vec<(u16, u16, u16)> {
        self.inner.free_cells()
    }

    pub fn sample_box_obs(&self, positions: Vec<(u16, u16, u16)>, radius: u32) -> Vec<f32> {
        self.inner.sample_box_observations(&positions, radius)
    }

    pub fn filled_count(&self) -> usize {
        self.inner.filled_count()
    }
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyVoxelSampler>()
}
