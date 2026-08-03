use pyo3::prelude::*;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyVoxelObservation {
    #[pyo3(get)]
    pub filled: usize,
    #[pyo3(get)]
    pub steps_remaining: u32,
    /// Manhattan distance to the active goal, or None if no goal is set.
    #[pyo3(get)]
    pub goal_distance: Option<u32>,
    /// Signed normalized delta from cursor to goal in x, y, z, or None if unset.
    #[pyo3(get)]
    pub goal_direction: Option<(f32, f32, f32)>,
    #[pyo3(get)]
    pub cursor_pos: (u16, u16, u16),
    #[pyo3(get)]
    pub local_grid: Option<Vec<f32>>,
}
