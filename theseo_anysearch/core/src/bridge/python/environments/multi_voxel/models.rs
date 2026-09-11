use pyo3::prelude::*;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyMultiVoxelObs {
    #[pyo3(get)]
    pub steps_remaining: u32,
    #[pyo3(get)]
    pub voxel_count: usize,
    #[pyo3(get)]
    pub cursors: Vec<(u16, u16, u16)>,
    #[pyo3(get)]
    pub goal_distances: Vec<Option<u32>>,
}

/// Step result returned by PyMultiVoxelEnv.step().
#[pyclass]
pub struct PyMultiVoxelStepResult {
    #[pyo3(get)]
    pub observation: PyMultiVoxelObs,
    #[pyo3(get)]
    pub rewards: Vec<f32>,
    #[pyo3(get)]
    pub done: bool,
}
