use std::collections::HashMap;

use pyo3::prelude::*;

use super::observation::PyVoxelObservation;

#[pyclass]
pub struct PyStepResultVoxel {
    pub(crate) obs: PyVoxelObservation,
    #[pyo3(get)]
    pub reward: f32,
    #[pyo3(get)]
    pub done: bool,
    #[pyo3(get)]
    pub terminated: bool,
    #[pyo3(get)]
    pub truncated: bool,
    #[pyo3(get)]
    pub collision: bool,
    #[pyo3(get)]
    pub goal_reached: bool,
    #[pyo3(get)]
    pub consecutive_collisions: u32,
    #[pyo3(get)]
    pub termination_reason: String,
    #[pyo3(get)]
    pub reward_breakdown: HashMap<String, f32>,
    #[pyo3(get)]
    pub goal_distance_l2: f32,
    #[pyo3(get)]
    pub construction_residual: usize,
    #[pyo3(get)]
    pub construction_overshoot: usize,
}

#[pymethods]
impl PyStepResultVoxel {
    #[getter]
    fn observation(&self) -> PyVoxelObservation {
        self.obs.clone()
    }
}
