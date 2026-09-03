use pyo3::prelude::*;

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PyAgentState {
    #[pyo3(get)]
    pub current: (u16, u16, u16),
    #[pyo3(get)]
    pub target: (u16, u16, u16),
    #[pyo3(get)]
    pub reached: bool,
}

#[pyclass(from_py_object)]
#[derive(Clone)]
pub struct PySurfaceObservation {
    #[pyo3(get)]
    pub steps_remaining: u32,
    #[pyo3(get)]
    pub reached_agents: usize,
    #[pyo3(get)]
    pub total_agents: usize,
    #[pyo3(get)]
    pub agents: Vec<PyAgentState>,
}

/// Python-visible step result returned by PySurfaceEnv.step().
#[pyclass]
pub struct PyStepResultSurface {
    pub(crate) obs: PySurfaceObservation,
    #[pyo3(get)]
    pub reward: f32,
    #[pyo3(get)]
    pub done: bool,
}

#[pymethods]
impl PyStepResultSurface {
    #[getter]
    fn observation(&self) -> PySurfaceObservation {
        self.obs.clone()
    }
}
