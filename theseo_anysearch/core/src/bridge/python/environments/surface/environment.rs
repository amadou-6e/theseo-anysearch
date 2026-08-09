use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{
    surface::{SurfaceAction, SurfaceEnv},
    voxel::world::{
        ingest::{parse_ascii_stl, voxelize_mesh},
        Coord,
    },
};

use super::models::{PyAgentState, PyStepResultSurface, PySurfaceObservation};

#[pyclass]
pub struct PySurfaceEnv {
    inner: SurfaceEnv,
}

#[pymethods]
impl PySurfaceEnv {
    /// Build from an explicit list of surface voxel coordinates.
    #[staticmethod]
    pub fn from_surface_coords(
        coords: Vec<(u16, u16, u16)>,
        agent_count: usize,
        max_steps: u32,
    ) -> Self {
        Self {
            inner: SurfaceEnv::from_surface(&coords, agent_count, max_steps),
        }
    }

    /// Build by voxelising an ASCII STL file and computing the exterior shell.
    #[staticmethod]
    pub fn from_stl(
        stl_ascii: &str,
        origin_x: u16,
        origin_y: u16,
        origin_z: u16,
        scale: f32,
        agent_count: usize,
        max_steps: u32,
    ) -> PyResult<Self> {
        let mesh = parse_ascii_stl(stl_ascii)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
        let (placements, dropped) = voxelize_mesh(&mesh, (origin_x, origin_y, origin_z), scale);
        if dropped > 0 {
            eprintln!(
                "warning: {dropped} point(s) fell outside world bounds during STL voxelization and were dropped"
            );
        }
        let filled: Vec<Coord> = placements.iter().map(|p| p.coord).collect();
        Ok(Self {
            inner: SurfaceEnv::from_filled_surface(&filled, agent_count, max_steps),
        })
    }

    /// Reset the environment and return the initial observation.
    pub fn reset(&mut self, seed: u64) -> PyResult<PySurfaceObservation> {
        let obs = self.inner.reset(seed).map_err(PyValueError::new_err)?;
        Ok(py_surface_obs(obs))
    }

    /// Step the environment (StepAll â€” `action` is ignored).
    pub fn step(&mut self, _action: i32) -> PyStepResultSurface {
        let result = self.inner.step(SurfaceAction::StepAll);
        PyStepResultSurface {
            obs: py_surface_obs(result.observation),
            reward: result.reward,
            done: result.done,
        }
    }
}

fn py_surface_obs(obs: crate::surface::SurfaceObservation) -> PySurfaceObservation {
    PySurfaceObservation {
        steps_remaining: obs.steps_remaining,
        reached_agents: obs.reached_agents,
        total_agents: obs.total_agents,
        agents: obs
            .agents
            .iter()
            .map(|a| PyAgentState {
                current: a.current,
                target: a.target,
                reached: a.reached,
            })
            .collect(),
    }
}
