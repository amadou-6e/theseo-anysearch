use pyo3::prelude::*;

use crate::{
    bridge::dto::{EnvConfig, GeometrySubmission},
    environments::Environment,
    surface::{SurfaceAction, SurfaceEnv},
    voxel::world::{
        ingest::{parse_ascii_stl, voxelize_mesh},
        Block, BlockUpdate, World, WorldState,
    },
    voxel::{VoxelAction, VoxelEnv},
};

pub fn run_submission_poc(
    submission: &GeometrySubmission,
    env_config: &EnvConfig,
) -> Result<String, String> {
    let mesh = parse_ascii_stl(&submission.stl_ascii).map_err(|e| format!("{e:?}"))?;
    let placements = voxelize_mesh(&mesh, submission.origin, submission.scale);

    let mut world = WorldState::new();
    for placement in &placements {
        world
            .set_block(placement.coord, placement.block.clone())
            .map_err(|e| format!("{e:?}"))?;
    }

    let mut env = VoxelEnv::new(world, env_config.max_steps);
    let _ = env.reset(42);
    let first = env.step(VoxelAction::Noop);

    Ok(format!(
        "submission={} triangles={} placements={} first_reward={:.3} filled={}",
        submission.name,
        mesh.triangles.len(),
        placements.len(),
        first.reward,
        first.observation.filled
    ))
}

#[pyfunction]
pub fn py_submit_geometry(
    name: String,
    stl_ascii: String,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    scale: f32,
    max_steps: u32,
) -> PyResult<String> {
    let submission = GeometrySubmission {
        name,
        stl_ascii,
        origin: (origin_x, origin_y, origin_z),
        scale,
    };
    let env_config = EnvConfig { max_steps };
    run_submission_poc(&submission, &env_config)
        .map_err(|msg| pyo3::exceptions::PyRuntimeError::new_err(msg))
}

#[pyfunction]
pub fn py_world_smoke_test() -> PyResult<String> {
    let mut world = WorldState::new();

    world
        .set_block((1, 2, 3), Block::default())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;

    world
        .update_block(
            (1, 2, 3),
            BlockUpdate {
                kind: Some(7),
                active: Some(false),
                reward_weight: Some(3.0),
            },
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;

    world
        .remove_block((1, 2, 3))
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;

    Ok(format!("world_interface_poc_ok filled={}", world.len()))
}

#[pyfunction]
pub fn py_surface_env_from_stl(
    stl_ascii: String,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    scale: f32,
    agent_count: usize,
    max_steps: u32,
    seed: u64,
) -> PyResult<String> {
    let mesh = parse_ascii_stl(&stl_ascii)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
    let placements = voxelize_mesh(&mesh, (origin_x, origin_y, origin_z), scale);
    let filled = placements.iter().map(|p| p.coord).collect::<Vec<_>>();

    let mut env = SurfaceEnv::from_filled_surface(&filled, agent_count, max_steps);
    let mut obs = env.reset(seed);
    let mut total_reward = 0.0f32;
    let mut steps = 0u32;

    while steps < max_steps && obs.reached_agents < obs.total_agents {
        let sr = env.step(SurfaceAction::StepAll);
        total_reward += sr.reward;
        obs = sr.observation;
        steps += 1;
        if sr.done {
            break;
        }
    }

    Ok(format!(
        "triangles={} filled={} surface={} agents={} reached={}/{} steps={} reward={:.3}",
        mesh.triangles.len(),
        filled.len(),
        env.surface_count(),
        obs.total_agents,
        obs.reached_agents,
        obs.total_agents,
        steps,
        total_reward
    ))
}

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_submit_geometry, module)?)?;
    module.add_function(wrap_pyfunction!(py_world_smoke_test, module)?)?;
    module.add_function(wrap_pyfunction!(py_surface_env_from_stl, module)?)?;
    Ok(())
}
