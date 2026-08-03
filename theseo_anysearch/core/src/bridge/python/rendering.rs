use std::path::Path;

use pyo3::prelude::*;

use crate::{
    voxel::rendering::{
        render_episode_gif, run_episode_trace, EpisodeStep, EpisodeTrace, EpochSummary,
        TrainingSummary,
    },
    voxel::world::ingest::{parse_ascii_stl, voxelize_mesh},
};

#[pyfunction]
pub fn py_surface_env_trace(
    stl_ascii: String,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    scale: f32,
    agent_count: usize,
    max_steps: u32,
    seed: u64,
) -> PyResult<String> {
    let payload = run_episode_trace(
        &stl_ascii,
        origin_x,
        origin_y,
        origin_z,
        scale,
        agent_count,
        max_steps,
        seed,
    )
    .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    serde_json::to_string(&payload)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("serialize error: {e}")))
}

#[pyfunction]
pub fn py_train_videos(
    stl_ascii: String,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    scale: f32,
    agent_count: usize,
    max_steps: u32,
    seed: u64,
    iterations: u32,
    video_every: u32,
    output_dir: String,
    camera_yaw_deg: f32,
    camera_pitch_deg: f32,
    disable_culling: bool,
) -> PyResult<String> {
    let out_dir = Path::new(&output_dir);
    std::fs::create_dir_all(out_dir)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("mkdir failed: {e}")))?;

    let mut summary = Vec::new();
    for epoch in 1..=iterations {
        let trace = run_episode_trace(
            &stl_ascii,
            origin_x,
            origin_y,
            origin_z,
            scale,
            agent_count,
            max_steps,
            seed + epoch as u64,
        )
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

        let painted_voxels = trace
            .trace
            .iter()
            .map(|s| s.new_paints.len())
            .sum::<usize>();
        summary.push(EpochSummary {
            epoch,
            reached: trace.reached_agents,
            total_agents: trace.total_agents,
            steps: trace.steps_executed,
            reward: trace.total_reward,
            painted_voxels,
        });

        if video_every > 0 && epoch % video_every == 0 {
            let file = out_dir.join(format!("epoch_{epoch:04}.gif"));
            render_episode_gif(
                &trace,
                &file,
                camera_yaw_deg,
                camera_pitch_deg,
                1.1,
                disable_culling,
            )
            .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
        }
    }

    let summary_file = out_dir.join("training_summary_rust.json");
    let payload = serde_json::to_string_pretty(&summary).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("summary serialize failed: {e}"))
    })?;
    std::fs::write(&summary_file, payload).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("summary write failed: {e}"))
    })?;

    let response = TrainingSummary {
        status: "ok".to_string(),
        iterations,
        video_every,
        output_dir,
        summary_file: summary_file.to_string_lossy().to_string(),
    };
    serde_json::to_string(&response).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("response serialize failed: {e}"))
    })
}

/// Render voxelized STL geometry (no agents) to a GIF for visual debugging.
#[pyfunction]
pub fn py_render_stl(
    stl_ascii: String,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    scale: f32,
    output_path: String,
    camera_yaw_deg: f32,
    camera_pitch_deg: f32,
) -> PyResult<String> {
    let mesh = parse_ascii_stl(&stl_ascii)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:?}")))?;
    let placements = voxelize_mesh(&mesh, (origin_x, origin_y, origin_z), scale);
    let geometry_coords: Vec<(u16, u16, u16)> = placements.iter().map(|p| p.coord).collect();
    let trace = EpisodeTrace {
        triangles: mesh.triangles.len(),
        filled: geometry_coords.len(),
        surface: 0,
        total_agents: 0,
        max_steps: 1,
        steps_executed: 1,
        reached_agents: 0,
        total_reward: 0.0,
        geometry_coords,
        surface_coords: vec![],
        trace: vec![EpisodeStep {
            step: 0,
            agents: vec![],
            reached_agents: 0,
            new_paints: vec![],
        }],
    };
    let path = std::path::Path::new(&output_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("mkdir failed: {e}")))?;
    }
    render_episode_gif(&trace, path, camera_yaw_deg, camera_pitch_deg, 1.1, false)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(format!(
        "stl {} triangles={} voxels={} yaw={} pitch={} -> {}",
        output_path,
        mesh.triangles.len(),
        trace.filled,
        camera_yaw_deg,
        camera_pitch_deg,
        output_path
    ))
}

// ---------------------------------------------------------------------------
#[pyfunction]
pub fn py_render_cube(
    output_path: String,
    cube_size: u16,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    camera_yaw_deg: f32,
    camera_pitch_deg: f32,
) -> PyResult<String> {
    let mut geometry_coords = Vec::new();
    for dz in 0..cube_size {
        for dy in 0..cube_size {
            for dx in 0..cube_size {
                geometry_coords.push((origin_x + dx, origin_y + dy, origin_z + dz));
            }
        }
    }
    let trace = EpisodeTrace {
        triangles: 0,
        filled: geometry_coords.len(),
        surface: 0,
        total_agents: 0,
        max_steps: 1,
        steps_executed: 1,
        reached_agents: 0,
        total_reward: 0.0,
        geometry_coords,
        surface_coords: vec![],
        trace: vec![EpisodeStep {
            step: 0,
            agents: vec![],
            reached_agents: 0,
            new_paints: vec![],
        }],
    };
    let path = std::path::Path::new(&output_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("mkdir failed: {e}")))?;
    }
    render_episode_gif(&trace, path, camera_yaw_deg, camera_pitch_deg, 3.0, false)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    Ok(format!(
        "cube {}^3 at ({},{},{}) yaw={} pitch={} -> {}",
        cube_size, origin_x, origin_y, origin_z, camera_yaw_deg, camera_pitch_deg, output_path
    ))
}

/// Lightweight geometry sampler for garden pre-training data collection.
///
/// Holds a voxelized world (no agents, no episode state). Exposes:
///   - load_stl / load_geometry_boxes  â€” build the world
///   - free_cells                       â€” list positions the agent could stand
///   - sample_box_obs                   â€” batch box observations at given positions
///
/// This avoids the per-step overhead of PyMultiVoxelEnv, giving 10-50Ã— faster
/// data collection for garden pre-training.
pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_surface_env_trace, module)?)?;
    module.add_function(wrap_pyfunction!(py_train_videos, module)?)?;
    module.add_function(wrap_pyfunction!(py_render_stl, module)?)?;
    module.add_function(wrap_pyfunction!(py_render_cube, module)?)?;
    Ok(())
}
