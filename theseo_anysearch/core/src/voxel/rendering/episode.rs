use std::collections::HashSet;

use serde::Serialize;

use crate::{
    environments::Environment,
    surface::{SurfaceAction, SurfaceEnv},
    voxel::world::ingest::{parse_ascii_stl, voxelize_mesh},
};

#[derive(Clone, Serialize)]
pub(crate) struct EpisodeStep {
    pub(crate) step: u32,
    pub(crate) agents: Vec<crate::surface::AgentState>,
    pub(crate) reached_agents: usize,
    pub(crate) new_paints: Vec<PaintVoxel>,
}

#[derive(Clone, Serialize)]
pub(crate) struct PaintVoxel {
    pub(crate) coord: (u16, u16, u16),
    pub(crate) owner: usize,
}

#[derive(Clone, Serialize)]
pub(crate) struct EpisodeTrace {
    pub(crate) triangles: usize,
    pub(crate) filled: usize,
    pub(crate) surface: usize,
    pub(crate) total_agents: usize,
    pub(crate) max_steps: u32,
    pub(crate) steps_executed: u32,
    pub(crate) reached_agents: usize,
    pub(crate) total_reward: f32,
    pub(crate) geometry_coords: Vec<(u16, u16, u16)>,
    pub(crate) surface_coords: Vec<(u16, u16, u16)>,
    pub(crate) trace: Vec<EpisodeStep>,
}

#[derive(Serialize)]
pub(crate) struct EpochSummary {
    pub(crate) epoch: u32,
    pub(crate) reached: usize,
    pub(crate) total_agents: usize,
    pub(crate) steps: u32,
    pub(crate) reward: f32,
    pub(crate) painted_voxels: usize,
}

#[derive(Serialize)]
pub(crate) struct TrainingSummary {
    pub(crate) status: String,
    pub(crate) iterations: u32,
    pub(crate) video_every: u32,
    pub(crate) output_dir: String,
    pub(crate) summary_file: String,
}

pub(crate) fn run_episode_trace(
    stl_ascii: &str,
    origin_x: u16,
    origin_y: u16,
    origin_z: u16,
    scale: f32,
    agent_count: usize,
    max_steps: u32,
    seed: u64,
) -> Result<EpisodeTrace, String> {
    let mesh = parse_ascii_stl(stl_ascii).map_err(|e| format!("{e:?}"))?;
    let (placements, dropped) = voxelize_mesh(&mesh, (origin_x, origin_y, origin_z), scale);
    if dropped > 0 {
        eprintln!(
            "warning: {dropped} point(s) fell outside world bounds during STL voxelization and were dropped"
        );
    }
    let filled = placements.iter().map(|p| p.coord).collect::<Vec<_>>();

    let mut env = SurfaceEnv::from_filled_surface(&filled, agent_count, max_steps);
    let mut obs = env.reset(seed);
    let mut total_reward = 0.0f32;
    let mut steps = 0u32;
    let mut seen_painted = HashSet::new();
    let mut first_paints = Vec::new();
    for (coord, owner) in env.trail_owner() {
        seen_painted.insert(*coord);
        first_paints.push(PaintVoxel {
            coord: *coord,
            owner: *owner,
        });
    }
    let mut trace = vec![EpisodeStep {
        step: 0,
        agents: obs.agents.clone(),
        reached_agents: obs.reached_agents,
        new_paints: first_paints,
    }];

    while steps < max_steps && obs.reached_agents < obs.total_agents {
        let sr = env.step(SurfaceAction::StepAll);
        total_reward += sr.reward;
        steps += 1;
        obs = sr.observation;
        let mut new_paints = Vec::new();
        for (coord, owner) in env.trail_owner() {
            if seen_painted.insert(*coord) {
                new_paints.push(PaintVoxel {
                    coord: *coord,
                    owner: *owner,
                });
            }
        }
        trace.push(EpisodeStep {
            step: steps,
            agents: obs.agents.clone(),
            reached_agents: obs.reached_agents,
            new_paints,
        });
        if sr.done {
            break;
        }
    }

    Ok(EpisodeTrace {
        triangles: mesh.triangles.len(),
        filled: filled.len(),
        surface: env.surface_count(),
        total_agents: obs.total_agents,
        max_steps,
        steps_executed: steps,
        reached_agents: obs.reached_agents,
        total_reward,
        geometry_coords: filled,
        surface_coords: env.surface_coords(),
        trace,
    })
}
