//! Multi-agent VoxelEnv: N agents share a single 32³ voxel grid.
//!
//! Each agent has its own cursor and (optionally) its own goal position.
//! Agents move simultaneously and cannot move into geometry or filled cells.
//! Agents may overlap each other (no inter-agent collision).
//! Trail mode: each successful move auto-fills the destination cell.
//! Episode ends when steps_remaining reaches 0, or all agents have reached
//! their goals (when geometry/goals are configured).

use crate::voxel::actions::OFFSETS_26;
use crate::voxel::rewards::RewardConfig;
use crate::voxel::world::{Coord, WorldState};

use super::geometry::{compute_surface_cells, l2, manhattan};

// ---------------------------------------------------------------------------
// Per-agent state
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AgentEntry {
    pub cursor: Coord,
    pub goal: Option<Coord>,
    pub prev_l2: f32,
    pub goal_reached: bool,
}

// ---------------------------------------------------------------------------
// MultiAgentVoxelEnv
// ---------------------------------------------------------------------------

pub struct MultiAgentVoxelEnv {
    pub world: WorldState,
    pub geometry: Vec<Coord>,
    pub geometry_len: usize,
    pub surface_cells: Vec<Coord>,
    pub trail_mode: bool,
    pub max_steps: u32,
    pub steps: u32,
    pub reward_config: RewardConfig,
    pub agents: Vec<AgentEntry>,
    pub agent_count: usize,
    /// Side length of the cubic grid (coords in [1, grid_size]³). Default 32.
    pub grid_size: u16,
}

pub struct MultiStepResult {
    pub steps_remaining: u32,
    pub voxel_count: usize,
    pub cursors: Vec<Coord>,
    pub goal_distances: Vec<Option<u32>>,
    pub rewards: Vec<f32>,
    pub done: bool,
}

impl MultiAgentVoxelEnv {
    pub fn new(
        agent_count: usize,
        max_steps: u32,
        trail_mode: bool,
        geometry: Vec<Coord>,
        reward_config: RewardConfig,
        grid_size: u16,
    ) -> Self {
        let mut world = WorldState::new();
        for &coord in &geometry {
            world.set(coord, true);
        }
        let geometry_len = geometry.len();
        let surface_cells = compute_surface_cells(&geometry, grid_size);
        let agents = (0..agent_count)
            .map(|_| AgentEntry {
                cursor: (1, 1, 1),
                goal: None,
                prev_l2: 0.0,
                goal_reached: false,
            })
            .collect();
        Self {
            world,
            geometry,
            geometry_len,
            surface_cells,
            trail_mode,
            max_steps,
            steps: 0,
            reward_config,
            agents,
            agent_count,
            grid_size,
        }
    }

    pub fn agent_count(&self) -> usize {
        self.agent_count
    }

    /// Replace geometry in-place without reinstantiating the env.
    /// Clears all filled cells (geometry + trail), fills the new geometry, and
    /// recomputes surface cells. Does not reset steps or agent positions — call
    /// reset() after set_geometry() to start a new episode.
    pub fn set_geometry(&mut self, geometry: Vec<Coord>) {
        self.world.clear();
        for &coord in &geometry {
            self.world.set(coord, true);
        }
        self.geometry_len = geometry.len();
        self.surface_cells = compute_surface_cells(&geometry, self.grid_size);
        self.geometry = geometry;
    }

    /// Total agent-filled cells (excludes geometry).
    fn voxel_count(&self) -> usize {
        self.world.len().saturating_sub(self.geometry_len)
    }

    pub fn reset(&mut self, seed: u64) -> (u32, usize, Vec<Coord>, Vec<Option<u32>>) {
        self.steps = 0;
        self.world.clear();
        for &coord in &self.geometry {
            self.world.set(coord, true);
        }

        // Assign each agent a distinct start position (and goal if geometry available).
        let n_surface = self.surface_cells.len();
        for (i, agent) in self.agents.iter_mut().enumerate() {
            let (start, goal) = if n_surface >= 2 {
                // Spread agents across surface cells using offset seeds.
                let seed_i = seed.wrapping_add(i as u64 * 7919);
                let seed_g = seed_i
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let idx_s = (seed_i as usize).wrapping_mul(2654435761) % n_surface;
                let idx_g_raw = (seed_g as usize).wrapping_mul(2654435761) % n_surface;
                let idx_g = if idx_g_raw == idx_s {
                    (idx_s + n_surface / 2) % n_surface
                } else {
                    idx_g_raw
                };
                let s = self.surface_cells[idx_s];
                let g = self.surface_cells[idx_g];
                let (s, g) = if (seed + i as u64) % 2 == 0 {
                    (s, g)
                } else {
                    (g, s)
                };
                (s, Some(g))
            } else {
                // No geometry — spread agents across the grid with fixed pattern.
                let span = (self.grid_size as usize).saturating_sub(1).max(1);
                let x = ((i * 8) % span + 1) as u16;
                let y = ((i * 5) % span + 1) as u16;
                ((x, y, 1), None)
            };

            agent.cursor = start;
            agent.goal = if goal != Some(start) { goal } else { None };
            agent.prev_l2 = agent.goal.map_or(0.0, |g| l2(start, g));
            agent.goal_reached = false;
        }

        let cursors: Vec<Coord> = self.agents.iter().map(|a| a.cursor).collect();
        let goal_distances: Vec<Option<u32>> = self
            .agents
            .iter()
            .map(|a| a.goal.map(|g| manhattan(a.cursor, g)))
            .collect();

        (self.max_steps, self.voxel_count(), cursors, goal_distances)
    }

    /// Process one simultaneous step for all agents.
    /// `actions` must have length == agent_count; out-of-range entries treated as collision.
    pub fn step_all(&mut self, actions: &[i32]) -> MultiStepResult {
        self.steps += 1;
        let mut rewards = vec![0.0f32; self.agent_count];

        for (i, agent) in self.agents.iter_mut().enumerate() {
            if agent.goal_reached {
                // Done agents get 0 reward and don't move.
                continue;
            }

            let action = actions.get(i).copied().unwrap_or(-1);
            let (x, y, z) = agent.cursor;

            let g = self.grid_size as i32;
            let dest = if (action as usize) < 26 {
                let (dx, dy, dz) = OFFSETS_26[action as usize];
                let nx = (x as i32 + dx).clamp(1, g) as u16;
                let ny = (y as i32 + dy).clamp(1, g) as u16;
                let nz = (z as i32 + dz).clamp(1, g) as u16;
                (nx, ny, nz)
            } else {
                (x, y, z)
            };

            let is_noop = action == 26;
            let moved = !is_noop && dest != (x, y, z) && !self.world.is_filled(dest);
            let mut step_reward = 0.0f32;

            if moved {
                agent.cursor = dest;
                if self.trail_mode {
                    self.world.set(dest, true);
                }
            } else if !is_noop && dest == (x, y, z) {
                // Boundary hit — collision_cost is negative, so add it as a penalty.
                step_reward += self.reward_config.collision_cost;
            }
            // Occupied-cell collision: no extra penalty (cursor stays, step consumed).

            // Goal reward + distance shaping.
            if let Some(goal) = agent.goal {
                let new_l2 = l2(agent.cursor, goal);
                step_reward +=
                    self.reward_config
                        .base_step_reward(agent.prev_l2, new_l2, self.grid_size);
                agent.prev_l2 = new_l2;

                if agent.cursor == goal {
                    step_reward += self.reward_config.goal_reward;
                    agent.goal_reached = true;
                }
            } else {
                step_reward += self.reward_config.step_cost;
            }

            rewards[i] = step_reward;
        }

        // Done when all agents with goals have reached them, or steps exhausted.
        let any_have_goal = self.agents.iter().any(|a| a.goal.is_some());
        let goals_done = any_have_goal
            && self
                .agents
                .iter()
                .all(|a| a.goal.is_none() || a.goal_reached);
        let all_done = goals_done || self.steps >= self.max_steps;

        let cursors: Vec<Coord> = self.agents.iter().map(|a| a.cursor).collect();
        let goal_distances: Vec<Option<u32>> = self
            .agents
            .iter()
            .map(|a| a.goal.map(|g| manhattan(a.cursor, g)))
            .collect();

        MultiStepResult {
            steps_remaining: self.max_steps.saturating_sub(self.steps),
            voxel_count: self.voxel_count(),
            cursors,
            goal_distances,
            rewards,
            done: all_done,
        }
    }
}
