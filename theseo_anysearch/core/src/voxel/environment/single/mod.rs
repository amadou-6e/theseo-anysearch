use crate::voxel::rewards::RewardConfig;
use crate::voxel::world::{
    Block, Coord, World, WorldRead, WorldState, BLOCK_KIND_GOAL, BLOCK_KIND_START,
};

use crate::environments::{Environment, StepResult};
use crate::voxel::{
    actions::{
        ActionExtensionSpec, ActionHistoryEntryV2, ConfiguredOutcome, ConfiguredPredicate,
        PendingMutations, OFFSETS_26,
    },
    common::ABI_VERSION,
    outcomes::{
        builtins as outcome_builtins, NativeOutcomeExtension, OutcomeContextV2, OutcomeResultV2,
    },
    predicates::{builtins as predicate_builtins, NativePredicateExtension, PredicateContextV2},
    rewards::{builtins as reward_components, NativeRewardExtension, RewardContextV2},
};
use std::collections::{HashMap, HashSet};
use std::path::Path;
// ---------------------------------------------------------------------------
// Action / Observation types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum VoxelAction {
    Place(Coord),
    Remove(Coord),
    Noop,
    /// Movement was blocked (boundary hit or occupied cell). Used by py_bindings
    /// to trigger the collision_cost penalty without moving the cursor.
    Collision,
}

#[derive(Clone, Debug)]
pub struct VoxelObservation {
    /// Number of agent-filled cells (excludes geometry obstacles).
    pub filled: usize,
    pub steps_remaining: u32,
    /// Manhattan distance to the active goal, if one is set.
    pub goal_distance: Option<u32>,
}

// ---------------------------------------------------------------------------
// VoxelEnv
// ---------------------------------------------------------------------------

pub struct VoxelEnv {
    world: WorldState,
    geometry_len: usize,
    agent_filled_count: usize,
    /// Cells 6-adjacent to geometry but not in geometry — valid start/goal positions.
    surface_cells: Vec<Coord>,
    /// When true, movement actions auto-fill the destination cell.
    trail_mode: bool,
    max_steps: u32,
    steps: u32,
    segment_steps: u32,
    segment_length: u32,
    /// Side length of the cubic grid (coords in [1, grid_size]³). Default 32.
    pub grid_size: u16,
    /// Independent one-based task-coordinate bounds for x, y, and z.
    pub extent: [u16; 3],
    // --- Navigation state ---
    /// Current agent cursor position in [1, grid_size]³.
    cursor: Coord,
    /// Fixed start/goal from waypoints file or editor (overrides random selection).
    fixed_start: Option<Coord>,
    fixed_goal: Option<Coord>,
    /// Active goal for the current episode.
    active_goal: Option<Coord>,
    /// Optional task-defined success targets. Empty means use the active waypoint.
    success_targets: Vec<Coord>,
    goal_tolerance: f32,
    /// L2 (Euclidean) distance from cursor to goal at the end of the previous step
    /// (used for potential-based shaping).
    prev_goal_dist_l2: f32,
    reward_config: RewardConfig,
    max_consecutive_collisions: Option<u32>,
    terminate_on_success: bool,
    consecutive_collisions: u32,
    construction_target: HashSet<Coord>,
    native_reward: Option<NativeRewardExtension>,
    action_predicates: Vec<ConfiguredPredicate>,
    action_outcomes: Vec<ConfiguredOutcome>,
    action_history: Vec<ActionHistoryEntryV2>,
    action_history_length: usize,
    pending_action_feasible: bool,
    pending_action_index: i32,
    pending_previous_cursor: Coord,
    pending_invalid_action: bool,
    last_reward_breakdown: HashMap<String, f32>,
    last_collision: bool,
    last_goal_reached: bool,
    last_terminated: bool,
    last_truncated: bool,
    last_termination_reason: String,
    last_reward_error: Option<String>,
    last_goal_distance_l2: f32,
    last_construction_residual: usize,
    last_construction_overshoot: usize,
}

impl VoxelEnv {
    pub fn new(world: WorldState, max_steps: u32) -> Self {
        Self {
            world,
            geometry_len: 0,
            agent_filled_count: 0,
            surface_cells: Vec::new(),
            trail_mode: false,
            max_steps,
            steps: 0,
            segment_steps: 0,
            segment_length: 0,
            grid_size: 32,
            extent: [32, 32, 32],
            cursor: (1, 1, 1),
            fixed_start: None,
            fixed_goal: None,
            active_goal: None,
            success_targets: Vec::new(),
            goal_tolerance: 0.0,
            prev_goal_dist_l2: 0.0,
            reward_config: RewardConfig::default(),
            max_consecutive_collisions: None,
            terminate_on_success: true,
            consecutive_collisions: 0,
            construction_target: HashSet::new(),
            native_reward: None,
            action_predicates: vec![
                ConfiguredPredicate::ValidAction,
                ConfiguredPredicate::Bounds,
                ConfiguredPredicate::Unoccupied,
            ],
            action_outcomes: vec![ConfiguredOutcome::CursorMovement],
            action_history: Vec::new(),
            action_history_length: 16,
            pending_action_feasible: true,
            pending_action_index: 26,
            pending_previous_cursor: (1, 1, 1),
            pending_invalid_action: false,
            last_reward_breakdown: HashMap::new(),
            last_collision: false,
            last_goal_reached: false,
            last_terminated: false,
            last_truncated: false,
            last_termination_reason: "in_progress".to_owned(),
            last_reward_error: None,
            last_goal_distance_l2: 0.0,
            last_construction_residual: 0,
            last_construction_overshoot: 0,
        }
    }

    /// Set the grid side length (coords in [1, grid_size]³). Must be called before with_geometry.
    pub fn with_grid_size(mut self, grid_size: u16) -> Self {
        self.grid_size = grid_size;
        self.extent = [grid_size; 3];
        self
    }

    pub fn with_extent(mut self, extent: [u16; 3]) -> Self {
        self.extent = extent;
        self.grid_size = *extent.iter().max().expect("extent has three axes");
        self
    }

    /// Pre-fill geometry obstacle cells. Computes surface cells automatically.
    pub fn with_geometry(mut self, geometry: Vec<Coord>) -> Self {
        self.world
            .replace_base_blocks(geometry.iter().copied().map(|coord| {
                (
                    coord,
                    Block {
                        kind: crate::voxel::world::BLOCK_KIND_OCCUPIED,
                        active: true,
                        reward_weight: 0.0,
                    },
                )
            }));
        self.geometry_len = geometry.len();
        self.surface_cells = compute_surface_cells(&geometry, self.extent);
        self
    }

    /// When trail_mode is true, successful movement auto-fills the destination.
    pub fn with_trail_mode(mut self, trail_mode: bool) -> Self {
        self.trail_mode = trail_mode;
        self.action_outcomes = if trail_mode {
            vec![
                ConfiguredOutcome::CursorMovement,
                ConfiguredOutcome::TrailPlacement,
            ]
        } else {
            vec![ConfiguredOutcome::CursorMovement]
        };
        self
    }

    /// Set reward parameters.
    pub fn with_reward_config(mut self, config: RewardConfig) -> Self {
        self.reward_config = config;
        self
    }

    pub fn with_terminate_on_success(mut self, value: bool) -> Self {
        self.terminate_on_success = value;
        self
    }
    pub fn with_success_contract(mut self, targets: Vec<Coord>, tolerance: f32) -> Self {
        self.success_targets = targets;
        self.goal_tolerance = tolerance;
        self
    }

    pub fn with_max_consecutive_collisions(mut self, value: Option<u32>) -> Self {
        self.max_consecutive_collisions = value;
        self
    }

    pub fn with_construction_target(mut self, target: Vec<Coord>) -> Self {
        self.construction_target = target.into_iter().collect();
        self
    }

    pub fn with_native_reward(
        mut self,
        path: &Path,
        name: &str,
        parameters_json: String,
    ) -> Result<Self, String> {
        self.native_reward = Some(NativeRewardExtension::load(path, name, parameters_json)?);
        Ok(self)
    }

    pub fn last_reward_breakdown(&self) -> &HashMap<String, f32> {
        &self.last_reward_breakdown
    }
    pub fn last_collision(&self) -> bool {
        self.last_collision
    }
    pub fn last_goal_reached(&self) -> bool {
        self.last_goal_reached
    }
    pub fn last_terminated(&self) -> bool {
        self.last_terminated
    }
    pub fn last_truncated(&self) -> bool {
        self.last_truncated
    }
    pub fn last_termination_reason(&self) -> &str {
        &self.last_termination_reason
    }
    pub fn consecutive_collisions(&self) -> u32 {
        self.consecutive_collisions
    }
    pub fn take_reward_error(&mut self) -> Option<String> {
        self.last_reward_error.take()
    }
    pub fn last_goal_distance_l2(&self) -> f32 {
        self.last_goal_distance_l2
    }
    pub fn last_construction_residual(&self) -> usize {
        self.last_construction_residual
    }
    pub fn last_construction_overshoot(&self) -> usize {
        self.last_construction_overshoot
    }
    /// Fix start and goal positions (overrides random selection on reset).
    pub fn set_waypoints(&mut self, start: Coord, goal: Coord) {
        self.set_waypoints_with_segment_length(start, goal, 0);
    }

    pub fn set_waypoints_with_segment_length(
        &mut self,
        start: Coord,
        goal: Coord,
        segment_length: u32,
    ) {
        self.fixed_start = Some(start);
        self.fixed_goal = Some(goal);
        self.segment_steps = 0;
        self.segment_length = segment_length;
    }

    pub fn set_active_goal_with_segment_length(
        &mut self,
        goal: Coord,
        segment_length: u32,
    ) -> VoxelObservation {
        self.active_goal = Some(goal);
        self.segment_steps = 0;
        self.segment_length = segment_length;
        self.prev_goal_dist_l2 = l2(self.cursor, goal);
        let _ = self.world.set_block(
            goal,
            Block {
                kind: BLOCK_KIND_GOAL,
                active: false,
                reward_weight: 0.0,
            },
        );
        self.observation()
    }

    pub fn observation(&self) -> VoxelObservation {
        VoxelObservation {
            filled: self.agent_filled(),
            steps_remaining: self.max_steps.saturating_sub(self.steps),
            goal_distance: self.active_goal.map(|goal| manhattan(self.cursor, goal)),
        }
    }
    /// Clear fixed waypoints so random selection resumes.
    pub fn clear_waypoints(&mut self) {
        self.fixed_start = None;
        self.fixed_goal = None;
    }

    pub fn world(&self) -> &WorldState {
        &self.world
    }

    pub fn replace_world(&mut self, world: WorldState) {
        let extent = world.extent();
        self.extent = [extent.x as u16, extent.y as u16, extent.z as u16];
        self.grid_size = *self.extent.iter().max().expect("extent has three axes");
        self.geometry_len = world.block_count() as usize;
        self.agent_filled_count = 0;
        self.surface_cells.clear();
        self.world = world;
    }

    pub fn trail_mode(&self) -> bool {
        self.trail_mode
    }

    pub fn cursor(&self) -> Coord {
        self.cursor
    }

    /// Update cursor position from outside (called by py_bindings after movement).
    pub fn set_cursor(&mut self, coord: Coord) {
        self.cursor = coord;
    }

    pub fn active_goal(&self) -> Option<Coord> {
        self.active_goal
    }

    pub fn surface_cells(&self) -> &[Coord] {
        &self.surface_cells
    }

    fn agent_filled(&self) -> usize {
        self.agent_filled_count
    }

    /// Replace geometry in-place without reinstantiating the env.
    /// Clears all filled cells (geometry + trail), fills the new geometry, and
    /// recomputes surface cells. Does not reset steps or cursor — call reset()
    /// after set_geometry() to start a new episode.
    pub fn set_geometry(&mut self, geometry: Vec<Coord>) {
        self.agent_filled_count = 0;
        self.world
            .replace_base_blocks(geometry.iter().copied().map(|coord| {
                (
                    coord,
                    Block {
                        kind: crate::voxel::world::BLOCK_KIND_OCCUPIED,
                        active: true,
                        reward_weight: 0.0,
                    },
                )
            }));
        self.geometry_len = geometry.len();
        self.surface_cells = compute_surface_cells(&geometry, self.extent);
    }
}

mod action_pipeline;
pub(crate) mod lifecycle;

use super::geometry::{compute_surface_cells, coord_to_i32, l2, manhattan};

#[cfg(test)]
mod tests;
