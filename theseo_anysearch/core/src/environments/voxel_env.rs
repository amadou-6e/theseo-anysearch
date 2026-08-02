use crate::world::{Block, Coord, World, WorldState, BLOCK_KIND_GOAL, BLOCK_KIND_START};
pub use crate::voxel::rewards::{DistanceRewardMode, RewardConfig, ZoneRewardCurve};

use crate::voxel::{
    actions::{
        ActionExtensionSpec, ActionHistoryEntryV2, ConfiguredOutcome, ConfiguredPredicate,
        PendingMutations,
    },
    common::ABI_VERSION,
    outcomes::{
        builtins as outcome_builtins, NativeOutcomeExtension, OutcomeContextV2, OutcomeResultV2,
    },
    predicates::{builtins as predicate_builtins, NativePredicateExtension, PredicateContextV2},
    rewards::{
        builtins as reward_components, NativeRewardExtension, RewardContextV2,
    },
};
use super::traits::{Environment, StepResult};
use std::collections::{HashMap, HashSet};
use std::path::Path;
const ACTION_OFFSETS_26: [(i32, i32, i32); 26] = [
    (-1, -1, -1),
    (-1, -1, 0),
    (-1, -1, 1),
    (-1, 0, -1),
    (-1, 0, 0),
    (-1, 0, 1),
    (-1, 1, -1),
    (-1, 1, 0),
    (-1, 1, 1),
    (0, -1, -1),
    (0, -1, 0),
    (0, -1, 1),
    (0, 0, -1),
    (0, 0, 1),
    (0, 1, -1),
    (0, 1, 0),
    (0, 1, 1),
    (1, -1, -1),
    (1, -1, 0),
    (1, -1, 1),
    (1, 0, -1),
    (1, 0, 0),
    (1, 0, 1),
    (1, 1, -1),
    (1, 1, 0),
    (1, 1, 1),
];
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
    /// Fixed obstacle geometry — restored on every reset.
    geometry: Vec<Coord>,
    geometry_len: usize,
    agent_filled_count: usize,
    /// Cells 6-adjacent to geometry but not in geometry — valid start/goal positions.
    surface_cells: Vec<Coord>,
    /// When true, movement actions auto-fill the destination cell.
    trail_mode: bool,
    max_steps: u32,
    steps: u32,
    /// Side length of the cubic grid (coords in [1, grid_size]³). Default 32.
    pub grid_size: u16,
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
            geometry: Vec::new(),
            geometry_len: 0,
            agent_filled_count: 0,
            surface_cells: Vec::new(),
            trail_mode: false,
            max_steps,
            steps: 0,
            grid_size: 32,
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
        self
    }

    /// Pre-fill geometry obstacle cells. Computes surface cells automatically.
    pub fn with_geometry(mut self, geometry: Vec<Coord>) -> Self {
        for &coord in &geometry {
            let _ = self.world.set_block(
                coord,
                Block {
                    kind: crate::world::BLOCK_KIND_OCCUPIED,
                    active: true,
                    reward_weight: 0.0,
                },
            );
        }
        self.geometry_len = geometry.len();
        self.surface_cells = compute_surface_cells(&geometry, self.grid_size);
        self.geometry = geometry;
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

    pub fn prepare_navigation_step(
        &mut self,
        action_index: i32,
        previous_cursor: Coord,
        invalid_action: bool,
    ) {
        self.pending_action_index = action_index;
        self.pending_previous_cursor = previous_cursor;
        self.pending_invalid_action = invalid_action;
    }

    pub fn configure_action_pipeline(
        &mut self,
        predicates_json: &str,
        outcomes_json: &str,
        history_length: usize,
        native_library: Option<&Path>,
    ) -> Result<(), String> {
        let predicate_specs: Vec<ActionExtensionSpec> = serde_json::from_str(predicates_json)
            .map_err(|error| format!("invalid action predicates: {error}"))?;
        let outcome_specs: Vec<ActionExtensionSpec> = serde_json::from_str(outcomes_json)
            .map_err(|error| format!("invalid action outcomes: {error}"))?;
        self.action_predicates = predicate_specs
            .into_iter()
            .map(|spec| {
                let parameters =
                    serde_json::to_string(&spec.parameters).expect("JSON map serializes");
                if let Some(path) = native_library {
                    if let Ok(extension) =
                        NativePredicateExtension::load(path, &spec.name, parameters.clone())
                    {
                        return Ok(ConfiguredPredicate::Native(extension));
                    }
                }
                match spec.name.as_str() {
                    "valid_action" => Ok(ConfiguredPredicate::ValidAction),
                    "bounds" => Ok(ConfiguredPredicate::Bounds),
                    "unoccupied" => Ok(ConfiguredPredicate::Unoccupied),
                    _ => Err(format!("unknown action predicate {:?}", spec.name)),
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        self.action_outcomes = outcome_specs
            .into_iter()
            .map(|spec| {
                let parameters =
                    serde_json::to_string(&spec.parameters).expect("JSON map serializes");
                if let Some(path) = native_library {
                    if let Ok(extension) =
                        NativeOutcomeExtension::load(path, &spec.name, parameters.clone())
                    {
                        return Ok(ConfiguredOutcome::Native(extension));
                    }
                }
                match spec.name.as_str() {
                    "cursor_movement" => Ok(ConfiguredOutcome::CursorMovement),
                    "trail_placement" => Ok(ConfiguredOutcome::TrailPlacement),
                    "place" => Ok(ConfiguredOutcome::Place),
                    "remove" => Ok(ConfiguredOutcome::Remove),
                    _ => Err(format!("unknown action outcome {:?}", spec.name)),
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        self.action_history_length = history_length;
        self.action_history.clear();
        Ok(())
    }

    fn proposed_action(&self, action_index: i32) -> (Coord, [i32; 3], bool, bool, bool) {
        let cursor = coord_to_i32(self.cursor);
        let valid_action = (0..=26).contains(&action_index);
        let destination = if (0..26).contains(&action_index) {
            let offset = ACTION_OFFSETS_26[action_index as usize];
            [
                cursor[0] + offset.0,
                cursor[1] + offset.1,
                cursor[2] + offset.2,
            ]
        } else {
            cursor
        };
        let grid = i32::from(self.grid_size);
        let in_bounds = destination.iter().all(|value| (1..=grid).contains(value));
        let coord = if in_bounds {
            (
                destination[0] as u16,
                destination[1] as u16,
                destination[2] as u16,
            )
        } else {
            self.cursor
        };
        let blocked = in_bounds && action_index != 26 && self.world.is_blocking(coord);
        (coord, destination, valid_action, in_bounds, blocked)
    }

    fn predicate_context(
        &self,
        action_index: i32,
        destination: [i32; 3],
        valid_action: bool,
        in_bounds: bool,
        blocked: bool,
    ) -> PredicateContextV2 {
        let observation = VoxelObservation {
            filled: self.agent_filled(),
            steps_remaining: self.max_steps.saturating_sub(self.steps),
            goal_distance: self.active_goal.map(|goal| manhattan(self.cursor, goal)),
        };
        PredicateContextV2 {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<PredicateContextV2>() as u32,
            step: u64::from(self.steps),
            grid_size: self.grid_size,
            action_index,
            cursor: coord_to_i32(self.cursor),
            destination,
            goal: self.active_goal.map_or([0; 3], coord_to_i32),
            has_goal: u8::from(self.active_goal.is_some()),
            valid_action: u8::from(valid_action),
            destination_in_bounds: u8::from(in_bounds),
            destination_blocked: u8::from(blocked),
            observation_filled: observation.filled,
            observation_steps_remaining: observation.steps_remaining,
            observation_goal_distance: observation.goal_distance.unwrap_or(0),
            has_observation_goal_distance: u8::from(observation.goal_distance.is_some()),
            history: self.action_history.as_slice().as_ptr(),
            history_len: self.action_history.len(),
            parameters_json: std::ptr::null(),
            parameters_json_len: 0,
        }
    }

    fn action_feasible(&self, action_index: i32) -> Result<(Coord, [i32; 3], bool), String> {
        let (coord, destination, valid_action, in_bounds, blocked) =
            self.proposed_action(action_index);
        let context =
            self.predicate_context(action_index, destination, valid_action, in_bounds, blocked);
        for predicate in &self.action_predicates {
            let feasible = match predicate {
                ConfiguredPredicate::ValidAction => {
                    predicate_builtins::valid_action::evaluate(valid_action)
                }
                ConfiguredPredicate::Bounds => predicate_builtins::bounds::evaluate(in_bounds),
                ConfiguredPredicate::Unoccupied => {
                    predicate_builtins::unoccupied::evaluate(action_index, blocked)
                }
                ConfiguredPredicate::Native(extension) => {
                    extension.evaluate(PredicateContextV2 { ..context })?
                }
            };
            if !feasible {
                return Ok((coord, destination, false));
            }
        }
        Ok((coord, destination, true))
    }

    pub fn action_mask(&mut self) -> Vec<u8> {
        (0..=26)
            .map(|action| match self.action_feasible(action) {
                Ok((_, _, feasible)) => u8::from(feasible),
                Err(error) => {
                    self.last_reward_error = Some(error);
                    0
                }
            })
            .collect()
    }

    pub fn execute_navigation_action(&mut self, action_index: i32) -> VoxelAction {
        let previous = self.cursor;
        let (_destination_coord, destination, feasible) = match self.action_feasible(action_index) {
            Ok(value) => value,
            Err(error) => {
                self.last_reward_error = Some(error);
                self.pending_action_feasible = false;
                return VoxelAction::Collision;
            }
        };
        self.pending_action_feasible = feasible;
        if !feasible {
            return if self.pending_invalid_action {
                VoxelAction::Noop
            } else {
                VoxelAction::Collision
            };
        }
        let history = self.action_history.as_slice();
        let context = OutcomeContextV2 {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<OutcomeContextV2>() as u32,
            step: u64::from(self.steps),
            grid_size: self.grid_size,
            action_index,
            cursor: coord_to_i32(previous),
            destination,
            goal: self.active_goal.map_or([0; 3], coord_to_i32),
            has_goal: u8::from(self.active_goal.is_some()),
            history: history.as_ptr(),
            history_len: history.len(),
            parameters_json: std::ptr::null(),
            parameters_json_len: 0,
        };
        let mut mutations = PendingMutations::default();
        for outcome in &self.action_outcomes {
            let result = match outcome {
                ConfiguredOutcome::CursorMovement => {
                    outcome_builtins::cursor_movement::apply(destination)
                }
                ConfiguredOutcome::TrailPlacement => {
                    outcome_builtins::trail_placement::apply(action_index, destination)
                }
                ConfiguredOutcome::Place => {
                    outcome_builtins::place::apply(action_index, destination)
                }
                ConfiguredOutcome::Remove => {
                    outcome_builtins::remove::apply(coord_to_i32(previous))
                }
                ConfiguredOutcome::Native(extension) => {
                    match extension.evaluate(OutcomeContextV2 { ..context }) {
                        Ok(result) => result,
                        Err(error) => {
                            self.last_reward_error = Some(error);
                            return VoxelAction::Collision;
                        }
                    }
                }
            };
            if let Err(error) = merge_mutations(&mut mutations, result, self.grid_size) {
                self.last_reward_error = Some(error);
                return VoxelAction::Collision;
            }
        }
        if let Some(coord) = mutations.cursor {
            if self.world.is_blocking(coord) && coord != previous {
                self.last_reward_error =
                    Some(format!("outcome cursor target {coord:?} is blocked"));
                return VoxelAction::Collision;
            }
        }
        if let Some(coord) = mutations.remove {
            let removable = self
                .world
                .get_block(coord)
                .is_some_and(|block| block.active && block.reward_weight > 0.0);
            if !removable {
                self.last_reward_error =
                    Some(format!("outcome cannot remove non-agent voxel {coord:?}"));
                return VoxelAction::Collision;
            }
        }
        if let Some(coord) = mutations.place {
            let occupied_by_non_agent = self
                .world
                .get_block(coord)
                .is_some_and(|block| block.active && block.reward_weight <= 0.0);
            if occupied_by_non_agent {
                self.last_reward_error =
                    Some(format!("outcome cannot overwrite geometry voxel {coord:?}"));
                return VoxelAction::Collision;
            }
        }
        if mutations.place.is_some() && mutations.place == mutations.remove {
            self.last_reward_error =
                Some("outcome cannot place and remove the same voxel".to_owned());
            return VoxelAction::Collision;
        }
        if let Some(coord) = mutations.remove {
            let was_agent_filled = self
                .world
                .get_block(coord)
                .is_some_and(|block| block.active && block.reward_weight > 0.0);
            let _ = self.world.remove_block(coord);
            if was_agent_filled {
                self.agent_filled_count = self.agent_filled_count.saturating_sub(1);
            }
        }
        if let Some(coord) = mutations.cursor {
            self.cursor = coord;
        }
        if let Some(coord) = mutations.place {
            let was_agent_filled = self
                .world
                .get_block(coord)
                .is_some_and(|block| block.active && block.reward_weight > 0.0);
            let _ = self.world.set_block(coord, Block::default());
            if !was_agent_filled {
                self.agent_filled_count += 1;
            }
        }
        VoxelAction::Noop
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
        self.fixed_start = Some(start);
        self.fixed_goal = Some(goal);
    }

    /// Clear fixed waypoints so random selection resumes.
    pub fn clear_waypoints(&mut self) {
        self.fixed_start = None;
        self.fixed_goal = None;
    }

    pub fn world(&self) -> &WorldState {
        &self.world
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
        self.world.clear();
        self.agent_filled_count = 0;
        for &coord in &geometry {
            let _ = self.world.set_block(
                coord,
                Block {
                    kind: crate::world::BLOCK_KIND_OCCUPIED,
                    active: true,
                    reward_weight: 0.0,
                },
            );
        }
        self.geometry_len = geometry.len();
        self.surface_cells = compute_surface_cells(&geometry, self.grid_size);
        self.geometry = geometry;
    }
}

// ---------------------------------------------------------------------------
// Surface cell computation
// ---------------------------------------------------------------------------

const DIRS: [(i32, i32, i32); 6] = [
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
];

/// Returns the set of all free (non-geometry) cells reachable from the grid
/// boundary via 6-connectivity BFS.  Used to exclude enclosed cavities.
fn reachable_from_boundary(
    geo_set: &std::collections::HashSet<Coord>,
    grid_size: u16,
) -> std::collections::HashSet<Coord> {
    use std::collections::{HashSet, VecDeque};
    let mut reachable: HashSet<Coord> = HashSet::new();
    let mut queue: VecDeque<Coord> = VecDeque::new();

    let g = grid_size as i32;
    for u in 1u16..=grid_size {
        for v in 1u16..=grid_size {
            for &w in &[1u16, grid_size] {
                for &c in &[(u, v, w), (u, w, v), (w, u, v)] {
                    if !geo_set.contains(&c) && reachable.insert(c) {
                        queue.push_back(c);
                    }
                }
            }
        }
    }

    while let Some((x, y, z)) = queue.pop_front() {
        for (dx, dy, dz) in DIRS {
            let nx = x as i32 + dx;
            let ny = y as i32 + dy;
            let nz = z as i32 + dz;
            if nx >= 1 && ny >= 1 && nz >= 1 && nx <= g && ny <= g && nz <= g {
                let nc = (nx as u16, ny as u16, nz as u16);
                if !geo_set.contains(&nc) && reachable.insert(nc) {
                    queue.push_back(nc);
                }
            }
        }
    }
    reachable
}

/// Returns cells that are 6-adjacent to geometry, not in geometry, and
/// reachable from the grid boundary (i.e. not enclosed inside a cavity).
fn compute_surface_cells(geometry: &[Coord], grid_size: u16) -> Vec<Coord> {
    use std::collections::HashSet;
    let geo_set: HashSet<Coord> = geometry.iter().copied().collect();
    let reachable = reachable_from_boundary(&geo_set, grid_size);

    let g = grid_size as i32;
    let mut surface: HashSet<Coord> = HashSet::new();
    for &(gx, gy, gz) in geometry {
        for (dx, dy, dz) in DIRS {
            let nx = gx as i32 + dx;
            let ny = gy as i32 + dy;
            let nz = gz as i32 + dz;
            if nx >= 1 && ny >= 1 && nz >= 1 && nx <= g && ny <= g && nz <= g {
                let nc = (nx as u16, ny as u16, nz as u16);
                if !geo_set.contains(&nc) && reachable.contains(&nc) {
                    surface.insert(nc);
                }
            }
        }
    }
    let mut cells: Vec<Coord> = surface.into_iter().collect();
    cells.sort(); // deterministic ordering
    cells
}

fn coord_to_i32(coord: Coord) -> [i32; 3] {
    [i32::from(coord.0), i32::from(coord.1), i32::from(coord.2)]
}

/// L2 (Euclidean) distance between two coords.
fn l2(a: Coord, b: Coord) -> f32 {
    let dx = a.0 as f32 - b.0 as f32;
    let dy = a.1 as f32 - b.1 as f32;
    let dz = a.2 as f32 - b.2 as f32;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Manhattan distance between two coords (used for the observation field only).
fn manhattan(a: Coord, b: Coord) -> u32 {
    let dx = (a.0 as i32 - b.0 as i32).unsigned_abs();
    let dy = (a.1 as i32 - b.1 as i32).unsigned_abs();
    let dz = (a.2 as i32 - b.2 as i32).unsigned_abs();
    dx + dy + dz
}

/// Pick an element from a slice using a seed (LCG shuffle index).
fn pick_from<T: Copy>(slice: &[T], seed: u64) -> T {
    let idx = (seed as usize).wrapping_mul(2654435761) % slice.len();
    slice[idx]
}

// ---------------------------------------------------------------------------
// Environment trait implementation
// ---------------------------------------------------------------------------

impl Environment for VoxelEnv {
    type Action = VoxelAction;
    type Observation = VoxelObservation;

    fn reset(&mut self, seed: u64) -> Self::Observation {
        self.steps = 0;
        self.consecutive_collisions = 0;
        self.action_history.clear();
        self.last_reward_breakdown.clear();
        self.last_collision = false;
        self.last_goal_reached = false;
        self.last_terminated = false;
        self.last_truncated = false;
        self.last_termination_reason = "in_progress".to_owned();
        self.last_reward_error = None;
        self.last_goal_distance_l2 = 0.0;
        self.last_construction_residual = 0;
        self.last_construction_overshoot = 0;
        self.world.clear();
        self.agent_filled_count = 0;
        for &coord in &self.geometry {
            let _ = self.world.set_block(
                coord,
                Block {
                    kind: crate::world::BLOCK_KIND_OCCUPIED,
                    active: true,
                    reward_weight: 0.0,
                },
            );
        }

        // Determine start and goal for this episode.
        let (start, goal) = if let (Some(s), Some(g)) = (self.fixed_start, self.fixed_goal) {
            (s, g)
        } else if self.surface_cells.len() >= 2 {
            let idx_a = (seed as usize).wrapping_mul(2654435761) % self.surface_cells.len();
            let seed2 = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx_b_raw = (seed2 as usize).wrapping_mul(2654435761) % self.surface_cells.len();
            // Ensure the two indices are distinct.
            let idx_b = if idx_b_raw == idx_a {
                (idx_a + self.surface_cells.len() / 2) % self.surface_cells.len()
            } else {
                idx_b_raw
            };
            let a = self.surface_cells[idx_a];
            let b = self.surface_cells[idx_b];
            // Swap on alternating seeds to train both navigation directions.
            if seed % 2 == 0 {
                (a, b)
            } else {
                (b, a)
            }
        } else {
            // No geometry / too few surface cells — fall back to fixed default.
            ((1, 1, 1), (1, 1, 1))
        };

        self.cursor = start;
        // Only set a goal when there are real surface cells or explicit waypoints.
        self.active_goal = if goal != start { Some(goal) } else { None };
        let _ = self.world.set_block(
            start,
            Block {
                kind: BLOCK_KIND_START,
                active: false,
                reward_weight: 0.0,
            },
        );
        if let Some(active_goal) = self.active_goal {
            let _ = self.world.set_block(
                active_goal,
                Block {
                    kind: BLOCK_KIND_GOAL,
                    active: false,
                    reward_weight: 0.0,
                },
            );
        }

        let goal_distance = self.active_goal.map(|g| manhattan(self.cursor, g));
        self.prev_goal_dist_l2 = if self.success_targets.is_empty() {
            self.active_goal.map_or(0.0, |goal| l2(self.cursor, goal))
        } else {
            self.success_targets
                .iter()
                .map(|goal| l2(self.cursor, *goal))
                .reduce(f32::min)
                .unwrap_or(0.0)
        };

        VoxelObservation {
            filled: self.agent_filled(),
            steps_remaining: self.max_steps,
            goal_distance,
        }
    }

    fn step(&mut self, action: Self::Action) -> StepResult<Self::Observation> {
        let is_collision = matches!(action, VoxelAction::Collision);
        self.steps += 1;
        if is_collision {
            self.consecutive_collisions += 1;
        } else {
            self.consecutive_collisions = 0;
            self.action_history.clear();
        }

        match action {
            VoxelAction::Place(coord) => {
                let was_agent_filled = self
                    .world
                    .get_block(coord)
                    .is_some_and(|block| block.active && block.reward_weight > 0.0);
                let _ = self.world.set_block(coord, Block::default());
                if !was_agent_filled {
                    self.agent_filled_count += 1;
                }
            }
            VoxelAction::Remove(coord) => {
                let was_agent_filled = self
                    .world
                    .get_block(coord)
                    .is_some_and(|block| block.active && block.reward_weight > 0.0);
                let _ = self.world.remove_block(coord);
                if was_agent_filled {
                    self.agent_filled_count = self.agent_filled_count.saturating_sub(1);
                }
            }
            VoxelAction::Noop | VoxelAction::Collision => {}
        };

        let accepted_targets = if self.success_targets.is_empty() {
            self.active_goal.iter().copied().collect::<Vec<_>>()
        } else {
            self.success_targets.clone()
        };
        let new_l2 = accepted_targets
            .iter()
            .map(|goal| l2(self.cursor, *goal))
            .reduce(f32::min)
            .unwrap_or(0.0);
        let goal_reached = !accepted_targets.is_empty() && new_l2 <= self.goal_tolerance;
        let goal_distance = self.active_goal.map(|goal| manhattan(self.cursor, goal));
        let step_cost = reward_components::step_cost::compute(&self.reward_config);
        let distance_reward = reward_components::distance::compute(
            &self.reward_config,
            self.active_goal.is_some(),
            self.prev_goal_dist_l2,
            new_l2,
            self.grid_size,
        );
        let (residual, overshoot) = if self.construction_target.is_empty() {
            (0, 0)
        } else {
            let filled: HashSet<Coord> = self
                .world
                .iter_filled()
                .copied()
                .filter(|coord| {
                    self.world
                        .get_block(*coord)
                        .is_some_and(|block| block.active && block.reward_weight > 0.0)
                })
                .collect();
            (
                self.construction_target.difference(&filled).count(),
                filled.difference(&self.construction_target).count(),
            )
        };
        let mut breakdown = HashMap::from([
            (reward_components::step_cost::NAME.to_owned(), step_cost),
            (reward_components::distance::NAME.to_owned(), distance_reward),
            (
                reward_components::goal::NAME.to_owned(),
                reward_components::goal::compute(&self.reward_config, goal_reached),
            ),
            (
                reward_components::invalid_action::NAME.to_owned(),
                reward_components::invalid_action::compute(
                    &self.reward_config,
                    self.pending_invalid_action,
                ),
            ),
            (
                reward_components::collision::NAME.to_owned(),
                reward_components::collision::compute(&self.reward_config, is_collision),
            ),
            (
                reward_components::construction::RESIDUAL_NAME.to_owned(),
                reward_components::construction::residual(&self.reward_config, residual),
            ),
            (
                reward_components::construction::OVERSHOOT_NAME.to_owned(),
                reward_components::construction::overshoot(&self.reward_config, overshoot),
            ),
        ]);
        let standard_reward: f32 = breakdown.values().sum();

        let collision_terminated = self
            .max_consecutive_collisions
            .is_some_and(|limit| self.consecutive_collisions >= limit);
        let terminated = (goal_reached && self.terminate_on_success) || collision_terminated;
        let truncated = self.steps >= self.max_steps && !terminated;
        let context = RewardContextV2 {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<RewardContextV2>() as u32,
            step: u64::from(self.steps),
            action_index: self.pending_action_index,
            previous_cursor: coord_to_i32(self.pending_previous_cursor),
            cursor: coord_to_i32(self.cursor),
            goal: self.active_goal.map_or([0; 3], coord_to_i32),
            has_goal: u8::from(self.active_goal.is_some()),
            invalid_action: u8::from(self.pending_invalid_action),
            collision: u8::from(is_collision),
            goal_reached: u8::from(goal_reached),
            terminated: u8::from(terminated),
            truncated: u8::from(truncated),
            consecutive_collisions: self.consecutive_collisions,
            previous_goal_distance: f64::from(self.prev_goal_dist_l2),
            goal_distance: f64::from(new_l2),
            standard_reward: f64::from(standard_reward),
            parameters_json: std::ptr::null(),
            parameters_json_len: 0,
        };
        let reward = if let Some(extension) = &self.native_reward {
            match extension.compute(context, &breakdown) {
                Ok((reward, native_breakdown)) => {
                    breakdown = native_breakdown;
                    reward
                }
                Err(error) => {
                    self.last_reward_error = Some(error);
                    standard_reward
                }
            }
        } else {
            standard_reward
        };
        self.prev_goal_dist_l2 = new_l2;
        self.last_reward_breakdown = breakdown;
        self.last_goal_distance_l2 = new_l2;
        self.last_construction_residual = residual;
        self.last_construction_overshoot = overshoot;
        self.last_collision = is_collision;
        self.last_goal_reached = goal_reached;
        self.last_terminated = terminated;
        self.last_truncated = truncated;
        self.last_termination_reason = if goal_reached {
            "success"
        } else if collision_terminated {
            "consecutive_collisions"
        } else if truncated {
            "step_limit"
        } else {
            "in_progress"
        }
        .to_owned();

        if self.action_history_length > 0 {
            self.action_history.push(ActionHistoryEntryV2 {
                action_index: self.pending_action_index,
                previous_cursor: coord_to_i32(self.pending_previous_cursor),
                cursor: coord_to_i32(self.cursor),
                feasible: u8::from(self.pending_action_feasible),
                collision: u8::from(is_collision),
            });
            if self.action_history.len() > self.action_history_length {
                self.action_history.remove(0);
            }
        }
        let observation = VoxelObservation {
            filled: self.agent_filled(),
            steps_remaining: self.max_steps.saturating_sub(self.steps),
            goal_distance,
        };
        StepResult {
            observation,
            reward,
            done: terminated || truncated,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn merge_mutations(
    target: &mut PendingMutations,
    result: OutcomeResultV2,
    grid_size: u16,
) -> Result<(), String> {
    fn coord(raw: [i32; 3], grid_size: u16) -> Result<Coord, String> {
        let grid = i32::from(grid_size);
        if !raw.iter().all(|value| (1..=grid).contains(value)) {
            return Err(format!("outcome coordinate {raw:?} is outside the grid"));
        }
        Ok((raw[0] as u16, raw[1] as u16, raw[2] as u16))
    }
    fn merge(slot: &mut Option<Coord>, value: Coord, kind: &str) -> Result<(), String> {
        if slot.is_some_and(|current| current != value) {
            return Err(format!("conflicting {kind} outcome mutations"));
        }
        *slot = Some(value);
        Ok(())
    }
    if result.set_cursor != 0 {
        merge(&mut target.cursor, coord(result.cursor, grid_size)?, "cursor")?;
    }
    if result.place_voxel != 0 {
        merge(&mut target.place, coord(result.place_coord, grid_size)?, "place")?;
    }
    if result.remove_voxel != 0 {
        merge(&mut target.remove, coord(result.remove_coord, grid_size)?, "remove")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_env(max_steps: u32) -> VoxelEnv {
        VoxelEnv::new(WorldState::new(), max_steps)
    }

    fn make_env_with_geometry() -> VoxelEnv {
        VoxelEnv::new(WorldState::new(), 20)
            .with_geometry(vec![(5, 5, 5), (6, 5, 5)])
            .with_trail_mode(true)
    }

    fn zone_reward_config() -> RewardConfig {
        RewardConfig {
            step_cost: -0.05,
            goal_reward: 10.0,
            distance_shaping: 0.02,
            collision_cost: -0.5,
            distance_reward_mode: DistanceRewardMode::Zone,
            zone_reward_min: -1.0,
            zone_reward_max: -0.01,
            zone_reward_curve: ZoneRewardCurve::Linear,
            ..Default::default()
        }
    }

    #[test]
    fn reset_clears_world_no_geometry() {
        let mut env = make_env(10);
        env.step(VoxelAction::Place((1, 1, 1)));
        let obs = env.reset(42);
        assert_eq!(obs.filled, 0);
        assert_eq!(obs.steps_remaining, 10);
        assert!(env.world().is_filled((1, 1, 1)));
        assert!(!env.world().is_blocking((1, 1, 1)));
    }

    #[test]
    fn reset_restores_geometry() {
        let mut env =
            VoxelEnv::new(WorldState::new(), 10).with_geometry(vec![(5, 5, 5), (6, 6, 6)]);
        env.step(VoxelAction::Place((1, 1, 1)));
        let obs = env.reset(42);
        // geometry_len=2, agent_filled=0
        assert_eq!(obs.filled, 0);
        assert!(env.world().is_filled((5, 5, 5)));
        assert!(!env.world().is_blocking(env.cursor()));
    }

    #[test]
    fn noop_step_no_change() {
        let mut env = make_env(10);
        env.reset(42);
        let before = env.agent_filled();
        env.step(VoxelAction::Noop);
        assert_eq!(env.agent_filled(), before);
    }

    #[test]
    fn place_step_fills() {
        let mut env = make_env(10);
        env.reset(42);
        let sr = env.step(VoxelAction::Place((50, 50, 50)));
        assert!(env.world().is_filled((50, 50, 50)));
        // reward = step_cost + shaping; just check done=false
        assert!(!sr.done);
    }

    #[test]
    fn remove_step_clears() {
        let mut env = make_env(10);
        env.reset(42);
        env.step(VoxelAction::Place((1, 1, 1)));
        env.step(VoxelAction::Remove((1, 1, 1)));
        assert!(!env.world().is_filled((1, 1, 1)));
    }

    #[test]
    fn done_at_max_steps() {
        let mut env = make_env(3);
        env.reset(42);
        env.step(VoxelAction::Noop);
        env.step(VoxelAction::Noop);
        let sr = env.step(VoxelAction::Noop);
        assert!(sr.done);
    }

    #[test]
    fn geometry_not_counted_in_agent_filled() {
        let mut env = VoxelEnv::new(WorldState::new(), 10).with_geometry(vec![(10, 10, 10)]);
        env.reset(42);
        assert_eq!(env.agent_filled(), 0);
        env.step(VoxelAction::Place((1, 1, 1)));
        assert_eq!(env.agent_filled(), 1);
    }

    #[test]
    fn deterministic_reset() {
        let mut env1 = make_env(10);
        let mut env2 = make_env(10);
        let obs1 = env1.reset(42);
        let obs2 = env2.reset(42);
        assert_eq!(obs1.filled, obs2.filled);
        assert_eq!(obs1.steps_remaining, obs2.steps_remaining);
    }

    #[test]
    fn surface_cells_computed_from_geometry() {
        let env = VoxelEnv::new(WorldState::new(), 10).with_geometry(vec![(5, 5, 5)]);
        // A single geometry cell at (5,5,5) should have 6 surface neighbours.
        assert_eq!(env.surface_cells().len(), 6);
        // All neighbours should be adjacent.
        for &(x, y, z) in env.surface_cells() {
            let dist = manhattan((5, 5, 5), (x, y, z));
            assert_eq!(dist, 1, "surface cell should be distance 1 from geometry");
        }
    }

    #[test]
    fn surface_cells_do_not_include_geometry() {
        let geo = vec![(5, 5, 5), (6, 5, 5)];
        let env = VoxelEnv::new(WorldState::new(), 10).with_geometry(geo.clone());
        for &c in env.surface_cells() {
            assert!(
                !geo.contains(&c),
                "surface cell must not be a geometry cell"
            );
        }
    }

    #[test]
    fn reset_places_cursor_on_surface_cell() {
        let mut env = make_env_with_geometry();
        env.reset(7);
        let (cx, cy, cz) = env.cursor();
        let gs = env.grid_size;
        // Cursor should be within grid bounds.
        assert!((1..=gs).contains(&cx));
        assert!((1..=gs).contains(&cy));
        assert!((1..=gs).contains(&cz));
    }

    #[test]
    fn goal_reached_ends_episode() {
        let mut env = VoxelEnv::new(WorldState::new(), 100).with_geometry(vec![(5, 5, 5)]);
        env.reset(42);
        // Force goal to a known cell.
        let goal = env.active_goal().unwrap_or((4, 5, 5));
        env.set_waypoints(env.cursor(), goal);
        env.reset(0); // picks fixed waypoints
                      // Move cursor to goal manually.
        env.set_cursor(goal);
        let sr = env.step(VoxelAction::Noop);
        assert!(sr.done, "episode must end when cursor reaches goal");
        assert!(
            sr.reward >= env.reward_config.goal_reward - 1.0,
            "goal reward must be included"
        );
    }

    #[test]
    fn zone_reward_prefers_closer_absolute_distance() {
        let mut closer_env = make_env(10).with_reward_config(zone_reward_config());
        let mut farther_env = make_env(10).with_reward_config(zone_reward_config());
        closer_env.set_waypoints((4, 4, 4), (4, 4, 6));
        farther_env.set_waypoints((4, 4, 4), (4, 4, 6));
        closer_env.reset(0);
        farther_env.reset(0);

        closer_env.set_cursor((4, 4, 5));
        farther_env.set_cursor((4, 4, 3));
        let closer = closer_env.step(VoxelAction::Noop).reward;
        let farther = farther_env.step(VoxelAction::Noop).reward;

        assert!(farther < closer);
        assert!(closer < 0.0);
    }

    #[test]
    fn zone_reward_goal_step_adds_terminal_bonus() {
        let mut env = make_env(10).with_reward_config(zone_reward_config());
        env.set_waypoints((4, 4, 4), (4, 4, 6));
        env.reset(0);

        env.set_cursor((4, 4, 6));
        let sr = env.step(VoxelAction::Noop);

        assert!(sr.done);
        assert!((sr.reward - 9.99).abs() < 0.0001);
    }

    #[test]
    fn fixed_waypoints_override_random() {
        let mut env = VoxelEnv::new(WorldState::new(), 10).with_geometry(vec![(10, 10, 10)]);
        env.set_waypoints((2, 2, 2), (3, 3, 3));
        env.reset(0);
        assert_eq!(env.cursor(), (2, 2, 2));
        assert_eq!(env.active_goal(), Some((3, 3, 3)));
    }

    #[test]
    fn manhattan_to_goal_in_observation() {
        let mut env = VoxelEnv::new(WorldState::new(), 10).with_geometry(vec![(10, 10, 10)]);
        env.set_waypoints((1, 1, 1), (4, 1, 1));
        let obs = env.reset(0);
        assert_eq!(obs.goal_distance, Some(3));
    }

    #[test]
    fn consecutive_collisions_terminate_in_rust() {
        let mut env = make_env(20).with_max_consecutive_collisions(Some(2));
        env.set_waypoints((1, 1, 1), (3, 1, 1));
        env.reset(0);
        env.prepare_navigation_step(0, env.cursor(), false);
        let first = env.step(VoxelAction::Collision);
        assert!(!first.done);
        env.prepare_navigation_step(0, env.cursor(), false);
        let second = env.step(VoxelAction::Collision);
        assert!(second.done);
        assert!(env.last_terminated());
        assert!(!env.last_truncated());
        assert_eq!(env.last_termination_reason(), "consecutive_collisions");
    }

    #[test]
    fn rust_breakdown_is_authoritative() {
        let mut env = make_env(20);
        env.set_waypoints((1, 1, 1), (3, 1, 1));
        env.reset(0);
        env.prepare_navigation_step(0, env.cursor(), false);
        let result = env.step(VoxelAction::Collision);
        let component_sum: f32 = env.last_reward_breakdown().values().sum();
        assert!((result.reward - component_sum).abs() < 1e-6);
        assert_eq!(env.last_reward_breakdown().get("collision"), Some(&0.0));
    }

    fn configure_pipeline(env: &mut VoxelEnv, outcomes: &str, history: usize) {
        env.configure_action_pipeline(
            r#"[{"name":"valid_action"},{"name":"bounds"},{"name":"unoccupied"}]"#,
            outcomes,
            history,
            None,
        )
        .unwrap();
    }

    #[test]
    fn cursor_navigation_moves_without_filling() {
        let mut env = make_env(10);
        env.set_waypoints((2, 2, 2), (6, 2, 2));
        env.reset(1);
        configure_pipeline(&mut env, r#"[{"name":"cursor_movement"}]"#, 4);
        env.prepare_navigation_step(21, (2, 2, 2), false);
        let action = env.execute_navigation_action(21);
        env.step(action);
        assert_eq!(env.cursor(), (3, 2, 2));
        assert_eq!(env.agent_filled(), 0);
    }

    #[test]
    fn trail_navigation_moves_and_fills_destination() {
        let mut env = make_env(10);
        env.set_waypoints((2, 2, 2), (6, 2, 2));
        env.reset(1);
        configure_pipeline(
            &mut env,
            r#"[{"name":"cursor_movement"},{"name":"trail_placement"}]"#,
            4,
        );
        env.prepare_navigation_step(21, (2, 2, 2), false);
        let action = env.execute_navigation_action(21);
        env.step(action);
        assert_eq!(env.cursor(), (3, 2, 2));
        assert_eq!(env.agent_filled(), 1);
    }

    #[test]
    fn action_mask_uses_the_same_bounds_and_occupancy_predicates() {
        let mut env = make_env(10);
        env.set_waypoints((1, 1, 1), (6, 2, 2));
        env.reset(1);
        configure_pipeline(&mut env, r#"[{"name":"cursor_movement"}]"#, 4);
        let mask = env.action_mask();
        assert_eq!(mask.len(), 27);
        assert_eq!(mask[0], 0);
        assert_eq!(mask[21], 1);
        assert_eq!(mask[26], 1);
    }

    #[test]
    fn action_history_is_bounded() {
        let mut env = make_env(10);
        env.set_waypoints((2, 2, 2), (6, 2, 2));
        env.reset(1);
        configure_pipeline(&mut env, r#"[{"name":"cursor_movement"}]"#, 1);
        for action_index in [21, 21] {
            let previous = env.cursor();
            env.prepare_navigation_step(action_index, previous, false);
            let action = env.execute_navigation_action(action_index);
            env.step(action);
        }
        assert_eq!(env.action_history.len(), 1);
        assert_eq!(env.action_history[0].previous_cursor, [3, 2, 2]);
        assert_eq!(env.action_history[0].cursor, [4, 2, 2]);
    }
}
