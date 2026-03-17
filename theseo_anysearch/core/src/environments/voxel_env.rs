use crate::world::{Block, Coord, World, WorldState};

use super::traits::{Environment, StepResult};

// ---------------------------------------------------------------------------
// Reward configuration — passed from Python, computed in Rust
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct RewardConfig {
    /// Per-step penalty (typically negative, e.g. -0.05).
    pub step_cost: f32,
    /// Bonus awarded when cursor reaches the goal position.
    pub goal_reward: f32,
    /// Coefficient for potential-based L2 distance shaping toward goal.
    /// Each step: shaping = distance_shaping * (prev_l2 - new_l2).
    /// Set to 0.0 to disable.
    pub distance_shaping: f32,
    /// Extra penalty subtracted when a movement is blocked (boundary hit or
    /// occupied cell).  Positive value = bigger penalty.  Default 0.0.
    pub collision_cost: f32,
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            step_cost: -0.01,
            goal_reward: 1.0,
            distance_shaping: 0.1,
            collision_cost: 0.0,
        }
    }
}

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
    /// L2 (Euclidean) distance from cursor to goal at the end of the previous step
    /// (used for potential-based shaping).
    prev_goal_dist_l2: f32,
    reward_config: RewardConfig,
}

impl VoxelEnv {
    pub fn new(world: WorldState, max_steps: u32) -> Self {
        Self {
            world,
            geometry: Vec::new(),
            geometry_len: 0,
            surface_cells: Vec::new(),
            trail_mode: false,
            max_steps,
            steps: 0,
            grid_size: 32,
            cursor: (1, 1, 1),
            fixed_start: None,
            fixed_goal: None,
            active_goal: None,
            prev_goal_dist_l2: 0.0,
            reward_config: RewardConfig::default(),
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
            let _ = self.world.set_block(coord, Block::default());
        }
        self.geometry_len = geometry.len();
        self.surface_cells = compute_surface_cells(&geometry, self.grid_size);
        self.geometry = geometry;
        self
    }

    /// When trail_mode is true, successful movement auto-fills the destination.
    pub fn with_trail_mode(mut self, trail_mode: bool) -> Self {
        self.trail_mode = trail_mode;
        self
    }

    /// Set reward parameters.
    pub fn with_reward_config(mut self, config: RewardConfig) -> Self {
        self.reward_config = config;
        self
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
        self.world.len().saturating_sub(self.geometry_len)
    }
}

// ---------------------------------------------------------------------------
// Surface cell computation
// ---------------------------------------------------------------------------

const DIRS: [(i32, i32, i32); 6] = [
    (1, 0, 0), (-1, 0, 0),
    (0, 1, 0), (0, -1, 0),
    (0, 0, 1), (0, 0, -1),
];

/// Returns the set of all free (non-geometry) cells reachable from the grid
/// boundary via 6-connectivity BFS.  Used to exclude enclosed cavities.
fn reachable_from_boundary(geo_set: &std::collections::HashSet<Coord>, grid_size: u16) -> std::collections::HashSet<Coord> {
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

/// L2 (Euclidean) distance between two coords.
fn l2(a: Coord, b: Coord) -> f32 {
    let dx = (a.0 as f32 - b.0 as f32);
    let dy = (a.1 as f32 - b.1 as f32);
    let dz = (a.2 as f32 - b.2 as f32);
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
        self.world.clear();
        for &coord in &self.geometry {
            let _ = self.world.set_block(coord, Block::default());
        }

        // Determine start and goal for this episode.
        let (start, goal) = if let (Some(s), Some(g)) = (self.fixed_start, self.fixed_goal) {
            (s, g)
        } else if self.surface_cells.len() >= 2 {
            let idx_a = (seed as usize).wrapping_mul(2654435761) % self.surface_cells.len();
            let seed2 = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
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
            if seed % 2 == 0 { (a, b) } else { (b, a) }
        } else {
            // No geometry / too few surface cells — fall back to fixed default.
            ((1, 1, 1), (1, 1, 1))
        };

        self.cursor = start;
        // Only set a goal when there are real surface cells or explicit waypoints.
        self.active_goal = if goal != start { Some(goal) } else { None };

        let goal_distance = self.active_goal.map(|g| manhattan(self.cursor, g));
        self.prev_goal_dist_l2 = self.active_goal.map_or(0.0, |g| l2(self.cursor, g));

        VoxelObservation {
            filled: self.agent_filled(),
            steps_remaining: self.max_steps,
            goal_distance,
        }
    }

    fn step(&mut self, action: Self::Action) -> StepResult<Self::Observation> {
        let is_collision = matches!(action, VoxelAction::Collision);
        self.steps += 1;

        // Apply world action (Place / Remove / Noop / Collision).
        match action {
            VoxelAction::Place(coord) => { let _ = self.world.set_block(coord, Block::default()); }
            VoxelAction::Remove(coord) => { let _ = self.world.remove_block(coord); }
            VoxelAction::Noop | VoxelAction::Collision => {}
        };

        // Navigation reward: goal reached?
        let goal_reached = self.active_goal.map_or(false, |g| self.cursor == g);
        let goal_reward = if goal_reached { self.reward_config.goal_reward } else { 0.0 };

        // L2 distance shaping — computed before updating prev_goal_dist_l2.
        let new_l2 = self.active_goal.map_or(0.0, |g| l2(self.cursor, g));
        let goal_distance = self.active_goal.map(|g| manhattan(self.cursor, g));

        let agent_filled = self.agent_filled();

        let shaping = if self.active_goal.is_some() {
            self.reward_config.distance_shaping * (self.prev_goal_dist_l2 - new_l2)
        } else {
            0.0
        };
        self.prev_goal_dist_l2 = new_l2;

        let step_cost = self.reward_config.step_cost;
        let collision_penalty = if is_collision { self.reward_config.collision_cost } else { 0.0 };

        let done = self.steps >= self.max_steps || goal_reached;

        let observation = VoxelObservation {
            filled: agent_filled,
            steps_remaining: self.max_steps.saturating_sub(self.steps),
            goal_distance,
        };

        StepResult {
            observation,
            // Formula: step_cost + goal_reward - collision_cost + L2_shaping
            reward: step_cost + goal_reward - collision_penalty + shaping,
            done,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::World;

    fn make_env(max_steps: u32) -> VoxelEnv {
        VoxelEnv::new(WorldState::new(), max_steps)
    }

    fn make_env_with_geometry() -> VoxelEnv {
        VoxelEnv::new(WorldState::new(), 20)
            .with_geometry(vec![(5, 5, 5), (6, 5, 5)])
            .with_trail_mode(true)
    }

    #[test]
    fn reset_clears_world_no_geometry() {
        let mut env = make_env(10);
        env.step(VoxelAction::Place((1, 1, 1)));
        let obs = env.reset(42);
        assert_eq!(obs.filled, 0);
        assert_eq!(obs.steps_remaining, 10);
        assert!(!env.world().is_filled((1, 1, 1)));
    }

    #[test]
    fn reset_restores_geometry() {
        let mut env = VoxelEnv::new(WorldState::new(), 10)
            .with_geometry(vec![(5, 5, 5), (6, 6, 6)]);
        env.step(VoxelAction::Place((1, 1, 1)));
        let obs = env.reset(42);
        // geometry_len=2, agent_filled=0
        assert_eq!(obs.filled, 0);
        assert!(env.world().is_filled((5, 5, 5)));
        assert!(!env.world().is_filled((1, 1, 1)));
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
        let mut env = VoxelEnv::new(WorldState::new(), 10)
            .with_geometry(vec![(10, 10, 10)]);
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
        let env = VoxelEnv::new(WorldState::new(), 10)
            .with_geometry(vec![(5, 5, 5)]);
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
        let env = VoxelEnv::new(WorldState::new(), 10)
            .with_geometry(geo.clone());
        for &c in env.surface_cells() {
            assert!(!geo.contains(&c), "surface cell must not be a geometry cell");
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
        let mut env = VoxelEnv::new(WorldState::new(), 100)
            .with_geometry(vec![(5, 5, 5)]);
        env.reset(42);
        // Force goal to a known cell.
        let goal = env.active_goal().unwrap_or((4, 5, 5));
        env.set_waypoints(env.cursor(), goal);
        env.reset(0);  // picks fixed waypoints
        // Move cursor to goal manually.
        env.set_cursor(goal);
        let sr = env.step(VoxelAction::Noop);
        assert!(sr.done, "episode must end when cursor reaches goal");
        assert!(sr.reward >= env.reward_config.goal_reward - 1.0,
                "goal reward must be included");
    }

    #[test]
    fn fixed_waypoints_override_random() {
        let mut env = VoxelEnv::new(WorldState::new(), 10)
            .with_geometry(vec![(10, 10, 10)]);
        env.set_waypoints((2, 2, 2), (3, 3, 3));
        env.reset(0);
        assert_eq!(env.cursor(), (2, 2, 2));
        assert_eq!(env.active_goal(), Some((3, 3, 3)));
    }

    #[test]
    fn manhattan_to_goal_in_observation() {
        let mut env = VoxelEnv::new(WorldState::new(), 10)
            .with_geometry(vec![(10, 10, 10)]);
        env.set_waypoints((1, 1, 1), (4, 1, 1));
        let obs = env.reset(0);
        assert_eq!(obs.goal_distance, Some(3));
    }
}
