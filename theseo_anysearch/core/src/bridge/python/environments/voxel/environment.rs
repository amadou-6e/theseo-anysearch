use std::path::Path;

use pyo3::{exceptions::PyValueError, prelude::*};

use crate::{
    environments::Environment,
    voxel::world::{World, WorldState, BLOCK_KIND_OCCUPIED},
    voxel::VoxelEnv,
};

use super::{
    observation::PyVoxelObservation,
    result::PyStepResultVoxel,
    settings::{default_action_outcomes, reward_config},
};

/// Upper bound on `box_radius`: keeps `2 * r + 1` well within `i32` range so
/// the side-length computation in `build_box_obs` can never overflow, and
/// keeps the resulting allocation ((2r+1)^3 f32s) within sane memory limits.
const MAX_BOX_RADIUS: u32 = 1024;

fn unit_goal_direction(cursor: (u16, u16, u16), goal: (u16, u16, u16)) -> (f32, f32, f32) {
    let dx = f32::from(goal.0) - f32::from(cursor.0);
    let dy = f32::from(goal.1) - f32::from(cursor.1);
    let dz = f32::from(goal.2) - f32::from(cursor.2);
    let norm = (dx * dx + dy * dy + dz * dz).sqrt();
    if norm == 0.0 {
        (0.0, 0.0, 0.0)
    } else {
        (dx / norm, dy / norm, dz / norm)
    }
}

fn validate_box_radius(radius: u32) -> PyResult<()> {
    if radius > MAX_BOX_RADIUS {
        return Err(crate::bridge::python::errors::invalid_value(format!(
            "box_radius must be <= {MAX_BOX_RADIUS}, got {radius}"
        )));
    }
    Ok(())
}

#[pyclass]
pub struct PyVoxelEnv {
    pub(crate) inner: VoxelEnv,
    pub(crate) box_radius: Option<u32>,
}

impl PyVoxelEnv {
    fn to_py_observation(&self, obs: crate::voxel::VoxelObservation) -> PyVoxelObservation {
        let goal_direction = self
            .inner
            .active_goal()
            .map(|goal| unit_goal_direction(self.inner.cursor(), goal));
        PyVoxelObservation {
            filled: obs.filled,
            steps_remaining: obs.steps_remaining,
            goal_distance: obs.goal_distance,
            goal_direction,
            cursor_pos: self.inner.cursor(),
            local_grid: self.box_radius.map(|radius| self.build_box_obs(radius)),
        }
    }

    fn build_box_obs(&self, radius: u32) -> Vec<f32> {
        let (cx, cy, cz) = self.inner.cursor();
        let r = radius as i32;
        let side = (2 * r + 1) as usize;
        let mut result = Vec::with_capacity(side * side * side);
        let grid_size = i32::from(self.inner.grid_size);
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    let x = i32::from(cx) + dx;
                    let y = i32::from(cy) + dy;
                    let z = i32::from(cz) + dz;
                    let kind = if x >= 1
                        && y >= 1
                        && z >= 1
                        && x <= grid_size
                        && y <= grid_size
                        && z <= grid_size
                    {
                        self.inner
                            .world()
                            .get_block((x as u16, y as u16, z as u16))
                            .map_or(0.0, |block| f32::from(block.kind))
                    } else {
                        f32::from(BLOCK_KIND_OCCUPIED)
                    };
                    result.push(kind);
                }
            }
        }
        result
    }
}

#[pymethods]
impl PyVoxelEnv {
    /// * `max_steps`         â€“ episode length limit.
    /// * `trail_mode`        â€“ movement auto-fills the destination cell (default true).
    /// * `geometry`          â€“ obstacle voxels pre-filled and restored on every reset.
    /// * `grid_size`         â€“ side length of the cubic grid (default 32).
    /// * `step_cost`         â€“ per-step penalty (default -0.01).
    /// * `goal_reward`       â€“ bonus when cursor reaches goal (default 1.0).
    /// * `distance_shaping`  â€“ L2 shaping coefficient toward goal (default 0.0).
    /// * `collision_cost`    â€“ extra penalty subtracted on blocked moves (default 0.0).
    #[new]
    #[pyo3(signature = (max_steps, trail_mode=true, geometry=None,
                        grid_size=32, step_cost=-0.01, goal_reward=1.0,
                        distance_shaping=0.0, collision_cost=0.0,
                        distance_reward_mode="progress".to_string(),
                        zone_reward_min=-1.0, zone_reward_max=-0.01,
                        zone_reward_curve="linear".to_string(),
                        invalid_action_cost=0.0,
                        construction_residual_weight=0.0,
                        construction_overshoot_weight=0.0,
                        construction_target=None,
                        max_consecutive_collisions=None,
                        terminate_on_success=true,
                        success_targets=None,
                        goal_tolerance=0.0,
                        native_reward_path=None,
                        custom_reward=None,
                        custom_reward_parameters_json=None,
                        native_action_path=None,
                        action_predicates_json="[{\"name\":\"valid_action\"},{\"name\":\"bounds\"},{\"name\":\"unoccupied\"}]".to_string(),
                        action_outcomes_json=None,
                        action_history_length=16,
                        box_radius=None))]
    pub fn new(
        max_steps: u32,
        trail_mode: bool,
        geometry: Option<Vec<(u16, u16, u16)>>,
        grid_size: u16,
        step_cost: f32,
        goal_reward: f32,
        distance_shaping: f32,
        collision_cost: f32,
        distance_reward_mode: String,
        zone_reward_min: f32,
        zone_reward_max: f32,
        zone_reward_curve: String,
        invalid_action_cost: f32,
        construction_residual_weight: f32,
        construction_overshoot_weight: f32,
        construction_target: Option<Vec<(u16, u16, u16)>>,
        max_consecutive_collisions: Option<u32>,
        terminate_on_success: bool,
        success_targets: Option<Vec<(u16, u16, u16)>>,
        goal_tolerance: f32,
        native_reward_path: Option<String>,
        custom_reward: Option<String>,
        custom_reward_parameters_json: Option<String>,
        native_action_path: Option<String>,
        action_predicates_json: String,
        action_outcomes_json: Option<String>,
        action_history_length: usize,
        box_radius: Option<u32>,
    ) -> PyResult<Self> {
        if let Some(radius) = box_radius {
            validate_box_radius(radius)?;
        }
        let reward_config = reward_config(
            step_cost,
            goal_reward,
            distance_shaping,
            collision_cost,
            invalid_action_cost,
            construction_residual_weight,
            construction_overshoot_weight,
            &distance_reward_mode,
            zone_reward_min,
            zone_reward_max,
            &zone_reward_curve,
        )
        .map_err(PyValueError::new_err)?;
        let env = VoxelEnv::new(WorldState::new(), max_steps)
            .with_grid_size(grid_size)
            .with_geometry(geometry.unwrap_or_default())
            .with_trail_mode(trail_mode)
            .with_reward_config(reward_config)
            .with_construction_target(construction_target.unwrap_or_default())
            .with_max_consecutive_collisions(max_consecutive_collisions)
            .with_terminate_on_success(terminate_on_success)
            .with_success_contract(success_targets.unwrap_or_default(), goal_tolerance);
        let env = match (native_reward_path, custom_reward) {
            (Some(path), Some(name)) => env
                .with_native_reward(
                    Path::new(&path),
                    &name,
                    custom_reward_parameters_json.unwrap_or_else(|| "{}".to_owned()),
                )
                .map_err(PyValueError::new_err)?,
            (None, None) | (None, Some(_)) => env,
            (Some(_), None) => {
                return Err(PyValueError::new_err(
                    "native_reward_path requires custom_reward",
                ))
            }
        };
        let mut env = env;
        let action_outcomes_json =
            action_outcomes_json.unwrap_or_else(|| default_action_outcomes(trail_mode));
        env.configure_action_pipeline(
            &action_predicates_json,
            &action_outcomes_json,
            action_history_length,
            native_action_path.as_deref().map(Path::new),
        )
        .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner: env,
            box_radius,
        })
    }

    /// Reset the environment and return the initial observation.
    /// Cursor is placed at the randomly-selected (or fixed) start position.
    pub fn reset(&mut self, seed: u64) -> PyVoxelObservation {
        let obs = self.inner.reset(seed);
        self.to_py_observation(obs)
    }

    /// Invoke a native scenario-v2 provider directly against the Rust world.
    #[pyo3(signature=(library_path, provider_name, seed, episode_index, scope, action_mode, action_offsets_json, previous_scenario_json, curriculum_json, parameters_json))]
    pub fn generate_native_scenario_v2(
        &self,
        library_path: String,
        provider_name: String,
        seed: u64,
        episode_index: u64,
        scope: String,
        action_mode: String,
        action_offsets_json: String,
        previous_scenario_json: String,
        curriculum_json: String,
        parameters_json: String,
    ) -> PyResult<String> {
        use crate::voxel::scenarios::{NativeScenarioV2, ScenarioInvocationV2};
        let extension = NativeScenarioV2::load(Path::new(&library_path), &provider_name)
            .map_err(PyValueError::new_err)?;
        extension
            .invoke(
                self.inner.world(),
                &ScenarioInvocationV2 {
                    seed,
                    episode_index,
                    grid_size: u32::from(self.inner.grid_size),
                    scope: &scope,
                    action_mode: &action_mode,
                    action_offsets_json: &action_offsets_json,
                    previous_scenario_json: &previous_scenario_json,
                    curriculum_json: &curriculum_json,
                    parameters_json: &parameters_json,
                },
            )
            .map_err(PyValueError::new_err)
    }

    /// Step the environment.
    /// Action space: Discrete(26) â€” all 26 neighbors in {-1,0,1}Â³ \ {origin},
    /// enumerated as (dx,dy,dz) with dz inner loop, skip (0,0,0) at index 13.
    pub fn step(&mut self, action: i32) -> PyResult<PyStepResultVoxel> {
        let previous_cursor = self.inner.cursor();
        let invalid_action = !(0..=26).contains(&action);
        if invalid_action {
            return Err(PyValueError::new_err(format!(
                "action {action} out of range: expected an integer in 0..=26"
            )));
        }
        self.inner
            .prepare_navigation_step(action, previous_cursor, invalid_action);
        let rust_action = self.inner.execute_navigation_action(action);
        let result = self.inner.step(rust_action);
        if let Some(error) = self.inner.take_reward_error() {
            return Err(PyValueError::new_err(error));
        }
        Ok(PyStepResultVoxel {
            obs: self.to_py_observation(result.observation),
            reward: result.reward,
            done: result.done,
            terminated: self.inner.last_terminated(),
            truncated: self.inner.last_truncated(),
            collision: self.inner.last_collision(),
            goal_reached: self.inner.last_goal_reached(),
            consecutive_collisions: self.inner.consecutive_collisions(),
            termination_reason: self.inner.last_termination_reason().to_owned(),
            reward_breakdown: self.inner.last_reward_breakdown().clone(),
            goal_distance_l2: self.inner.last_goal_distance_l2(),
            construction_residual: self.inner.last_construction_residual(),
            construction_overshoot: self.inner.last_construction_overshoot(),
        })
    }

    /// Return feasibility for all 26 canonical moves plus no-op.
    pub fn action_mask(&mut self) -> PyResult<Vec<u8>> {
        let mask = self.inner.action_mask();
        if let Some(error) = self.inner.take_reward_error() {
            return Err(PyValueError::new_err(error));
        }
        Ok(mask)
    }
    /// Returns the current cursor position as (x, y, z) in [1, 32].
    pub fn cursor_pos(&self) -> (u16, u16, u16) {
        self.inner.cursor()
    }

    /// Returns the active goal position for this episode, or None if unset.
    pub fn goal_pos(&self) -> Option<(u16, u16, u16)> {
        self.inner.active_goal()
    }

    /// Fix start and goal positions for all subsequent episodes.
    /// Overrides random surface-cell selection.
    #[pyo3(signature = (start, goal, segment_length=None))]
    pub fn set_waypoints(
        &mut self,
        start: (u16, u16, u16),
        goal: (u16, u16, u16),
        segment_length: Option<u32>,
    ) {
        self.inner
            .set_waypoints_with_segment_length(start, goal, segment_length.unwrap_or(0));
    }

    #[pyo3(signature = (goal, segment_length=None))]
    pub fn set_goal(
        &mut self,
        goal: (u16, u16, u16),
        segment_length: Option<u32>,
    ) -> PyVoxelObservation {
        let observation = self
            .inner
            .set_active_goal_with_segment_length(goal, segment_length.unwrap_or(0));
        self.to_py_observation(observation)
    }

    /// Clear fixed waypoints so random selection resumes.
    pub fn clear_waypoints(&mut self) {
        self.inner.clear_waypoints();
    }

    /// Replace geometry in-place without reinstantiating the env.
    /// Each element of `filled_cells` is (x, y, z) in [1, grid_size]Â³.
    /// Clears all filled cells (geometry + trail) and recomputes surface cells.
    /// Call reset() afterwards to start a fresh episode on the new geometry.
    pub fn set_geometry(&mut self, filled_cells: Vec<(u16, u16, u16)>) {
        self.inner.set_geometry(filled_cells);
    }

    /// Returns a flattened (2*radius+1)Â³ binary array centred on the cursor.
    /// Each element is 1.0 if the world cell at (cursor + offset) is filled, else 0.0.
    /// Cells outside the configured grid are reported as filled.
    /// Elements are ordered x-outer, y-mid, z-inner (x changes slowest).
    pub fn box_obs(&self, radius: u32) -> PyResult<Vec<f32>> {
        validate_box_radius(radius)?;
        Ok(self.build_box_obs(radius))
    }

    /// Returns a 26-element proximity array, one per non-self direction
    /// (dx, dy, dz) âˆˆ {-1,0,+1}Â³ \ {(0,0,0)} sorted lexicographically on
    /// (dx+1, dy+1, dz+1).
    ///
    /// Encoding:
    ///   0.0                              â€” no hit within max_len
    ///   1.0 - (d - 1) as f32 / max_len  â€” hit at step d  (d âˆˆ 1..=max_len)
    ///
    /// Lower value = no detection; higher = closer hit.
    pub fn radial_obs(&self, max_len: u32) -> Vec<f32> {
        let (cx, cy, cz) = self.inner.cursor();
        let g = i32::from(self.inner.grid_size);
        let mut result = Vec::with_capacity(26);
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                for dz in -1i32..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let mut proximity = 0.0f32;
                    for step in 1..=max_len {
                        let nx = cx as i32 + dx * step as i32;
                        let ny = cy as i32 + dy * step as i32;
                        let nz = cz as i32 + dz * step as i32;
                        if nx < 1 || ny < 1 || nz < 1 || nx > g || ny > g || nz > g {
                            proximity = 1.0 - (step - 1) as f32 / max_len as f32;
                            break;
                        }
                        if self
                            .inner
                            .world()
                            .is_filled((nx as u16, ny as u16, nz as u16))
                        {
                            proximity = 1.0 - (step - 1) as f32 / max_len as f32;
                            break;
                        }
                    }
                    result.push(proximity);
                }
            }
        }
        result
    }

    /// Returns a 26-element kind array aligned with the action-space ray order.
    /// Each value is the ``Block.kind`` of the first filled voxel hit in that
    /// direction, or 0.0 when no hit is detected within ``max_len``.
    pub fn radial_obs_types(&self, max_len: u32) -> Vec<f32> {
        let (cx, cy, cz) = self.inner.cursor();
        let g = i32::from(self.inner.grid_size);
        let mut result = Vec::with_capacity(26);
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                for dz in -1i32..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    let mut kind = 0.0f32;
                    for step in 1..=max_len {
                        let nx = cx as i32 + dx * step as i32;
                        let ny = cy as i32 + dy * step as i32;
                        let nz = cz as i32 + dz * step as i32;
                        if nx < 1 || ny < 1 || nz < 1 || nx > g || ny > g || nz > g {
                                kind = f32::from(BLOCK_KIND_OCCUPIED);
                            break;
                        }
                        if let Some(block) = self
                            .inner
                            .world()
                            .get_block((nx as u16, ny as u16, nz as u16))
                        {
                            kind = f32::from(block.kind);
                            break;
                        }
                    }
                    result.push(kind);
                }
            }
        }
        result
    }

    /// Returns all currently filled voxel coordinates as a list of (x, y, z) tuples.
    /// Used to snapshot the world state at episode start for trajectory recording.
    pub fn filled_voxels(&self) -> Vec<(u16, u16, u16)> {
        self.inner.world().iter_filled().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        voxel::world::{Block, Coord, World, WorldState},
        voxel::VoxelEnv,
    };

    fn env_at_cursor(cursor: (u16, u16, u16)) -> PyVoxelEnv {
        let mut inner = VoxelEnv::new(WorldState::new(), 100);
        inner.set_cursor(cursor);
        PyVoxelEnv {
            inner,
            box_radius: None,
        }
    }

    fn env_with_block(cursor: (u16, u16, u16), coord: Coord) -> PyVoxelEnv {
        let mut world = WorldState::new();
        world.set_block(coord, Block::default()).unwrap();
        let mut inner = VoxelEnv::new(world, 100);
        inner.set_cursor(cursor);
        PyVoxelEnv {
            inner,
            box_radius: None,
        }
    }

    fn env_with_block_kind(cursor: (u16, u16, u16), coord: Coord, kind: u8) -> PyVoxelEnv {
        let mut world = WorldState::new();
        world
            .set_block(
                coord,
                Block {
                    kind,
                    ..Block::default()
                },
            )
            .unwrap();
        let mut inner = VoxelEnv::new(world, 100);
        inner.set_cursor(cursor);
        PyVoxelEnv {
            inner,
            box_radius: None,
        }
    }

    #[test]
    fn action_26_is_noop_not_collision() {
        let reward_config = crate::voxel::RewardConfig {
            collision_cost: -1.0,
            ..Default::default()
        };
        let mut inner = VoxelEnv::new(WorldState::new(), 10).with_reward_config(reward_config);
        inner.set_cursor((5, 5, 5));
        let mut env = PyVoxelEnv {
            inner,
            box_radius: None,
        };

        let result = env.step(26).unwrap();

        assert_eq!(env.cursor_pos(), (5, 5, 5));
        assert!((result.reward + 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn goal_direction_is_a_unit_vector_for_axis_and_diagonal_goals() {
        assert_eq!(unit_goal_direction((4, 4, 4), (4, 4, 6)), (0.0, 0.0, 1.0));

        let direction = unit_goal_direction((4, 4, 4), (6, 6, 6));
        let expected = 1.0 / 3.0f32.sqrt();
        assert!((direction.0 - expected).abs() < 1e-6);
        assert!((direction.1 - expected).abs() < 1e-6);
        assert!((direction.2 - expected).abs() < 1e-6);
    }

    #[test]
    fn goal_direction_is_distance_invariant_and_zero_at_goal() {
        assert_eq!(
            unit_goal_direction((4, 4, 4), (5, 4, 4)),
            unit_goal_direction((4, 4, 4), (20, 4, 4))
        );
        assert_eq!(unit_goal_direction((4, 4, 4), (4, 4, 4)), (0.0, 0.0, 0.0));
    }

    #[test]
    fn step_rejects_negative_action_before_execute() {
        let mut env = env_at_cursor((5, 5, 5));
        let result = env.step(-1);
        assert!(result.is_err());
    }

    #[test]
    fn step_rejects_action_above_range_before_execute() {
        let mut env = env_at_cursor((5, 5, 5));
        let result = env.step(27);
        assert!(result.is_err());
    }
    // ----- box_obs -----

    #[test]
    fn box_obs_empty_world_all_zeros() {
        let env = env_at_cursor((16, 16, 16));
        let obs = env.box_obs(2).unwrap();
        assert_eq!(obs.len(), 125);
        assert!(obs.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn box_obs_radius_0_single_cell() {
        let env = env_at_cursor((1, 1, 1)); // cursor at (1,1,1), empty world
        let obs = env.box_obs(0).unwrap();
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0], 0.0);
    }

    #[test]
    fn box_obs_preserves_filled_kind_at_cursor() {
        // cursor at (1,1,1)
        let env = env_with_block((5, 5, 5), (5, 5, 5));
        let obs = env.box_obs(2).unwrap(); // 5Â³ = 125 cells
                                  // centre element: dx=0,dy=0,dz=0 â†’ flat index = 2*25 + 2*5 + 2 = 62
        assert_eq!(obs[62], 5.0);
        let non_centre: Vec<f32> = obs
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != 62)
            .map(|(_, &v)| v)
            .collect();
        assert!(non_centre.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn box_obs_preserves_every_stored_block_kind() {
        for kind in 1..=5 {
            let env = env_with_block_kind((5, 5, 5), (5, 5, 5), kind);
            assert_eq!(env.box_obs(0).unwrap(), vec![f32::from(kind)]);
        }
    }

    #[test]
    fn box_obs_out_of_grid_is_blocked() {
        // cursor at (1,1,1); radius 2 â†’ cells at coord 0 and -1 are OOB or edge
        // cell at (1-2, 1, 1) = (-1, 1, 1) â€” out of bounds (negative i32)
        let env = env_at_cursor((1, 1, 1)); // (1,1,1)
        let obs = env.box_obs(2).unwrap();
        // Outside-grid cells intentionally look identical to occupied geometry.
        assert_eq!(obs[0], 1.0);
    }

    #[test]
    fn box_obs_length_correct_for_radius() {
        let env = env_at_cursor((1, 1, 1));
        assert_eq!(env.box_obs(0).unwrap().len(), 1);
        assert_eq!(env.box_obs(1).unwrap().len(), 27);
        assert_eq!(env.box_obs(2).unwrap().len(), 125);
        assert_eq!(env.box_obs(3).unwrap().len(), 343);
    }

    #[test]
    fn box_obs_excessive_radius_returns_value_error() {
        let env = env_at_cursor((16, 16, 16));
        assert!(env.box_obs(MAX_BOX_RADIUS + 1).is_err());
        assert!(validate_box_radius(MAX_BOX_RADIUS + 1).is_err());
    }

    #[test]
    fn box_obs_radius_near_i32_overflow_returns_value_error_not_panic() {
        let env = env_at_cursor((16, 16, 16));
        // Without the bounds check, `2 * (radius as i32) + 1` overflows i32
        // and the subsequent `as usize` cast produces a huge allocation size.
        assert!(env.box_obs(u32::MAX / 2).is_err());
    }

    #[test]
    fn new_rejects_excessive_box_radius() {
        let result = PyVoxelEnv::new(
            10,
            true,
            None,
            32,
            -0.01,
            1.0,
            0.0,
            0.0,
            "progress".to_string(),
            -1.0,
            -0.01,
            "linear".to_string(),
            0.0,
            0.0,
            0.0,
            None,
            None,
            true,
            None,
            0.0,
            None,
            None,
            None,
            None,
            "[{\"name\":\"valid_action\"},{\"name\":\"bounds\"},{\"name\":\"unoccupied\"}]".to_string(),
            None,
            16,
            Some(MAX_BOX_RADIUS + 1),
        );
        assert!(result.is_err());
    }

    // ----- radial_obs -----

    #[test]
    fn radial_obs_length_is_26() {
        let env = env_at_cursor((1, 1, 1));
        assert_eq!(env.radial_obs(16).len(), 26);
    }

    #[test]
    fn radial_obs_empty_world_all_zeros() {
        let env = env_at_cursor((16, 16, 16));
        let obs = env.radial_obs(4);
        assert!(obs.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn radial_obs_adjacent_hit_proximity_is_1() {
        // cursor at (5,5,5), fill (6,5,5) â€” direction (+1,0,0), step d=1
        let env = env_with_block((5, 5, 5), (6, 5, 5));
        let obs = env.radial_obs(16);
        // direction (+1,0,0): index 21 in the 26-direction non-self ordering
        assert_eq!(obs[21], 1.0);
    }

    #[test]
    fn radial_obs_distant_hit() {
        // cursor at (5,5,5), fill (13,5,5) â€” direction (+1,0,0), step d=8, max_len=16
        // expected: 1.0 - (8-1)/16.0 = 1.0 - 7/16.0 = 0.5625
        let env = env_with_block((5, 5, 5), (13, 5, 5));
        let obs = env.radial_obs(16);
        // direction (+1,0,0): index 21
        let expected = 1.0 - 7.0f32 / 16.0;
        assert!((obs[21] - expected).abs() < 1e-6);
    }

    #[test]
    fn radial_obs_hit_at_max_len_distinguishable() {
        // cursor at (5,5,5), fill (5+16,5,5) = (21,5,5) â€” direction (+1,0,0), d=16
        // expected: 1.0 - (16-1)/16.0 = 1/16.0 > 0
        let env = env_with_block((5, 5, 5), (21, 5, 5));
        let obs = env.radial_obs(16);
        let expected = 1.0f32 / 16.0;
        assert!((obs[21] - expected).abs() < 1e-6);
        assert!(obs[21] > 0.0);
    }

    #[test]
    fn radial_obs_no_hit_is_zero() {
        // empty world, direction (+1,0,0) should be 0.0
        let env = env_at_cursor((5, 5, 5));
        let obs = env.radial_obs(4);
        assert_eq!(obs[21], 0.0);
    }

    #[test]
    fn radial_obs_grid_boundary_is_blocked() {
        let env = env_at_cursor((1, 5, 5));
        let obs = env.radial_obs(16);
        let types = env.radial_obs_types(16);
        assert_eq!(obs[4], 1.0);
        assert_eq!(types[4], f32::from(BLOCK_KIND_OCCUPIED));
    }
}
