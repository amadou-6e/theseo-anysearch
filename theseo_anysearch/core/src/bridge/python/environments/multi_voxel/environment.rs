use pyo3::{exceptions::PyValueError, prelude::*};

use crate::voxel::world::{
    BoundedRegion, StorageCoord, World, WorldExtent, WorldRead, WorldResidency, WorldState,
    BLOCK_KIND_GOAL, BLOCK_KIND_OCCUPIED,
};
use std::path::Path;

use super::models::{PyMultiVoxelObs, PyMultiVoxelStepResult};

/// Upper bound on `box_radius`: keeps `2 * r + 1` well within `i32` range so
/// the side-length computation in `box_obs` can never overflow, and keeps
/// the resulting allocation ((2r+1)^3 f32s) within sane memory limits.
const MAX_BOX_RADIUS: u32 = 1024;

fn validate_box_radius(radius: u32) -> PyResult<()> {
    if radius > MAX_BOX_RADIUS {
        return Err(crate::bridge::python::errors::invalid_value(format!(
            "box_radius must be <= {MAX_BOX_RADIUS}, got {radius}"
        )));
    }
    Ok(())
}

#[pyclass]
pub struct PyMultiVoxelEnv {
    inner: crate::voxel::MultiAgentVoxelEnv,
    residency_radius: u32,
    pending_prefetch: Vec<crate::voxel::world::PrefetchRequest>,
}

fn cursor_region(cursor: (u16, u16, u16), radius: u32, extent: WorldExtent) -> BoundedRegion {
    let center = StorageCoord {
        x: u32::from(cursor.0).min(extent.x - 1),
        y: u32::from(cursor.1).min(extent.y - 1),
        z: u32::from(cursor.2).min(extent.z - 1),
    };
    BoundedRegion::new(
        StorageCoord {
            x: center.x.saturating_sub(radius),
            y: center.y.saturating_sub(radius),
            z: center.z.saturating_sub(radius),
        },
        StorageCoord {
            x: center
                .x
                .saturating_add(radius)
                .saturating_add(1)
                .min(extent.x),
            y: center
                .y
                .saturating_add(radius)
                .saturating_add(1)
                .min(extent.y),
            z: center
                .z
                .saturating_add(radius)
                .saturating_add(1)
                .min(extent.z),
        },
        extent,
    )
    .expect("cursor clamped to a positive finite extent produces a valid region")
}

#[pymethods]
impl PyMultiVoxelEnv {
    #[new]
    #[pyo3(signature = (agent_count, max_steps, trail_mode=true, geometry=None,
                        grid_size=32, extent=None, step_cost=-0.01, goal_reward=1.0,
                        distance_shaping=0.0, collision_cost=0.0,
                        distance_reward_mode="progress".to_string(),
                        zone_reward_min=-1.0, zone_reward_max=-0.01,
                        zone_reward_curve="linear".to_string(), agents_json=None,
                        hunter_and_hunted_json=None, native_action_path=None))]
    pub fn new(
        agent_count: usize,
        max_steps: u32,
        trail_mode: bool,
        geometry: Option<Vec<(u16, u16, u16)>>,
        grid_size: u16,
        extent: Option<(u16, u16, u16)>,
        step_cost: f32,
        goal_reward: f32,
        distance_shaping: f32,
        collision_cost: f32,
        distance_reward_mode: String,
        zone_reward_min: f32,
        zone_reward_max: f32,
        zone_reward_curve: String,
        agents_json: Option<String>,
        hunter_and_hunted_json: Option<String>,
        native_action_path: Option<String>,
    ) -> PyResult<Self> {
        let distance_reward_mode =
            crate::voxel::DistanceRewardMode::from_name(distance_reward_mode.as_str()).ok_or_else(
                || PyValueError::new_err("distance_reward_mode must be 'progress' or 'zone'"),
            )?;
        let zone_reward_curve =
            crate::voxel::ZoneRewardCurve::from_name(zone_reward_curve.as_str()).ok_or_else(
                || PyValueError::new_err("zone_reward_curve must be 'linear' or 'exponential'"),
            )?;
        let reward_config = crate::voxel::RewardConfig {
            step_cost,
            goal_reward,
            distance_shaping,
            collision_cost,
            invalid_action_cost: 0.0,
            construction_residual_weight: 0.0,
            construction_overshoot_weight: 0.0,
            distance_reward_mode,
            zone_reward_min,
            zone_reward_max,
            zone_reward_curve,
        };
        reward_config
            .validate_finite()
            .map_err(PyValueError::new_err)?;
        let resolved_extent = extent.unwrap_or((grid_size, grid_size, grid_size));
        if resolved_extent.0 == 0 || resolved_extent.1 == 0 || resolved_extent.2 == 0 {
            return Err(PyValueError::new_err("extent axes must be positive"));
        }
        let mut inner = crate::voxel::MultiAgentVoxelEnv::new(
            agent_count,
            max_steps,
            trail_mode,
            geometry.unwrap_or_default(),
            reward_config,
            grid_size,
        )
        .with_extent([resolved_extent.0, resolved_extent.1, resolved_extent.2]);
        if let Some(agents_json) = agents_json {
            inner
                .configure_agents(
                    &agents_json,
                    native_action_path.as_deref().map(std::path::Path::new),
                )
                .map_err(PyValueError::new_err)?;
        }
        inner
            .configure_capture_task(hunter_and_hunted_json.as_deref())
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            inner,
            residency_radius: 3,
            pending_prefetch: Vec::new(),
        })
    }

    pub fn reset(&mut self, seed: u64) -> PyResult<PyMultiVoxelObs> {
        let (steps_remaining, voxel_count, cursors, goal_distances) = self.inner.reset(seed);
        for cursor in &cursors {
            let region = cursor_region(*cursor, self.residency_radius, self.inner.world.extent());
            self.inner
                .world
                .prefetch_region(region)
                .map_err(|error| PyValueError::new_err(format!("{error:?}")))?;
        }
        Ok(PyMultiVoxelObs {
            steps_remaining,
            voxel_count,
            cursors,
            goal_distances,
        })
    }

    /// `actions` must be a list of length == agent_count; each value is 0..25.
    pub fn step(&mut self, actions: Vec<i32>) -> PyResult<PyMultiVoxelStepResult> {
        check_actions_len(actions.len(), self.inner.agents.len())?;
        for request in self.pending_prefetch.drain(..) {
            request
                .wait()
                .map_err(|error| PyValueError::new_err(format!("{error:?}")))?;
        }
        let handle = self.inner.world.world_handle();
        let guards = self
            .inner
            .agents
            .iter()
            .map(|agent| {
                handle.pin_region(cursor_region(
                    agent.cursor,
                    self.residency_radius,
                    self.inner.world.extent(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| PyValueError::new_err(format!("{error:?}")))?;
        let r = self.inner.step_all(&actions);
        drop(guards);
        self.pending_prefetch = self
            .inner
            .agents
            .iter()
            .filter_map(|agent| {
                self.inner.world.request_prefetch_region(cursor_region(
                    agent.cursor,
                    self.residency_radius,
                    self.inner.world.extent(),
                ))
            })
            .collect();
        Ok(PyMultiVoxelStepResult {
            observation: PyMultiVoxelObs {
                steps_remaining: r.steps_remaining,
                voxel_count: r.voxel_count,
                cursors: r.cursors,
                goal_distances: r.goal_distances,
            },
            rewards: r.rewards,
            done: r.done,
        })
    }

    pub fn agent_count(&self) -> usize {
        self.inner.agent_count()
    }

    /// Returns all currently filled voxel coordinates (geometry + agent trail).
    pub fn filled_voxels(&self) -> Vec<(u16, u16, u16)> {
        self.inner.world.iter_filled().collect()
    }

    /// Return the resolved episode overlay without enumerating immutable base geometry.
    pub fn overlay_mutations(&self) -> Vec<(u32, u32, u32, bool, u8, bool, f32)> {
        self.inner
            .world
            .overlay_mutations()
            .into_iter()
            .map(|mutation| match mutation.block {
                Some(block) => (
                    mutation.coordinate.x,
                    mutation.coordinate.y,
                    mutation.coordinate.z,
                    true,
                    block.kind,
                    block.active,
                    block.reward_weight,
                ),
                None => (
                    mutation.coordinate.x,
                    mutation.coordinate.y,
                    mutation.coordinate.z,
                    false,
                    0,
                    false,
                    0.0,
                ),
            })
            .collect()
    }

    /// Returns each agent's current cursor position.
    pub fn cursor_positions(&self) -> Vec<(u16, u16, u16)> {
        self.inner.agents.iter().map(|a| a.cursor).collect()
    }

    /// Returns each agent's current goal position, or None if unset.
    pub fn goal_positions(&self) -> Vec<Option<(u16, u16, u16)>> {
        self.inner.agents.iter().map(|a| a.goal).collect()
    }

    /// Replace geometry in-place without reinstantiating the env.
    /// Each element of `filled_cells` is (x, y, z) in [1, grid_size]Â³.
    /// Clears all filled cells (geometry + trail) and recomputes surface cells.
    /// Call reset() afterwards to start a fresh episode on the new geometry.
    pub fn set_geometry(&mut self, filled_cells: Vec<(u16, u16, u16)>) {
        self.inner.set_geometry(filled_cells);
    }

    pub fn set_compiled_world(
        &mut self,
        root: String,
        maximum_decoded_bytes: usize,
    ) -> PyResult<()> {
        let world = WorldState::from_compiled_pack(Path::new(&root), maximum_decoded_bytes)
            .map_err(|error| PyValueError::new_err(format!("{error:?}")))?;
        self.inner.replace_world(world);
        Ok(())
    }

    pub fn set_world_residency_radius(&mut self, radius: u32) {
        self.residency_radius = radius;
    }

    pub fn world_cache_metrics(&self) -> Option<std::collections::HashMap<String, u64>> {
        self.inner.world.disk_cache_metrics().map(|metrics| {
            [
                ("cache_hits".to_owned(), metrics.cache_hits),
                ("cache_misses".to_owned(), metrics.cache_misses),
                ("pack_reads".to_owned(), metrics.pack_reads),
                ("evictions".to_owned(), metrics.evictions),
                ("decoded_bytes".to_owned(), metrics.decoded_bytes as u64),
                ("pinned_bytes".to_owned(), metrics.pinned_bytes as u64),
                (
                    "pinned_overcommit_bytes".to_owned(),
                    metrics.pinned_overcommit_bytes as u64,
                ),
                ("resident_chunks".to_owned(), metrics.resident_chunks as u64),
                ("pinned_chunks".to_owned(), metrics.pinned_chunks as u64),
            ]
            .into_iter()
            .collect()
        })
    }

    /// 6 binary values for agent `agent_idx`'s cardinal face-neighbors (+x,-x,+y,-y,+z,-z).
    /// 1.0 = that neighbor cell is filled or outside the configured grid.
    pub fn face_neighbors(&self, agent_idx: usize) -> PyResult<Vec<f32>> {
        check_agent_idx(agent_idx, self.inner.agents.len())?;
        let (cx, cy, cz) = self.inner.agents[agent_idx].cursor;
        let g = i32::from(self.inner.grid_size);
        let dirs: [(i32, i32, i32); 6] = [
            (1, 0, 0),
            (-1, 0, 0),
            (0, 1, 0),
            (0, -1, 0),
            (0, 0, 1),
            (0, 0, -1),
        ];
        Ok(dirs
            .iter()
            .map(|&(dx, dy, dz)| {
                let x = cx as i32 + dx;
                let y = cy as i32 + dy;
                let z = cz as i32 + dz;
                if x >= 1 && y >= 1 && z >= 1 && x <= g && y <= g && z <= g {
                    self.inner.world.is_filled((x as u16, y as u16, z as u16)) as u8 as f32
                } else {
                    1.0
                }
            })
            .collect())
    }

    /// Flattened (2*radius+1)Â³ binary box observation centred on agent `agent_idx`'s cursor.
    /// 1.0 = filled or outside the configured grid. Ordered x-outer, y-mid, z-inner.
    pub fn box_obs(&self, agent_idx: usize, radius: u32) -> PyResult<Vec<f32>> {
        check_agent_idx(agent_idx, self.inner.agents.len())?;
        validate_box_radius(radius)?;
        let (cx, cy, cz) = self.inner.agents[agent_idx].cursor;
        let g = i32::from(self.inner.grid_size);
        let r = radius as i32;
        let side = (2 * r + 1) as usize;
        let mut result = Vec::with_capacity(side * side * side);
        for dx in -r..=r {
            for dy in -r..=r {
                for dz in -r..=r {
                    let x = cx as i32 + dx;
                    let y = cy as i32 + dy;
                    let z = cz as i32 + dz;
                    let kind = if x >= 1 && y >= 1 && z >= 1 && x <= g && y <= g && z <= g {
                        let coord = (x as u16, y as u16, z as u16);
                        if self.inner.agents[agent_idx].goal == Some(coord) {
                            f32::from(BLOCK_KIND_GOAL)
                        } else {
                            self.inner
                                .world
                                .get_block(coord)
                                .map_or(0.0, |block| f32::from(block.kind))
                        }
                    } else {
                        f32::from(BLOCK_KIND_OCCUPIED)
                    };
                    result.push(kind);
                }
            }
        }
        Ok(result)
    }

    /// 27-element ray-cast observation from agent `agent_idx`'s cursor, one value per
    /// direction (dx,dy,dz) âˆˆ {-1,0,+1}Â³ sorted lexicographically on (dx+1,dy+1,dz+1).
    ///
    /// Encoding (distance, not proximity):
    ///   0.0                        â€” filled cell immediately adjacent (d=1) or self filled (0,0,0)
    ///   (d-1) as f32 / max_len     â€” hit at step d (d âˆˆ 2..=max_len); larger = farther
    ///   1.0                        â€” no hit within max_len, or self empty for (0,0,0)
    pub fn ray_cast(&self, agent_idx: usize, max_len: u32) -> PyResult<Vec<f32>> {
        check_agent_idx(agent_idx, self.inner.agents.len())?;
        let (cx, cy, cz) = self.inner.agents[agent_idx].cursor;
        let g = i32::from(self.inner.grid_size);
        let mut result = Vec::with_capacity(27);
        for dx in -1i32..=1 {
            for dy in -1i32..=1 {
                for dz in -1i32..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        let filled = self.inner.world.is_filled((cx, cy, cz));
                        result.push(if filled { 0.0 } else { 1.0 });
                    } else {
                        let mut value = 1.0f32;
                        for step in 1..=max_len {
                            let nx = cx as i32 + dx * step as i32;
                            let ny = cy as i32 + dy * step as i32;
                            let nz = cz as i32 + dz * step as i32;
                            if nx < 1 || ny < 1 || nz < 1 || nx > g || ny > g || nz > g {
                                value = (step - 1) as f32 / max_len as f32;
                                break;
                            }
                            if self
                                .inner
                                .world
                                .is_filled((nx as u16, ny as u16, nz as u16))
                            {
                                value = (step - 1) as f32 / max_len as f32;
                                break;
                            }
                        }
                        result.push(value);
                    }
                }
            }
        }
        Ok(result)
    }
}

/// Validates `agent_idx` against `agent_count`, returning a descriptive
/// `PyValueError` when the index is out of range.
fn check_agent_idx(agent_idx: usize, agent_count: usize) -> PyResult<()> {
    if agent_idx >= agent_count {
        return Err(PyValueError::new_err(format!(
            "agent_idx {} out of range for {} agents",
            agent_idx, agent_count
        )));
    }
    Ok(())
}

/// Validates `actions.len()` against `agent_count`, returning a descriptive
/// `PyValueError` on mismatch.
fn check_actions_len(actions_len: usize, agent_count: usize) -> PyResult<()> {
    if actions_len != agent_count {
        return Err(PyValueError::new_err(format!(
            "actions length {} does not match agent_count {}",
            actions_len, agent_count
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_agent_idx_rejects_out_of_range() {
        assert!(check_agent_idx(0, 0).is_err());
        assert!(check_agent_idx(2, 2).is_err());
        assert!(check_agent_idx(1, 2).is_ok());
    }

    #[test]
    fn check_agent_idx_accepts_in_range() {
        assert!(check_agent_idx(0, 3).is_ok());
        assert!(check_agent_idx(2, 3).is_ok());
    }

    #[test]
    fn check_actions_len_rejects_mismatch() {
        assert!(check_actions_len(1, 2).is_err());
        assert!(check_actions_len(3, 2).is_err());
    }

    #[test]
    fn check_actions_len_accepts_match() {
        assert!(check_actions_len(2, 2).is_ok());
        assert!(check_actions_len(0, 0).is_ok());
    }

    fn make_env() -> PyMultiVoxelEnv {
        PyMultiVoxelEnv::new(
            1,
            10,
            true,
            None,
            32,
            None,
            -0.01,
            1.0,
            0.0,
            0.0,
            "progress".to_string(),
            -1.0,
            -0.01,
            "linear".to_string(),
            None,
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn box_obs_valid_radius_ok() {
        let mut env = make_env();
        env.reset(0).unwrap();
        let obs = env.box_obs(0, 2).unwrap();
        assert_eq!(obs.len(), 125);
    }

    #[test]
    fn box_obs_excessive_radius_returns_value_error() {
        let mut env = make_env();
        env.reset(0).unwrap();
        assert!(env.box_obs(0, MAX_BOX_RADIUS + 1).is_err());
        assert!(validate_box_radius(MAX_BOX_RADIUS + 1).is_err());
    }

    #[test]
    fn box_obs_radius_near_i32_overflow_returns_value_error_not_panic() {
        let mut env = make_env();
        env.reset(0).unwrap();
        // Without the bounds check, `2 * (radius as i32) + 1` overflows i32
        // and the subsequent `as usize` cast produces a huge allocation size.
        assert!(env.box_obs(0, u32::MAX / 2).is_err());
    }
}
