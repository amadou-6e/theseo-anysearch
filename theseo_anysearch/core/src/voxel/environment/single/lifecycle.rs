use super::*;

// ---------------------------------------------------------------------------
// Environment trait implementation
// ---------------------------------------------------------------------------

impl Environment for VoxelEnv {
    type Action = VoxelAction;
    type Observation = VoxelObservation;

    fn reset(&mut self, seed: u64) -> Self::Observation {
        self.steps = 0;
        self.segment_steps = 0;
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
                    kind: crate::voxel::world::BLOCK_KIND_OCCUPIED,
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
        self.segment_steps += 1;
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

        let (new_l2, has_accepted_target) = if self.success_targets.is_empty() {
            (
                self.active_goal
                    .map(|goal| l2(self.cursor, goal))
                    .unwrap_or(0.0),
                self.active_goal.is_some(),
            )
        } else {
            (
                self.success_targets
                    .iter()
                    .map(|goal| l2(self.cursor, *goal))
                    .reduce(f32::min)
                    .unwrap_or(0.0),
                true,
            )
        };
        let goal_reached = has_accepted_target && new_l2 <= self.goal_tolerance;
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
            (
                reward_components::distance::NAME.to_owned(),
                distance_reward,
            ),
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
            segment_step: u64::from(self.segment_steps),
            segment_length: u64::from(self.segment_length),
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

pub(crate) fn merge_mutations(
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
        merge(
            &mut target.cursor,
            coord(result.cursor, grid_size)?,
            "cursor",
        )?;
    }
    if result.place_voxel != 0 {
        merge(
            &mut target.place,
            coord(result.place_coord, grid_size)?,
            "place",
        )?;
    }
    if result.remove_voxel != 0 {
        merge(
            &mut target.remove,
            coord(result.remove_coord, grid_size)?,
            "remove",
        )?;
    }
    Ok(())
}
