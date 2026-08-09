use super::*;

impl VoxelEnv {
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
                    return NativePredicateExtension::load(path, &spec.name, parameters.clone())
                        .map(ConfiguredPredicate::Native)
                        .map_err(|error| {
                            format!(
                                "failed to load native predicate {:?} from {}: {error}",
                                spec.name,
                                path.display()
                            )
                        });
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
                    return NativeOutcomeExtension::load(path, &spec.name, parameters.clone())
                        .map(ConfiguredOutcome::Native)
                        .map_err(|error| {
                            format!(
                                "failed to load native outcome {:?} from {}: {error}",
                                spec.name,
                                path.display()
                            )
                        });
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
            let offset = OFFSETS_26[action_index as usize];
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
                #[cfg(test)]
                ConfiguredPredicate::Failing(message) => Err(message.clone())?,
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
            if let Err(error) =
                super::lifecycle::merge_mutations(&mut mutations, result, self.grid_size)
            {
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
}
