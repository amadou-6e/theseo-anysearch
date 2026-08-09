//! Execution of one configured heterogeneous-agent action.

use crate::voxel::{
    actions::{ConfiguredOutcome, ConfiguredPredicate, PendingMutations, OFFSETS_26},
    common::ABI_VERSION,
    outcomes::{builtins as outcome_builtins, OutcomeContextV2},
    predicates::{builtins as predicate_builtins, PredicateContextV2},
    world::{Coord, WorldState},
};

use super::multi_action::AgentActionPipeline;

pub struct AgentActionResult {
    pub cursor: Coord,
    pub collision: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn execute_agent_action(
    pipeline: &mut AgentActionPipeline,
    world: &mut WorldState,
    action_index: i32,
    cursor: Coord,
    goal: Option<Coord>,
    step: u32,
    max_steps: u32,
    grid_size: u16,
    observation_filled: usize,
) -> Result<AgentActionResult, String> {
    let cursor_raw = coord_to_i32(cursor);
    let valid_action = (0..=26).contains(&action_index);
    let destination = if (0..26).contains(&action_index) {
        let offset = OFFSETS_26[action_index as usize];
        [
            cursor_raw[0] + offset.0,
            cursor_raw[1] + offset.1,
            cursor_raw[2] + offset.2,
        ]
    } else {
        cursor_raw
    };
    let grid = i32::from(grid_size);
    let in_bounds = destination.iter().all(|value| (1..=grid).contains(value));
    let destination_coord = in_bounds.then(|| raw_to_coord(destination));
    let blocked = destination_coord.is_some_and(|coord| world.is_blocking(coord));
    let context = PredicateContextV2 {
        abi_version: ABI_VERSION,
        struct_size: std::mem::size_of::<PredicateContextV2>() as u32,
        step: u64::from(step),
        grid_size,
        action_index,
        cursor: cursor_raw,
        destination,
        goal: goal.map_or([0; 3], coord_to_i32),
        has_goal: u8::from(goal.is_some()),
        valid_action: u8::from(valid_action),
        destination_in_bounds: u8::from(in_bounds),
        destination_blocked: u8::from(blocked),
        observation_filled,
        observation_steps_remaining: max_steps.saturating_sub(step),
        observation_goal_distance: goal.map_or(0, |target| manhattan(cursor, target)),
        has_observation_goal_distance: u8::from(goal.is_some()),
        history: pipeline.history.as_ptr(),
        history_len: pipeline.history.len(),
        parameters_json: std::ptr::null(),
        parameters_json_len: 0,
    };
    for predicate in &pipeline.predicates {
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
            record(
                pipeline,
                action_index,
                cursor,
                cursor,
                false,
                action_index != 26,
            );
            return Ok(AgentActionResult {
                cursor,
                collision: action_index != 26,
            });
        }
    }
    let outcome_context = OutcomeContextV2 {
        abi_version: ABI_VERSION,
        struct_size: std::mem::size_of::<OutcomeContextV2>() as u32,
        step: u64::from(step),
        grid_size,
        action_index,
        cursor: cursor_raw,
        destination,
        goal: goal.map_or([0; 3], coord_to_i32),
        has_goal: u8::from(goal.is_some()),
        history: pipeline.history.as_ptr(),
        history_len: pipeline.history.len(),
        parameters_json: std::ptr::null(),
        parameters_json_len: 0,
    };
    let mut mutations = PendingMutations::default();
    for outcome in &pipeline.outcomes {
        let result = match outcome {
            ConfiguredOutcome::CursorMovement => {
                outcome_builtins::cursor_movement::apply(destination)
            }
            ConfiguredOutcome::TrailPlacement => {
                outcome_builtins::trail_placement::apply(action_index, destination)
            }
            ConfiguredOutcome::Place => outcome_builtins::place::apply(action_index, destination),
            ConfiguredOutcome::Remove => outcome_builtins::remove::apply(cursor_raw),
            ConfiguredOutcome::Native(extension) => {
                extension.evaluate(OutcomeContextV2 { ..outcome_context })?
            }
        };
        super::single::lifecycle::merge_mutations(&mut mutations, result, grid_size)?;
    }
    if let Some(target) = mutations.cursor {
        if world.is_blocking(target) && target != cursor {
            record(pipeline, action_index, cursor, cursor, false, true);
            return Ok(AgentActionResult {
                cursor,
                collision: true,
            });
        }
    }
    if let Some(coord) = mutations.remove {
        world.set(coord, false);
    }
    let cursor = mutations.cursor.unwrap_or(cursor);
    if let Some(coord) = mutations.place {
        world.set(coord, true);
    }
    Ok(AgentActionResult {
        cursor,
        collision: false,
    })
}

fn coord_to_i32(coord: Coord) -> [i32; 3] {
    [i32::from(coord.0), i32::from(coord.1), i32::from(coord.2)]
}

fn raw_to_coord(raw: [i32; 3]) -> Coord {
    (raw[0] as u16, raw[1] as u16, raw[2] as u16)
}

fn manhattan(a: Coord, b: Coord) -> u32 {
    u32::from(a.0.abs_diff(b.0) + a.1.abs_diff(b.1) + a.2.abs_diff(b.2))
}

fn record(
    pipeline: &mut AgentActionPipeline,
    action_index: i32,
    previous: Coord,
    cursor: Coord,
    feasible: bool,
    collision: bool,
) {
    if pipeline.history_length == 0 {
        return;
    }
    pipeline
        .history
        .push(crate::voxel::actions::ActionHistoryEntryV2 {
            action_index,
            previous_cursor: coord_to_i32(previous),
            cursor: coord_to_i32(cursor),
            feasible: u8::from(feasible),
            collision: u8::from(collision),
        });
    let excess = pipeline
        .history
        .len()
        .saturating_sub(pipeline.history_length);
    if excess > 0 {
        pipeline.history.drain(0..excess);
    }
}
