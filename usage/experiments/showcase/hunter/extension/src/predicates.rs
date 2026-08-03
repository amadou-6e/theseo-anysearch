use anysearch_extension::{anysearch_predicate, PredicateContext, PredicateResult};

fn multiplier(context: &PredicateContext) -> i32 {
    context
        .parameters
        .get("multiplier")
        .and_then(|value| value.as_i64())
        .unwrap_or(2) as i32
}

#[anysearch_predicate]
pub fn double_step_in_bounds(context: &PredicateContext) -> PredicateResult {
    let multiplier = multiplier(context);
    let offset = [
        context.destination[0] - context.cursor[0],
        context.destination[1] - context.cursor[1],
        context.destination[2] - context.cursor[2],
    ];
    let grid_size = i32::from(context.grid_size);
    let target = [
        context.cursor[0] + multiplier * offset[0],
        context.cursor[1] + multiplier * offset[1],
        context.cursor[2] + multiplier * offset[2],
    ];
    if context.valid_action
        && target
            .iter()
            .all(|coordinate| (1..=grid_size).contains(coordinate))
    {
        PredicateResult::allow()
    } else {
        PredicateResult::deny()
    }
}