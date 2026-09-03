use anysearch_extension::{anysearch_outcome, OutcomeContext, OutcomeMutations, OutcomeResult};

fn multiplier(context: &OutcomeContext) -> i32 {
    context
        .parameters
        .get("multiplier")
        .and_then(|value| value.as_i64())
        .unwrap_or(2) as i32
}

#[anysearch_outcome]
pub fn double_step(
    context: &OutcomeContext,
    mutations: &mut OutcomeMutations,
) -> OutcomeResult {
    let multiplier = multiplier(context);
    let offset = [
        context.destination[0] - context.cursor[0],
        context.destination[1] - context.cursor[1],
        context.destination[2] - context.cursor[2],
    ];
    mutations.set_cursor([
        context.cursor[0] + multiplier * offset[0],
        context.cursor[1] + multiplier * offset[1],
        context.cursor[2] + multiplier * offset[2],
    ]);
    OutcomeResult::applied()
}