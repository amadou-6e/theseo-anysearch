use anysearch_extension::{anysearch_scenario, ScenarioContext, ScenarioResult};

#[anysearch_scenario]
pub fn adjacent_goal_rust(context: &ScenarioContext) -> ScenarioResult {
    let center = i32::from(context.grid_size + 1) / 2;
    let seed_base = context
        .parameters
        .get("seed_base")
        .and_then(|value| value.as_u64())
        .unwrap_or(0);
    let index = if context.scope == "evaluation" {
        context.seed.saturating_sub(seed_base) as usize % context.action_offsets.len()
    } else {
        context.seed as usize % context.action_offsets.len()
    };
    let offset = context.action_offsets[index];
    ScenarioResult::goal(
        [center, center, center],
        [center + offset[0], center + offset[1], center + offset[2]],
        format!("adjacent-{index:02}"),
    )
}
