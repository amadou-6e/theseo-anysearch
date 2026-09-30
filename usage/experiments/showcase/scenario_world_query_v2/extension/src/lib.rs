use anysearch_extension::{
    anysearch_scenario, anysearch_scenario_v2, ScenarioContext, ScenarioContextV2, ScenarioResult,
};

#[no_mangle]
pub extern "C" fn anysearch_extension_abi_version() -> u32 {
    anysearch_extension::ABI_VERSION
}
#[no_mangle]
pub extern "C" fn anysearch_extension_capabilities() -> u64 {
    32
}

/// Demonstrates point, bounded-region, count, and ray queries without voxel serialization.
#[anysearch_scenario_v2]
fn queried_route(context: &ScenarioContextV2<'_>) -> ScenarioResult {
    let _origin = context.world.point([0, 0, 0]).expect("point query");
    let _nearby = context
        .world
        .region([0, 0, 0], [8, 8, 8])
        .expect("region query");
    let _count = context
        .world
        .count([0, 0, 0], [8, 8, 8])
        .expect("count query");
    let hit = context
        .world
        .ray([0, 1, 1], [1, 0, 0], context.grid_size)
        .expect("ray query");
    let goal_x = hit.map_or(context.grid_size.saturating_sub(1), |value| {
        value.coordinate.x.saturating_sub(1)
    });
    ScenarioResult::goal(
        [1, 1, 1],
        [goal_x as i32, 1, 1],
        format!("queried-{}", context.episode_index),
    )
}

/// A v1 export in the same independently compiled library proves compatibility.
#[anysearch_scenario]
fn legacy_route(context: &ScenarioContext) -> ScenarioResult {
    ScenarioResult::goal(
        [1, 1, 1],
        [2, 1, 1],
        format!("legacy-{}", context.episode_index),
    )
}
