use anysearch_extension::{anysearch_reward, RewardContext, RewardResult};

#[anysearch_reward]
pub fn native_collision(context: &RewardContext) -> RewardResult {
    let configured_penalty = context
        .parameters
        .get("collision_penalty")
        .and_then(|value| value.as_f64())
        .unwrap_or(-0.02);
    let penalty = if context.collision {
        configured_penalty
    } else {
        0.0
    };
    RewardResult::add(penalty).with_component("native_collision", penalty)
}
