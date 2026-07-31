use anysearch_extension::{anysearch_reward, RewardContext, RewardResult};

#[anysearch_reward]
pub fn native_collision(context: &RewardContext) -> RewardResult {
    let penalty = if context.collision { -0.02 } else { 0.0 };
    RewardResult::add(penalty).with_component("native_collision", penalty)
}
