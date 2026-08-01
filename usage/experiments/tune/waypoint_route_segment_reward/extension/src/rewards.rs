use anysearch_extension::{anysearch_reward, RewardContext, RewardResult};

const MIN_REWARD: u64 = 1;

fn reward_value(goal_reached: bool, segment_length: u64, segment_step: u64) -> f64 {
    if !goal_reached {
        return 0.0;
    }
    segment_length
        .saturating_mul(2)
        .saturating_sub(segment_step)
        .max(MIN_REWARD) as f64
}

#[anysearch_reward]
pub fn segment_countdown_goal(context: &RewardContext) -> RewardResult {
    let reward = reward_value(
        context.goal_reached,
        context.segment_length,
        context.segment_step,
    );
    RewardResult::replace(reward).with_component("segment_countdown_goal", reward)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_an_independent_budget_for_each_segment() {
        assert_eq!(reward_value(false, 4, 1), 0.0);
        assert_eq!(reward_value(true, 4, 4), 4.0);
        assert_eq!(reward_value(true, 4, 7), 1.0);
        assert_eq!(reward_value(true, 2, 2), 2.0);
    }
}