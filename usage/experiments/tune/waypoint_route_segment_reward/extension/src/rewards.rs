use anysearch_extension::{anysearch_reward, RewardContext, RewardResult};

fn reward_value(
    goal_reached: bool,
    segment_length: u64,
    segment_step: u64,
    budget_multiplier: f64,
    minimum_reward: f64,
) -> f64 {
    if !goal_reached {
        return 0.0;
    }
    (budget_multiplier * segment_length as f64 - segment_step as f64).max(minimum_reward)
}

#[anysearch_reward]
pub fn segment_countdown_goal(context: &RewardContext) -> RewardResult {
    let budget_multiplier = context.parameters["budget_multiplier"]
        .as_f64()
        .expect("budget_multiplier must be numeric");
    let minimum_reward = context.parameters["minimum_reward"]
        .as_f64()
        .expect("minimum_reward must be numeric");
    let reward = reward_value(
        context.goal_reached,
        context.segment_length,
        context.segment_step,
        budget_multiplier,
        minimum_reward,
    );
    RewardResult::replace(reward).with_component("segment_countdown_goal", reward)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_an_independent_configured_budget_for_each_segment() {
        assert_eq!(reward_value(false, 4, 1, 2.0, 1.0), 0.0);
        assert_eq!(reward_value(true, 4, 4, 2.0, 1.0), 4.0);
        assert_eq!(reward_value(true, 4, 7, 2.0, 1.0), 1.0);
        assert_eq!(reward_value(true, 2, 2, 3.0, 0.5), 4.0);
    }
}
