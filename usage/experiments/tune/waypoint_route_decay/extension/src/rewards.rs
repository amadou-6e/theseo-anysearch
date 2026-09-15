use anysearch_extension::{anysearch_reward, RewardContext, RewardResult};

const MIN_REWARD: f64 = 1.0;

fn reward_value(goal_reached: bool, max_reward: f64, step: u64) -> f64 {
    if goal_reached {
        (max_reward - step as f64).max(MIN_REWARD)
    } else {
        0.0
    }
}

#[anysearch_reward]
pub fn time_decayed_goal(context: &RewardContext) -> RewardResult {
    let reward = reward_value(
        context.goal_reached,
        context.standard_reward,
        context.step,
    );
    RewardResult::replace(reward).with_component("time_decayed_goal", reward)
}

#[cfg(test)]
mod tests {
    use super::reward_value;

    #[test]
    fn only_goal_steps_pay_and_reward_has_a_floor() {
        assert_eq!(reward_value(false, 150.0, 1), 0.0);
        assert_eq!(reward_value(true, 150.0, 1), 149.0);
        assert_eq!(reward_value(true, 150.0, 20), 130.0);
        assert_eq!(reward_value(true, 150.0, 149), 1.0);
        assert_eq!(reward_value(true, 150.0, 200), 1.0);
    }
}