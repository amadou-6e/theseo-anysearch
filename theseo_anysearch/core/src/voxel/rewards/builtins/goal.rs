use crate::voxel::rewards::RewardConfig;

pub const NAME: &str = "success";

pub fn compute(config: &RewardConfig, goal_reached: bool) -> f32 {
    if goal_reached {
        config.goal_reward
    } else {
        0.0
    }
}
