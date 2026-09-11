use crate::voxel::rewards::{DistanceRewardMode, RewardConfig};

pub const NAME: &str = "step_cost";

pub fn compute(config: &RewardConfig) -> f32 {
    if config.distance_reward_mode == DistanceRewardMode::Progress {
        config.step_cost
    } else {
        0.0
    }
}
