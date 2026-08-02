use crate::voxel::rewards::{DistanceRewardMode, RewardConfig};

pub const NAME: &str = "distance_progress";

pub fn compute(
    config: &RewardConfig,
    has_goal: bool,
    previous_distance: f32,
    current_distance: f32,
    grid_size: u16,
) -> f32 {
    if !has_goal {
        0.0
    } else if config.distance_reward_mode == DistanceRewardMode::Progress {
        config.distance_shaping * (previous_distance - current_distance)
    } else {
        config.zone_reward(current_distance, grid_size)
    }
}
