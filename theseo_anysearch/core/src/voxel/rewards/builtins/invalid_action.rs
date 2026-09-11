use crate::voxel::rewards::RewardConfig;

pub const NAME: &str = "invalid_action";

pub fn compute(config: &RewardConfig, invalid_action: bool) -> f32 {
    if invalid_action {
        config.invalid_action_cost
    } else {
        0.0
    }
}
