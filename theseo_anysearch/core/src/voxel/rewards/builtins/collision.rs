use crate::voxel::rewards::RewardConfig;

pub const NAME: &str = "collision";

pub fn compute(config: &RewardConfig, collision: bool) -> f32 {
    if collision {
        config.collision_cost
    } else {
        0.0
    }
}
