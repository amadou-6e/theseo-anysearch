use crate::voxel::rewards::RewardConfig;

pub const RESIDUAL_NAME: &str = "construction_residual";
pub const OVERSHOOT_NAME: &str = "construction_overshoot";

pub fn residual(config: &RewardConfig, count: usize) -> f32 {
    -(count as f32) * config.construction_residual_weight
}

pub fn overshoot(config: &RewardConfig, count: usize) -> f32 {
    -(count as f32) * config.construction_overshoot_weight
}
