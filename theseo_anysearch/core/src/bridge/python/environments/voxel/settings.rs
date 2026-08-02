use crate::environments::voxel_env::{DistanceRewardMode, RewardConfig, ZoneRewardCurve};

pub(crate) fn reward_config(
    step_cost: f32,
    goal_reward: f32,
    distance_shaping: f32,
    collision_cost: f32,
    invalid_action_cost: f32,
    construction_residual_weight: f32,
    construction_overshoot_weight: f32,
    distance_reward_mode: &str,
    zone_reward_min: f32,
    zone_reward_max: f32,
    zone_reward_curve: &str,
) -> Result<RewardConfig, String> {
    let distance_reward_mode = DistanceRewardMode::from_name(distance_reward_mode)
        .ok_or_else(|| "distance_reward_mode must be 'progress' or 'zone'".to_owned())?;
    let zone_reward_curve = ZoneRewardCurve::from_name(zone_reward_curve)
        .ok_or_else(|| "zone_reward_curve must be 'linear' or 'exponential'".to_owned())?;
    Ok(RewardConfig {
        step_cost,
        goal_reward,
        distance_shaping,
        collision_cost,
        invalid_action_cost,
        construction_residual_weight,
        construction_overshoot_weight,
        distance_reward_mode,
        zone_reward_min,
        zone_reward_max,
        zone_reward_curve,
    })
}

pub(crate) fn default_action_outcomes(trail_mode: bool) -> String {
    if trail_mode {
        r#"[{"name":"cursor_movement"},{"name":"trail_placement"}]"#.to_owned()
    } else {
        r#"[{"name":"cursor_movement"}]"#.to_owned()
    }
}
