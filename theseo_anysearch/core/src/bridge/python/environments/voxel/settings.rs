use crate::voxel::{DistanceRewardMode, RewardConfig, ZoneRewardCurve};

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
    let config = RewardConfig {
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
    };
    config.validate_finite()?;
    Ok(config)
}

pub(crate) fn default_action_outcomes(trail_mode: bool) -> String {
    if trail_mode {
        r#"[{"name":"cursor_movement"},{"name":"trail_placement"}]"#.to_owned()
    } else {
        r#"[{"name":"cursor_movement"}]"#.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_args() -> (f32, f32, f32, f32, f32, f32, f32, &'static str, f32, f32, &'static str)
    {
        (
            -0.01, 1.0, 0.1, 0.0, 0.0, 0.0, 0.0, "progress", -1.0, -0.01, "linear",
        )
    }

    #[test]
    fn finite_values_construct_successfully() {
        let (
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
        ) = valid_args();
        let result = reward_config(
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
        );
        assert!(result.is_ok());
    }

    #[test]
    fn nan_step_cost_is_rejected() {
        let (
            _step_cost,
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
        ) = valid_args();
        let result = reward_config(
            f32::NAN,
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
        );
        let err = result.unwrap_err();
        assert!(err.contains("step_cost"));
    }

    #[test]
    fn infinite_distance_shaping_is_rejected() {
        let (
            step_cost,
            goal_reward,
            _distance_shaping,
            collision_cost,
            invalid_action_cost,
            construction_residual_weight,
            construction_overshoot_weight,
            distance_reward_mode,
            zone_reward_min,
            zone_reward_max,
            zone_reward_curve,
        ) = valid_args();
        let result = reward_config(
            step_cost,
            goal_reward,
            f32::INFINITY,
            collision_cost,
            invalid_action_cost,
            construction_residual_weight,
            construction_overshoot_weight,
            distance_reward_mode,
            zone_reward_min,
            zone_reward_max,
            zone_reward_curve,
        );
        let err = result.unwrap_err();
        assert!(err.contains("distance_shaping"));
    }

    #[test]
    fn nan_zone_reward_max_is_rejected() {
        let (
            step_cost,
            goal_reward,
            distance_shaping,
            collision_cost,
            invalid_action_cost,
            construction_residual_weight,
            construction_overshoot_weight,
            distance_reward_mode,
            zone_reward_min,
            _zone_reward_max,
            zone_reward_curve,
        ) = valid_args();
        let result = reward_config(
            step_cost,
            goal_reward,
            distance_shaping,
            collision_cost,
            invalid_action_cost,
            construction_residual_weight,
            construction_overshoot_weight,
            distance_reward_mode,
            zone_reward_min,
            f32::NAN,
            zone_reward_curve,
        );
        let err = result.unwrap_err();
        assert!(err.contains("zone_reward_max"));
    }
}
