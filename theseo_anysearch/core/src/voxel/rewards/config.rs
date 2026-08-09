//! Built-in voxel reward configuration.\n\n
#[derive(Clone, Debug)]
pub struct RewardConfig {
    /// Per-step penalty (typically negative, e.g. -0.05).
    pub step_cost: f32,
    /// Bonus awarded when cursor reaches the goal position.
    pub goal_reward: f32,
    /// Coefficient for potential-based L2 distance shaping toward goal.
    /// Each step: shaping = distance_shaping * (prev_l2 - new_l2).
    /// Set to 0.0 to disable.
    pub distance_shaping: f32,
    /// Extra penalty subtracted when a movement is blocked (boundary hit or
    /// occupied cell).  Positive value = bigger penalty.  Default 0.0.
    pub collision_cost: f32,
    pub invalid_action_cost: f32,
    pub construction_residual_weight: f32,
    pub construction_overshoot_weight: f32,
    /// Distance reward strategy. "progress" preserves potential shaping.
    /// "zone" gives a negative per-step reward based on absolute goal distance.
    pub distance_reward_mode: DistanceRewardMode,
    /// Most negative zone reward, applied at maximum possible goal distance.
    pub zone_reward_min: f32,
    /// Least negative zone reward, applied at the goal distance.
    pub zone_reward_max: f32,
    /// Interpolation curve used by zone reward mode.
    pub zone_reward_curve: ZoneRewardCurve,
}

#[derive(Clone, Debug, PartialEq)]
pub enum DistanceRewardMode {
    Progress,
    Zone,
}

impl DistanceRewardMode {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "progress" => Some(Self::Progress),
            "zone" => Some(Self::Zone),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ZoneRewardCurve {
    Linear,
    Exponential,
}

impl ZoneRewardCurve {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "linear" => Some(Self::Linear),
            "exponential" => Some(Self::Exponential),
            _ => None,
        }
    }
}

impl RewardConfig {
    /// Validates that every numeric reward-shaping field is finite (not NaN or
    /// +/-infinity). A non-finite value would silently poison every reward
    /// computed for the run, so this must be checked at construction time
    /// (e.g. right after a `RewardConfig` is built from raw Python/YAML input).
    ///
    /// Returns `Err` naming the first offending field encountered.
    pub fn validate_finite(&self) -> Result<(), String> {
        let fields: [(&str, f32); 9] = [
            ("step_cost", self.step_cost),
            ("goal_reward", self.goal_reward),
            ("distance_shaping", self.distance_shaping),
            ("collision_cost", self.collision_cost),
            ("invalid_action_cost", self.invalid_action_cost),
            (
                "construction_residual_weight",
                self.construction_residual_weight,
            ),
            (
                "construction_overshoot_weight",
                self.construction_overshoot_weight,
            ),
            ("zone_reward_min", self.zone_reward_min),
            ("zone_reward_max", self.zone_reward_max),
        ];
        for (name, value) in fields {
            if !value.is_finite() {
                return Err(format!(
                    "{name} must be a finite number, got {value} (NaN/Inf are not allowed)"
                ));
            }
        }
        Ok(())
    }

    pub fn base_step_reward(&self, previous_l2: f32, current_l2: f32, grid_size: u16) -> f32 {
        match self.distance_reward_mode {
            DistanceRewardMode::Progress => {
                self.step_cost + self.distance_shaping * (previous_l2 - current_l2)
            }
            DistanceRewardMode::Zone => self.zone_reward(current_l2, grid_size),
        }
    }

    pub fn zone_reward(&self, distance_l2: f32, grid_size: u16) -> f32 {
        let max_l2 = (3.0f32).sqrt() * f32::from(grid_size.saturating_sub(1).max(1));
        let normalized = (distance_l2 / max_l2).clamp(0.0, 1.0);
        let curved = match self.zone_reward_curve {
            ZoneRewardCurve::Linear => normalized,
            ZoneRewardCurve::Exponential => {
                let steepness = 3.0f32;
                (steepness * normalized).exp_m1() / steepness.exp_m1()
            }
        };
        self.zone_reward_max + (self.zone_reward_min - self.zone_reward_max) * curved
    }
}

impl Default for RewardConfig {
    fn default() -> Self {
        Self {
            step_cost: -0.01,
            goal_reward: 1.0,
            distance_shaping: 0.1,
            collision_cost: 0.0,
            invalid_action_cost: 0.0,
            construction_residual_weight: 0.0,
            construction_overshoot_weight: 0.0,
            distance_reward_mode: DistanceRewardMode::Progress,
            zone_reward_min: -1.0,
            zone_reward_max: -0.01,
            zone_reward_curve: ZoneRewardCurve::Linear,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates_as_finite() {
        assert!(RewardConfig::default().validate_finite().is_ok());
    }

    #[test]
    fn nan_step_cost_is_rejected() {
        let config = RewardConfig {
            step_cost: f32::NAN,
            ..Default::default()
        };
        let err = config.validate_finite().unwrap_err();
        assert!(err.contains("step_cost"));
    }

    #[test]
    fn infinite_goal_reward_is_rejected() {
        let config = RewardConfig {
            goal_reward: f32::INFINITY,
            ..Default::default()
        };
        let err = config.validate_finite().unwrap_err();
        assert!(err.contains("goal_reward"));
    }

    #[test]
    fn negative_infinite_distance_shaping_is_rejected() {
        let config = RewardConfig {
            distance_shaping: f32::NEG_INFINITY,
            ..Default::default()
        };
        let err = config.validate_finite().unwrap_err();
        assert!(err.contains("distance_shaping"));
    }

    #[test]
    fn nan_collision_cost_is_rejected() {
        let config = RewardConfig {
            collision_cost: f32::NAN,
            ..Default::default()
        };
        let err = config.validate_finite().unwrap_err();
        assert!(err.contains("collision_cost"));
    }

    #[test]
    fn nan_zone_reward_min_is_rejected() {
        let config = RewardConfig {
            zone_reward_min: f32::NAN,
            ..Default::default()
        };
        let err = config.validate_finite().unwrap_err();
        assert!(err.contains("zone_reward_min"));
    }

    #[test]
    fn infinite_zone_reward_max_is_rejected() {
        let config = RewardConfig {
            zone_reward_max: f32::INFINITY,
            ..Default::default()
        };
        let err = config.validate_finite().unwrap_err();
        assert!(err.contains("zone_reward_max"));
    }

    #[test]
    fn every_numeric_field_is_checked_exhaustively() {
        // Each (field-setter) pair independently poisons the config with a
        // non-finite value; every one of them must be caught.
        let poisoners: Vec<(&str, RewardConfig)> = vec![
            (
                "step_cost",
                RewardConfig {
                    step_cost: f32::NAN,
                    ..Default::default()
                },
            ),
            (
                "goal_reward",
                RewardConfig {
                    goal_reward: f32::NAN,
                    ..Default::default()
                },
            ),
            (
                "distance_shaping",
                RewardConfig {
                    distance_shaping: f32::INFINITY,
                    ..Default::default()
                },
            ),
            (
                "collision_cost",
                RewardConfig {
                    collision_cost: f32::NEG_INFINITY,
                    ..Default::default()
                },
            ),
            (
                "invalid_action_cost",
                RewardConfig {
                    invalid_action_cost: f32::NAN,
                    ..Default::default()
                },
            ),
            (
                "construction_residual_weight",
                RewardConfig {
                    construction_residual_weight: f32::NAN,
                    ..Default::default()
                },
            ),
            (
                "construction_overshoot_weight",
                RewardConfig {
                    construction_overshoot_weight: f32::NAN,
                    ..Default::default()
                },
            ),
            (
                "zone_reward_min",
                RewardConfig {
                    zone_reward_min: f32::NAN,
                    ..Default::default()
                },
            ),
            (
                "zone_reward_max",
                RewardConfig {
                    zone_reward_max: f32::NAN,
                    ..Default::default()
                },
            ),
        ];
        for (field, config) in poisoners {
            assert!(
                config.validate_finite().is_err(),
                "expected {field} to be rejected"
            );
        }
    }
}
