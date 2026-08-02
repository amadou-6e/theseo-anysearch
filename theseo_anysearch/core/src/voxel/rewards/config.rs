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
