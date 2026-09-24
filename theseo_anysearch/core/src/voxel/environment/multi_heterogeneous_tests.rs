use super::multi::MultiAgentVoxelEnv;
use crate::voxel::rewards::{DistanceRewardMode, RewardConfig, ZoneRewardCurve};

fn rewards() -> RewardConfig {
    RewardConfig {
        step_cost: 0.0,
        goal_reward: 0.0,
        distance_shaping: 0.0,
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

fn env(max_steps: u32, agents: &str) -> MultiAgentVoxelEnv {
    let mut env = MultiAgentVoxelEnv::new(2, max_steps, false, vec![], rewards(), 16);
    env.configure_agents(agents, None).unwrap();
    env
}

#[test]
fn configured_order_makes_earlier_outcomes_visible_to_later_agents() {
    let mut env = env(
        5,
        r#"[
          {"id":"first","policy":"first","start":[1,1,1],"action_predicates":[{"name":"valid_action"},{"name":"bounds"},{"name":"unoccupied"}],"action_outcomes":[{"name":"cursor_movement"},{"name":"trail_placement"}]},
          {"id":"second","policy":"second","start":[3,1,1],"action_predicates":[{"name":"valid_action"},{"name":"bounds"},{"name":"unoccupied"}],"action_outcomes":[{"name":"cursor_movement"}]}
        ]"#,
    );
    env.reset(42);
    let result = env.step_all(&[21, 4]);
    assert_eq!(result.cursors, vec![(2, 1, 1), (3, 1, 1)]);
}

#[test]
fn adjacent_capture_rewards_hunter_and_ends_episode() {
    let mut env = env(
        5,
        r#"[
          {"id":"hunted","policy":"hunted","start":[4,1,1],"action_predicates":[{"name":"valid_action"},{"name":"bounds"},{"name":"unoccupied"}],"action_outcomes":[{"name":"cursor_movement"}]},
          {"id":"hunter","policy":"hunter","start":[2,1,1],"action_predicates":[{"name":"valid_action"},{"name":"bounds"},{"name":"unoccupied"}],"action_outcomes":[{"name":"cursor_movement"}]}
        ]"#,
    );
    env.configure_capture_task(Some(r#"{"hunter":"hunter","hunted":"hunted","capture_distance":1,"hunter_capture_reward":2.0,"hunted_escape_reward":3.0}"#)).unwrap();
    env.reset(42);
    let result = env.step_all(&[26, 21]);
    assert!(result.done);
    assert_eq!(result.rewards, vec![0.0, 2.0]);
}

#[test]
fn timeout_rewards_hunted() {
    let mut env = env(
        1,
        r#"[
          {"id":"hunted","policy":"hunted","start":[10,1,1],"action_predicates":[{"name":"valid_action"},{"name":"bounds"},{"name":"unoccupied"}],"action_outcomes":[{"name":"cursor_movement"}]},
          {"id":"hunter","policy":"hunter","start":[1,1,1],"action_predicates":[{"name":"valid_action"},{"name":"bounds"},{"name":"unoccupied"}],"action_outcomes":[{"name":"cursor_movement"}]}
        ]"#,
    );
    env.configure_capture_task(Some(r#"{"hunter":"hunter","hunted":"hunted","capture_distance":1,"hunter_capture_reward":2.0,"hunted_escape_reward":3.0}"#)).unwrap();
    env.reset(42);
    let result = env.step_all(&[26, 26]);
    assert!(result.done);
    assert_eq!(result.rewards, vec![3.0, 0.0]);
}
