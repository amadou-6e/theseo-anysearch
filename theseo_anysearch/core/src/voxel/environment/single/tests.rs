use super::*;
use crate::voxel::{DistanceRewardMode, ZoneRewardCurve};

fn make_env(max_steps: u32) -> VoxelEnv {
    VoxelEnv::new(WorldState::new(), max_steps)
}

#[test]
fn non_cubic_extent_masks_each_axis_independently() {
    let mut env = VoxelEnv::new(WorldState::new(), 10)
        .with_extent([5, 3, 2])
        .with_geometry(Vec::new());
    env.set_waypoints((5, 3, 2), (1, 1, 1));
    env.reset(1);

    let mask = env.action_mask();
    for (index, (dx, dy, dz)) in OFFSETS_26.iter().copied().enumerate() {
        let expected = dx <= 0 && dy <= 0 && dz <= 0;
        assert_eq!(mask[index] == 1, expected, "offset {:?}", (dx, dy, dz));
    }
}

fn make_env_with_geometry() -> VoxelEnv {
    VoxelEnv::new(WorldState::new(), 20)
        .with_geometry(vec![(5, 5, 5), (6, 5, 5)])
        .with_trail_mode(true)
}

fn make_backend_env(world: WorldState) -> VoxelEnv {
    let mut env = VoxelEnv::new(world, 20)
        .with_grid_size(8)
        .with_geometry(vec![(4, 4, 4), (5, 4, 4)])
        .with_trail_mode(true);
    env.set_waypoints((3, 4, 4), (6, 4, 4));
    env
}

#[test]
fn environments_sharing_a_base_keep_episode_mutations_isolated() {
    let mut base = WorldState::new();
    base.replace_base_blocks([((5, 5, 5), Block::default())]);
    let handle = base.world_handle();
    let mut first = VoxelEnv::new(WorldState::from_handle(handle.clone()), 10);
    let second = VoxelEnv::new(WorldState::from_handle(handle), 10);

    first.world.set_block((6, 5, 5), Block::default()).unwrap();
    first.world.remove_block((5, 5, 5)).unwrap();

    assert!(second.world.is_filled((5, 5, 5)));
    assert!(!second.world.is_filled((6, 5, 5)));
}

#[test]
fn environment_results_and_masks_match_across_world_backends() {
    let mut oracle = make_backend_env(WorldState::new_hashmap());
    let mut chunked = make_backend_env(WorldState::new_chunked(4));
    let oracle_observation = oracle.reset(17);
    let chunked_observation = chunked.reset(17);
    assert_eq!(oracle_observation.filled, chunked_observation.filled);
    assert_eq!(
        oracle_observation.steps_remaining,
        chunked_observation.steps_remaining
    );
    assert_eq!(
        oracle_observation.goal_distance,
        chunked_observation.goal_distance
    );
    assert_eq!(oracle.action_mask(), chunked.action_mask());

    for action in [
        VoxelAction::Place((2, 2, 2)),
        VoxelAction::Noop,
        VoxelAction::Remove((2, 2, 2)),
    ] {
        let oracle_result = oracle.step(action.clone());
        let chunked_result = chunked.step(action);
        assert_eq!(oracle_result.reward, chunked_result.reward);
        assert_eq!(oracle_result.done, chunked_result.done);
        assert_eq!(oracle.last_collision(), chunked.last_collision());
        assert_eq!(oracle.last_goal_reached(), chunked.last_goal_reached());
        assert_eq!(oracle.last_terminated(), chunked.last_terminated());
        assert_eq!(oracle.last_truncated(), chunked.last_truncated());
        assert_eq!(
            oracle.last_termination_reason(),
            chunked.last_termination_reason()
        );
        assert_eq!(
            oracle.last_reward_breakdown(),
            chunked.last_reward_breakdown()
        );
        assert_eq!(
            oracle.world().iter_filled().collect::<Vec<_>>(),
            chunked.world().iter_filled().collect::<Vec<_>>()
        );
    }
}

fn zone_reward_config() -> RewardConfig {
    RewardConfig {
        step_cost: -0.05,
        goal_reward: 10.0,
        distance_shaping: 0.02,
        collision_cost: -0.5,
        distance_reward_mode: DistanceRewardMode::Zone,
        zone_reward_min: -1.0,
        zone_reward_max: -0.01,
        zone_reward_curve: ZoneRewardCurve::Linear,
        ..Default::default()
    }
}

#[test]
fn reset_clears_world_no_geometry() {
    let mut env = make_env(10);
    env.step(VoxelAction::Place((1, 1, 1)));
    let obs = env.reset(42);
    assert_eq!(obs.filled, 0);
    assert_eq!(obs.steps_remaining, 10);
    assert!(env.world().is_filled((1, 1, 1)));
    assert!(!env.world().is_blocking((1, 1, 1)));
}

#[test]
fn reset_restores_geometry() {
    let mut env = VoxelEnv::new(WorldState::new(), 10).with_geometry(vec![(5, 5, 5), (6, 6, 6)]);
    env.step(VoxelAction::Place((1, 1, 1)));
    let obs = env.reset(42);
    // geometry_len=2, agent_filled=0
    assert_eq!(obs.filled, 0);
    assert!(env.world().is_filled((5, 5, 5)));
    assert!(!env.world().is_blocking(env.cursor()));
}

#[test]
fn removing_base_geometry_uses_a_tombstone_until_reset() {
    let mut env = VoxelEnv::new(WorldState::new(), 10).with_geometry(vec![(5, 5, 5)]);
    env.step(VoxelAction::Remove((5, 5, 5)));
    assert!(!env.world().is_filled((5, 5, 5)));

    env.reset(42);

    assert!(env.world().is_filled((5, 5, 5)));
}

#[test]
fn construction_targets_only_measure_resolved_agent_placements() {
    let mut env = VoxelEnv::new(WorldState::new(), 10)
        .with_geometry(vec![(5, 5, 5)])
        .with_construction_target(vec![(2, 2, 2)]);
    env.step(VoxelAction::Place((2, 2, 2)));
    assert_eq!(env.last_construction_residual(), 0);
    assert_eq!(env.last_construction_overshoot(), 0);

    env.reset(42);
    env.step(VoxelAction::Noop);
    assert_eq!(env.last_construction_residual(), 1);
    assert_eq!(env.last_construction_overshoot(), 0);
}

#[test]
fn noop_step_no_change() {
    let mut env = make_env(10);
    env.reset(42);
    let before = env.agent_filled();
    env.step(VoxelAction::Noop);
    assert_eq!(env.agent_filled(), before);
}

#[test]
fn place_step_fills() {
    let mut env = make_env(10);
    env.reset(42);
    let sr = env.step(VoxelAction::Place((50, 50, 50)));
    assert!(env.world().is_filled((50, 50, 50)));
    // reward = step_cost + shaping; just check done=false
    assert!(!sr.done);
}

#[test]
fn remove_step_clears() {
    let mut env = make_env(10);
    env.reset(42);
    env.step(VoxelAction::Place((1, 1, 1)));
    env.step(VoxelAction::Remove((1, 1, 1)));
    assert!(!env.world().is_filled((1, 1, 1)));
}

#[test]
fn done_at_max_steps() {
    let mut env = make_env(3);
    env.reset(42);
    env.step(VoxelAction::Noop);
    env.step(VoxelAction::Noop);
    let sr = env.step(VoxelAction::Noop);
    assert!(sr.done);
}

#[test]
fn geometry_not_counted_in_agent_filled() {
    let mut env = VoxelEnv::new(WorldState::new(), 10).with_geometry(vec![(10, 10, 10)]);
    env.reset(42);
    assert_eq!(env.agent_filled(), 0);
    env.step(VoxelAction::Place((1, 1, 1)));
    assert_eq!(env.agent_filled(), 1);
}

#[test]
fn deterministic_reset() {
    let mut env1 = make_env(10);
    let mut env2 = make_env(10);
    let obs1 = env1.reset(42);
    let obs2 = env2.reset(42);
    assert_eq!(obs1.filled, obs2.filled);
    assert_eq!(obs1.steps_remaining, obs2.steps_remaining);
}

#[test]
fn surface_cells_computed_from_geometry() {
    let env = VoxelEnv::new(WorldState::new(), 10).with_geometry(vec![(5, 5, 5)]);
    // A single geometry cell at (5,5,5) should have 6 surface neighbours.
    assert_eq!(env.surface_cells().len(), 6);
    // All neighbours should be adjacent.
    for &(x, y, z) in env.surface_cells() {
        let dist = manhattan((5, 5, 5), (x, y, z));
        assert_eq!(dist, 1, "surface cell should be distance 1 from geometry");
    }
}

#[test]
fn surface_cells_do_not_include_geometry() {
    let geo = vec![(5, 5, 5), (6, 5, 5)];
    let env = VoxelEnv::new(WorldState::new(), 10).with_geometry(geo.clone());
    for &c in env.surface_cells() {
        assert!(
            !geo.contains(&c),
            "surface cell must not be a geometry cell"
        );
    }
}

#[test]
fn reset_places_cursor_on_surface_cell() {
    let mut env = make_env_with_geometry();
    env.reset(7);
    let (cx, cy, cz) = env.cursor();
    let gs = env.grid_size;
    // Cursor should be within grid bounds.
    assert!((1..=gs).contains(&cx));
    assert!((1..=gs).contains(&cy));
    assert!((1..=gs).contains(&cz));
}

#[test]
fn goal_reached_ends_episode() {
    let mut env = VoxelEnv::new(WorldState::new(), 100).with_geometry(vec![(5, 5, 5)]);
    env.reset(42);
    // Force goal to a known cell.
    let goal = env.active_goal().unwrap_or((4, 5, 5));
    env.set_waypoints(env.cursor(), goal);
    env.reset(0); // picks fixed waypoints
                  // Move cursor to goal manually.
    env.set_cursor(goal);
    let sr = env.step(VoxelAction::Noop);
    assert!(sr.done, "episode must end when cursor reaches goal");
    assert!(
        sr.reward >= env.reward_config.goal_reward - 1.0,
        "goal reward must be included"
    );
}

#[test]
fn zone_reward_prefers_closer_absolute_distance() {
    let mut closer_env = make_env(10).with_reward_config(zone_reward_config());
    let mut farther_env = make_env(10).with_reward_config(zone_reward_config());
    closer_env.set_waypoints((4, 4, 4), (4, 4, 6));
    farther_env.set_waypoints((4, 4, 4), (4, 4, 6));
    closer_env.reset(0);
    farther_env.reset(0);

    closer_env.set_cursor((4, 4, 5));
    farther_env.set_cursor((4, 4, 3));
    let closer = closer_env.step(VoxelAction::Noop).reward;
    let farther = farther_env.step(VoxelAction::Noop).reward;

    assert!(farther < closer);
    assert!(closer < 0.0);
}

#[test]
fn zone_reward_goal_step_adds_terminal_bonus() {
    let mut env = make_env(10).with_reward_config(zone_reward_config());
    env.set_waypoints((4, 4, 4), (4, 4, 6));
    env.reset(0);

    env.set_cursor((4, 4, 6));
    let sr = env.step(VoxelAction::Noop);

    assert!(sr.done);
    assert!((sr.reward - 9.99).abs() < 0.0001);
}

#[test]
fn fixed_waypoints_override_random() {
    let mut env = VoxelEnv::new(WorldState::new(), 10).with_geometry(vec![(10, 10, 10)]);
    env.set_waypoints((2, 2, 2), (3, 3, 3));
    env.reset(0);
    assert_eq!(env.cursor(), (2, 2, 2));
    assert_eq!(env.active_goal(), Some((3, 3, 3)));
}

#[test]
fn manhattan_to_goal_in_observation() {
    let mut env = VoxelEnv::new(WorldState::new(), 10).with_geometry(vec![(10, 10, 10)]);
    env.set_waypoints((1, 1, 1), (4, 1, 1));
    let obs = env.reset(0);
    assert_eq!(obs.goal_distance, Some(3));
}

#[test]
fn consecutive_collisions_terminate_in_rust() {
    let mut env = make_env(20).with_max_consecutive_collisions(Some(2));
    env.set_waypoints((1, 1, 1), (3, 1, 1));
    env.reset(0);
    env.prepare_navigation_step(0, env.cursor(), false);
    let first = env.step(VoxelAction::Collision);
    assert!(!first.done);
    env.prepare_navigation_step(0, env.cursor(), false);
    let second = env.step(VoxelAction::Collision);
    assert!(second.done);
    assert!(env.last_terminated());
    assert!(!env.last_truncated());
    assert_eq!(env.last_termination_reason(), "consecutive_collisions");
}

#[test]
fn rust_breakdown_is_authoritative() {
    let mut env = make_env(20);
    env.set_waypoints((1, 1, 1), (3, 1, 1));
    env.reset(0);
    env.prepare_navigation_step(0, env.cursor(), false);
    let result = env.step(VoxelAction::Collision);
    let component_sum: f32 = env.last_reward_breakdown().values().sum();
    assert!((result.reward - component_sum).abs() < 1e-6);
    assert_eq!(env.last_reward_breakdown().get("collision"), Some(&0.0));
}

fn configure_pipeline(env: &mut VoxelEnv, outcomes: &str, history: usize) {
    env.configure_action_pipeline(
        r#"[{"name":"valid_action"},{"name":"bounds"},{"name":"unoccupied"}]"#,
        outcomes,
        history,
        None,
    )
    .unwrap();
}

#[test]
fn cursor_navigation_moves_without_filling() {
    let mut env = make_env(10);
    env.set_waypoints((2, 2, 2), (6, 2, 2));
    env.reset(1);
    configure_pipeline(&mut env, r#"[{"name":"cursor_movement"}]"#, 4);
    env.prepare_navigation_step(21, (2, 2, 2), false);
    let action = env.execute_navigation_action(21);
    env.step(action);
    assert_eq!(env.cursor(), (3, 2, 2));
    assert_eq!(env.agent_filled(), 0);
}

#[test]
fn trail_navigation_moves_and_fills_destination() {
    let mut env = make_env(10);
    env.set_waypoints((2, 2, 2), (6, 2, 2));
    env.reset(1);
    configure_pipeline(
        &mut env,
        r#"[{"name":"cursor_movement"},{"name":"trail_placement"}]"#,
        4,
    );
    env.prepare_navigation_step(21, (2, 2, 2), false);
    let action = env.execute_navigation_action(21);
    env.step(action);
    assert_eq!(env.cursor(), (3, 2, 2));
    assert_eq!(env.agent_filled(), 1);
}

#[test]
fn action_mask_uses_the_same_bounds_and_occupancy_predicates() {
    let mut env = make_env(10);
    env.set_waypoints((1, 1, 1), (6, 2, 2));
    env.reset(1);
    configure_pipeline(&mut env, r#"[{"name":"cursor_movement"}]"#, 4);
    let mask = env.action_mask();
    assert_eq!(mask.len(), 27);
    assert_eq!(mask[0], 0);
    assert_eq!(mask[21], 1);
    assert_eq!(mask[26], 1);
}

#[test]
fn action_mask_error_surfaces_immediately_not_on_later_step() {
    // Regression test for the bug where a predicate error raised during
    // action_mask() was stashed in `last_reward_error` and only drained by
    // the next call to `take_reward_error()` — which, at the PyO3 binding
    // layer, happens inside `step()`. That misattributed a masking failure
    // to a later, unrelated step (or silently dropped it if step() was
    // never called again before reset()).
    //
    // The fix is for the caller of `action_mask()` (the PyO3 binding) to
    // drain `take_reward_error()` immediately after computing the mask.
    // This test exercises that exact call sequence at the Rust level.
    let mut env = make_env(10);
    env.set_waypoints((1, 1, 1), (6, 2, 2));
    env.reset(1);
    configure_pipeline(&mut env, r#"[{"name":"cursor_movement"}]"#, 4);
    env.action_predicates.push(ConfiguredPredicate::Failing(
        "predicate exploded".to_owned(),
    ));

    // action_mask() swallows the predicate error into `last_reward_error`
    // and marks the corresponding action(s) infeasible.
    let mask = env.action_mask();
    assert!(mask.iter().all(|&feasible| feasible == 0));

    // The error must be observable right at the action_mask() call site —
    // this is what the fixed PyO3 binding now checks for.
    let error = env.take_reward_error();
    assert_eq!(error.as_deref(), Some("predicate exploded"));

    // Having drained the error at the mask call site, a subsequent
    // unrelated, successful step must NOT resurface it.
    env.action_predicates.pop();
    env.prepare_navigation_step(21, env.cursor(), false);
    let action = env.execute_navigation_action(21);
    let _ = env.step(action);
    assert_eq!(env.take_reward_error(), None);
}

#[test]
fn action_history_is_bounded() {
    let mut env = make_env(10);
    env.set_waypoints((2, 2, 2), (6, 2, 2));
    env.reset(1);
    configure_pipeline(&mut env, r#"[{"name":"cursor_movement"}]"#, 1);
    for action_index in [21, 21] {
        let previous = env.cursor();
        env.prepare_navigation_step(action_index, previous, false);
        let action = env.execute_navigation_action(action_index);
        env.step(action);
    }
    assert_eq!(env.action_history.len(), 1);
    assert_eq!(env.action_history[0].previous_cursor, [3, 2, 2]);
    assert_eq!(env.action_history[0].cursor, [4, 2, 2]);
}

#[test]
fn configure_action_pipeline_fails_when_native_predicate_load_fails() {
    let mut env = make_env(10);
    let bad_library = Path::new("this/path/does/not/exist.so");
    let result = env.configure_action_pipeline(
        r#"[{"name":"bounds"}]"#,
        r#"[{"name":"cursor_movement"}]"#,
        4,
        Some(bad_library),
    );
    assert!(
        result.is_err(),
        "construction must fail instead of silently falling back to the builtin predicate"
    );
    let message = result.unwrap_err();
    assert!(
        message.contains("bounds"),
        "error should mention the predicate name: {message}"
    );
}

#[test]
fn configure_action_pipeline_fails_when_native_outcome_load_fails() {
    let mut env = make_env(10);
    let bad_library = Path::new("this/path/does/not/exist.so");
    let result = env.configure_action_pipeline(
        "[]",
        r#"[{"name":"cursor_movement"}]"#,
        4,
        Some(bad_library),
    );
    assert!(
        result.is_err(),
        "construction must fail instead of silently falling back to the builtin outcome"
    );
    let message = result.unwrap_err();
    assert!(
        message.contains("cursor_movement"),
        "error should mention the outcome name: {message}"
    );
}
