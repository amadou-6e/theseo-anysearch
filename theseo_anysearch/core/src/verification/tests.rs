use crate::{
    environments::Environment,
    verification::{
        benchmark::{self, BenchmarkConfig},
        fault::{FaultInjectedWorld, FaultOperation, FaultRule},
        parity::{
            apply_mutation, capture_abi_read, capture_read, compare_exact, Mutation, ReadProbe,
        },
        PendingFaultCase,
    },
    voxel::{
        rewards::RewardConfig,
        world::{
            Block, BlockUpdate, BoundedRegion, ChunkedWorld, GridRay, HashMapWorld, StorageCoord,
            WorldAccessError, WorldExtent, WorldMutation, WorldRead, WorldResidency, WorldState,
        },
        MultiAgentVoxelEnv, VoxelAction, VoxelEnv,
    },
};

fn block(kind: u8) -> Block {
    Block {
        kind,
        active: kind % 2 == 1,
        reward_weight: f32::from(kind) / 10.0,
    }
}

fn exact_probe(extent: WorldExtent) -> ReadProbe {
    ReadProbe {
        points: vec![
            StorageCoord { x: 0, y: 0, z: 0 },
            StorageCoord {
                x: extent.x - 1,
                y: extent.y - 1,
                z: extent.z - 1,
            },
            StorageCoord {
                x: 16.min(extent.x - 1),
                y: 16.min(extent.y - 1),
                z: 16.min(extent.z - 1),
            },
        ],
        regions: vec![
            BoundedRegion::new(
                StorageCoord { x: 0, y: 0, z: 0 },
                StorageCoord {
                    x: 17.min(extent.x),
                    y: 17.min(extent.y),
                    z: 17.min(extent.z),
                },
                extent,
            )
            .unwrap(),
            BoundedRegion::new(
                StorageCoord {
                    x: extent.x.saturating_sub(9),
                    y: extent.y.saturating_sub(9),
                    z: extent.z.saturating_sub(9),
                },
                StorageCoord {
                    x: extent.x,
                    y: extent.y,
                    z: extent.z,
                },
                extent,
            )
            .unwrap(),
        ],
        rays: vec![
            GridRay::new(StorageCoord { x: 0, y: 0, z: 0 }, [1, 1, 1], 64).unwrap(),
            GridRay::new(
                StorageCoord {
                    x: extent.x - 1,
                    y: extent.y - 1,
                    z: extent.z - 1,
                },
                [-1, 0, 0],
                64,
            )
            .unwrap(),
        ],
    }
}

fn exercise_backends(extent: WorldExtent, chunk_shape: WorldExtent) {
    let mut oracle = HashMapWorld::new(extent);
    let mut chunked = ChunkedWorld::new(extent, chunk_shape).unwrap();
    let coordinates = [
        StorageCoord { x: 0, y: 0, z: 0 },
        StorageCoord {
            x: 15.min(extent.x - 1),
            y: 15.min(extent.y - 1),
            z: 15.min(extent.z - 1),
        },
        StorageCoord {
            x: 16.min(extent.x - 1),
            y: 16.min(extent.y - 1),
            z: 16.min(extent.z - 1),
        },
        StorageCoord {
            x: extent.x - 1,
            y: extent.y - 1,
            z: extent.z - 1,
        },
    ];
    for (index, coordinate) in coordinates.into_iter().enumerate() {
        let mutation = Mutation::Set(coordinate, block(index as u8 + 1));
        assert_eq!(
            apply_mutation(&mut oracle, mutation.clone()),
            apply_mutation(&mut chunked, mutation)
        );
    }
    let update = Mutation::Update(
        coordinates[2],
        BlockUpdate {
            kind: Some(9),
            active: Some(true),
            reward_weight: Some(1.25),
        },
    );
    assert_eq!(
        apply_mutation(&mut oracle, update.clone()),
        apply_mutation(&mut chunked, update)
    );
    let remove = Mutation::Remove(coordinates[1]);
    assert_eq!(
        apply_mutation(&mut oracle, remove.clone()),
        apply_mutation(&mut chunked, remove)
    );
    compare_exact(&oracle, &chunked, &exact_probe(extent)).unwrap();
    assert_eq!(oracle.block_count(), chunked.block_count());
    assert_eq!(chunked.resident_chunk_count(), 3);
}

#[test]
fn hashmap_and_chunked_match_for_cubic_partial_edge_chunks() {
    exercise_backends(WorldExtent::cubic(35), WorldExtent::cubic(16));
}

#[test]
fn hashmap_and_chunked_match_for_non_cubic_extents_and_chunks() {
    exercise_backends(
        WorldExtent {
            x: 35,
            y: 23,
            z: 19,
        },
        WorldExtent { x: 16, y: 8, z: 7 },
    );
}

#[test]
fn empty_uniform_sparse_and_dense_logical_chunks_match() {
    let extent = WorldExtent::cubic(33);
    let mut oracle = HashMapWorld::new(extent);
    let mut chunked = ChunkedWorld::new(extent, WorldExtent::cubic(16)).unwrap();
    compare_exact(&oracle, &chunked, &exact_probe(extent)).unwrap();
    for z in 0..16 {
        for y in 0..16 {
            for x in 0..16 {
                if (x + y + z) % 5 == 0 || (x < 4 && y < 4 && z < 4) {
                    let coordinate = StorageCoord { x, y, z };
                    oracle.set_block_value(coordinate, block(5)).unwrap();
                    chunked.set_block_value(coordinate, block(5)).unwrap();
                }
            }
        }
    }
    compare_exact(&oracle, &chunked, &exact_probe(extent)).unwrap();
}

#[test]
fn world_state_facade_backends_match_exact_environment_outputs() {
    fn env(world: WorldState) -> VoxelEnv {
        let mut env = VoxelEnv::new(world, 3)
            .with_grid_size(8)
            .with_geometry(vec![(4, 4, 4), (5, 4, 4)])
            .with_trail_mode(true)
            .with_construction_target(vec![(2, 2, 2)]);
        env.set_waypoints((3, 4, 4), (6, 4, 4));
        env
    }
    let mut oracle = env(WorldState::new_hashmap());
    let mut candidate = env(WorldState::new_chunked(16));
    let left = oracle.reset(42);
    let right = candidate.reset(42);
    assert_eq!(left.filled, right.filled);
    assert_eq!(left.steps_remaining, right.steps_remaining);
    assert_eq!(left.goal_distance, right.goal_distance);
    assert_eq!(oracle.action_mask(), candidate.action_mask());
    for action in [
        VoxelAction::Collision,
        VoxelAction::Place((2, 2, 2)),
        VoxelAction::Noop,
    ] {
        let left = oracle.step(action.clone());
        let right = candidate.step(action);
        assert_eq!(left.reward, right.reward);
        assert_eq!(left.done, right.done);
        assert_eq!(
            oracle.last_reward_breakdown(),
            candidate.last_reward_breakdown()
        );
        assert_eq!(oracle.last_collision(), candidate.last_collision());
        assert_eq!(oracle.last_terminated(), candidate.last_terminated());
        assert_eq!(oracle.last_truncated(), candidate.last_truncated());
        assert_eq!(
            oracle.world().iter_filled().collect::<Vec<_>>(),
            candidate.world().iter_filled().collect::<Vec<_>>()
        );
    }
}

#[test]
fn overlay_resolution_and_reset_match_a_materialized_oracle() {
    let extent = WorldExtent::cubic(1000);
    let base_coordinate = StorageCoord { x: 4, y: 4, z: 4 };
    let removed_coordinate = StorageCoord { x: 5, y: 4, z: 4 };
    let added_coordinate = StorageCoord { x: 6, y: 4, z: 4 };
    let mut candidate = WorldState::new_chunked(16);
    candidate.replace_base_blocks([((4, 4, 4), block(1)), ((5, 4, 4), block(2))]);
    let shared_base = candidate.world_handle();
    let mut oracle = HashMapWorld::new(extent);
    oracle.set_block_value(base_coordinate, block(1)).unwrap();
    oracle
        .set_block_value(removed_coordinate, block(2))
        .unwrap();
    let probe = ReadProbe {
        points: vec![base_coordinate, removed_coordinate, added_coordinate],
        regions: vec![BoundedRegion::new(
            StorageCoord { x: 3, y: 3, z: 3 },
            StorageCoord { x: 8, y: 6, z: 6 },
            extent,
        )
        .unwrap()],
        rays: vec![GridRay::new(StorageCoord { x: 3, y: 4, z: 4 }, [1, 0, 0], 8).unwrap()],
    };
    compare_exact(&oracle, &candidate, &probe).unwrap();

    candidate
        .update_block_value(
            base_coordinate,
            BlockUpdate {
                kind: Some(9),
                active: Some(true),
                reward_weight: Some(2.0),
            },
        )
        .unwrap();
    oracle.set_block_value(base_coordinate, block(9)).unwrap();
    oracle
        .update_block_value(
            base_coordinate,
            BlockUpdate {
                reward_weight: Some(2.0),
                ..BlockUpdate::default()
            },
        )
        .unwrap();
    candidate.remove_block_value(removed_coordinate).unwrap();
    oracle.remove_block_value(removed_coordinate).unwrap();
    candidate
        .set_block_value(added_coordinate, block(3))
        .unwrap();
    oracle.set_block_value(added_coordinate, block(3)).unwrap();
    compare_exact(&oracle, &candidate, &probe).unwrap();
    assert!(candidate.estimated_overlay_bytes() > 0);

    candidate.clear_blocks();
    oracle.clear_blocks();
    oracle.set_block_value(base_coordinate, block(1)).unwrap();
    oracle
        .set_block_value(removed_coordinate, block(2))
        .unwrap();
    compare_exact(&oracle, &candidate, &probe).unwrap();
    compare_exact(&oracle, &shared_base, &probe).unwrap();
    assert_eq!(candidate.estimated_overlay_bytes(), 0);
}

#[test]
fn scenario_v2_callbacks_match_overlay_aware_generic_snapshot() {
    let extent = WorldExtent::cubic(1000);
    let mut world = WorldState::new_chunked(16);
    world.replace_base_blocks([((2, 2, 2), block(1)), ((3, 2, 2), block(2))]);
    world
        .remove_block_value(StorageCoord { x: 2, y: 2, z: 2 })
        .unwrap();
    world
        .set_block_value(StorageCoord { x: 4, y: 2, z: 2 }, block(7))
        .unwrap();
    let probe = ReadProbe {
        points: vec![
            StorageCoord { x: 2, y: 2, z: 2 },
            StorageCoord { x: 3, y: 2, z: 2 },
            StorageCoord { x: 4, y: 2, z: 2 },
        ],
        regions: vec![BoundedRegion::new(
            StorageCoord { x: 1, y: 1, z: 1 },
            StorageCoord { x: 6, y: 4, z: 4 },
            extent,
        )
        .unwrap()],
        rays: vec![GridRay::new(StorageCoord { x: 1, y: 2, z: 2 }, [1, 0, 0], 8).unwrap()],
    };

    assert_eq!(
        capture_abi_read(&world, &probe, 226).unwrap(),
        capture_read(&world, &probe)
    );
}

#[test]
fn multi_agent_and_heterogeneous_resets_are_deterministic() {
    fn multi(trail: bool) -> MultiAgentVoxelEnv {
        MultiAgentVoxelEnv::new(
            2,
            4,
            trail,
            vec![(4, 4, 4), (5, 4, 4)],
            RewardConfig::default(),
            8,
        )
    }
    let mut left = multi(true);
    let mut right = multi(true);
    assert_eq!(left.reset(91), right.reset(91));
    let left_step = left.step_all_legacy(&[0, 1]);
    let right_step = right.step_all_legacy(&[0, 1]);
    assert_eq!(left_step.cursors, right_step.cursors);
    assert_eq!(left_step.rewards, right_step.rewards);
    assert_eq!(left_step.done, right_step.done);

    let agents = r#"[
      {"id":"first","policy":"a","start":[1,1,1],"action_predicates":[{"name":"valid_action"},{"name":"bounds"},{"name":"unoccupied"}],"action_outcomes":[{"name":"cursor_movement"},{"name":"trail_placement"}]},
      {"id":"second","policy":"b","start":[3,1,1],"action_predicates":[{"name":"valid_action"},{"name":"bounds"},{"name":"unoccupied"}],"action_outcomes":[{"name":"cursor_movement"}]}
    ]"#;
    left.configure_agents(agents, None).unwrap();
    right.configure_agents(agents, None).unwrap();
    left.reset(7);
    right.reset(7);
    let left_step = left.step_all(&[13, 12]);
    let right_step = right.step_all(&[13, 12]);
    assert_eq!(left_step.cursors, right_step.cursors);
    assert_eq!(left_step.rewards, right_step.rewards);
    assert_eq!(left_step.done, right_step.done);
}

#[test]
fn heterogeneous_agent_trails_form_one_resolved_union_and_reset_isolates_it() {
    let mut env = MultiAgentVoxelEnv::new(2, 4, false, vec![(8, 8, 8)], RewardConfig::default(), 8);
    let agents = r#"[
      {"id":"first","policy":"a","start":[1,1,2],"action_predicates":[{"name":"valid_action"},{"name":"bounds"},{"name":"unoccupied"}],"action_outcomes":[{"name":"cursor_movement"},{"name":"trail_placement"}]},
      {"id":"second","policy":"b","start":[3,1,2],"action_predicates":[{"name":"valid_action"},{"name":"bounds"},{"name":"unoccupied"}],"action_outcomes":[{"name":"cursor_movement"},{"name":"trail_placement"}]}
    ]"#;
    env.configure_agents(agents, None).unwrap();
    env.reset(7);
    let base = env.world.world_handle();

    let result = env.step_all(&[13, 12]);

    assert_eq!(result.cursors, vec![(1, 1, 3), (3, 1, 1)]);
    assert!(env.world.is_filled((1, 1, 3)));
    assert!(env.world.is_filled((3, 1, 1)));
    assert_eq!(env.world.len(), base.block_count() as usize + 2);
    assert_eq!(base.block_count(), 1);
    env.reset(7);
    assert_eq!(env.world.len(), 1);
    assert!(env.world.is_filled((8, 8, 8)));
}

#[test]
fn invalid_inputs_and_arithmetic_overflow_are_exact() {
    let extent = WorldExtent { x: 4, y: 5, z: 6 };
    assert!(ChunkedWorld::new(extent, WorldExtent { x: 0, y: 1, z: 1 }).is_err());
    assert_eq!(
        extent.voxel_count(),
        Some(120),
        "checked arithmetic must retain exact small products"
    );
    assert_eq!(
        WorldExtent {
            x: u32::MAX,
            y: u32::MAX,
            z: u32::MAX
        }
        .voxel_count(),
        None
    );
    assert!(BoundedRegion::new(
        StorageCoord { x: 2, y: 2, z: 2 },
        StorageCoord { x: 2, y: 3, z: 3 },
        extent
    )
    .is_err());
    assert!(GridRay::new(StorageCoord { x: 0, y: 0, z: 0 }, [0, 0, 0], 1).is_err());
    let world = HashMapWorld::new(extent);
    assert_eq!(
        world.get_block_value(StorageCoord { x: 4, y: 0, z: 0 }),
        Err(WorldAccessError::OutOfBounds(StorageCoord {
            x: 4,
            y: 0,
            z: 0
        }))
    );
}

#[test]
fn injected_query_and_mutation_failures_are_observable() {
    let coordinate = StorageCoord { x: 1, y: 1, z: 1 };
    let error = WorldAccessError::NotFound(coordinate);
    let mut world = FaultInjectedWorld::new(
        HashMapWorld::new(WorldExtent::cubic(4)),
        vec![
            FaultRule {
                operation: FaultOperation::Point,
                fail_on_call: 1,
                error,
            },
            FaultRule {
                operation: FaultOperation::Set,
                fail_on_call: 1,
                error,
            },
            FaultRule {
                operation: FaultOperation::Region,
                fail_on_call: 1,
                error,
            },
            FaultRule {
                operation: FaultOperation::Ray,
                fail_on_call: 1,
                error,
            },
            FaultRule {
                operation: FaultOperation::Update,
                fail_on_call: 1,
                error,
            },
            FaultRule {
                operation: FaultOperation::Remove,
                fail_on_call: 1,
                error,
            },
        ],
    );
    assert_eq!(world.get_block_value(coordinate), Err(error));
    assert_eq!(world.get_block_value(coordinate), Ok(None));
    assert_eq!(world.set_block_value(coordinate, block(1)), Err(error));
    assert_eq!(world.set_block_value(coordinate, block(1)), Ok(None));
    let region = BoundedRegion::new(
        StorageCoord { x: 0, y: 0, z: 0 },
        StorageCoord { x: 2, y: 2, z: 2 },
        WorldExtent::cubic(4),
    )
    .unwrap();
    let ray = GridRay::new(StorageCoord { x: 0, y: 0, z: 0 }, [1, 0, 0], 2).unwrap();
    assert_eq!(world.blocks_in_region(region), Err(error));
    assert_eq!(world.raycast(ray), Err(error));
    assert_eq!(
        world.update_block_value(coordinate, BlockUpdate::default()),
        Err(error)
    );
    assert_eq!(world.remove_block_value(coordinate), Err(error));
}

#[test]
fn scenario_v2_callback_surfaces_injected_backend_failure() {
    let coordinate = StorageCoord { x: 1, y: 1, z: 1 };
    let world = FaultInjectedWorld::new(
        HashMapWorld::new(WorldExtent::cubic(4)),
        vec![FaultRule {
            operation: FaultOperation::Point,
            fail_on_call: 1,
            error: WorldAccessError::BackendFailure,
        }],
    );
    let probe = ReadProbe {
        points: vec![coordinate],
        regions: vec![],
        rays: vec![],
    };

    let error = capture_abi_read(&world, &probe, 227).unwrap_err();

    assert!(
        error.contains("BackendFailure"),
        "unexpected error: {error}"
    );
}

#[test]
fn in_memory_residency_contracts_match_exactly() {
    let extent = WorldExtent {
        x: 35,
        y: 23,
        z: 19,
    };
    let region = BoundedRegion::new(
        StorageCoord { x: 15, y: 7, z: 6 },
        StorageCoord {
            x: 35,
            y: 23,
            z: 19,
        },
        extent,
    )
    .unwrap();
    let oracle = HashMapWorld::new(extent);
    let chunked = ChunkedWorld::new(extent, WorldExtent { x: 16, y: 8, z: 7 }).unwrap();
    assert_eq!(
        oracle.is_region_resident(region),
        chunked.is_region_resident(region)
    );
    assert_eq!(
        oracle.pin_region(region).unwrap().region(),
        chunked.pin_region(region).unwrap().region()
    );
}

#[test]
fn future_faults_remain_explicitly_dependency_gated() {
    assert_eq!(PendingFaultCase::ShortRead.dependency_issues(), &[224]);
    assert_eq!(PendingFaultCase::FailedPrefetch.dependency_issues(), &[224]);
}

#[test]
fn benchmark_smoke_matrix_is_ci_safe_and_separates_future_memory_metrics() {
    let report = benchmark::run(&BenchmarkConfig::smoke());
    assert_eq!(report.measurements.len(), 3);
    assert!(report
        .measurements
        .iter()
        .all(|measurement| measurement.encoded_bytes.is_none()
            && measurement.overlay_memory_bytes.is_some()
            && measurement.pinned_memory_bytes.is_none()
            && measurement.operating_system_file_cache_bytes.is_none()));
}
