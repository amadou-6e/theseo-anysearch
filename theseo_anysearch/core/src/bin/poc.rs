use theseo_core::{
    bridge::{run_submission_poc, EnvConfig, GeometrySubmission},
    environments::Environment,
    voxel::{VoxelAction, VoxelEnv},
    world::{
        ingest::{parse_ascii_stl, voxelize_mesh},
        Block, BlockUpdate, World, WorldState,
    },
};

fn main() {
    let mut world = WorldState::new();

    let _ = world.set_block((10, 10, 10), Block::default());
    let _ = world.update_block(
        (10, 10, 10),
        BlockUpdate {
            kind: Some(2),
            active: Some(true),
            reward_weight: Some(2.5),
        },
    );
    let _ = world.remove_block((10, 10, 10));

    let sample_stl = r#"
solid tri
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid tri
"#;

    let mesh = parse_ascii_stl(sample_stl).expect("stl parse should work in poc");
    let placements = voxelize_mesh(&mesh, (50, 50, 50), 2.0);
    for p in &placements {
        let _ = world.set_block(p.coord, p.block.clone());
    }

    let mut env = VoxelEnv::new(world.clone(), 8);
    let initial = env.reset(7);
    let step = env.step(VoxelAction::Noop);
    let step2 = env.step(VoxelAction::Place((55, 55, 55)));
    let step3 = env.step(VoxelAction::Remove((55, 55, 55)));

    let submission = GeometrySubmission {
        name: "triangle".to_string(),
        stl_ascii: sample_stl.to_string(),
        origin: (100, 100, 100),
        scale: 1.0,
    };
    let env_cfg = EnvConfig { max_steps: 12 };
    let bridge_summary = run_submission_poc(&submission, &env_cfg).expect("bridge poc should work");

    println!(
        "PoC OK | initial_filled={} after_steps=[{:.2}, {:.2}, {:.2}] bridge=\"{}\"",
        initial.filled, step.reward, step2.reward, step3.reward, bridge_summary
    );
}
