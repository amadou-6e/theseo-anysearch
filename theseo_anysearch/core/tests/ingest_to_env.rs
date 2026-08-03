use theseo_core::environments::Environment;
use theseo_core::surface::{SurfaceAction, SurfaceEnv};
use theseo_core::voxel::world::ingest::{parse_ascii_stl, voxelize_mesh};

const CUBE_STL: &str = r#"solid cube
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 1 1 0
    endloop
  endfacet
  facet normal 0 0 -1
    outer loop
      vertex 0 0 0
      vertex 1 1 0
      vertex 0 1 0
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 1 1 1
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 0 1
    outer loop
      vertex 0 0 1
      vertex 0 1 1
      vertex 1 1 1
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 1 0 1
    endloop
  endfacet
  facet normal 0 -1 0
    outer loop
      vertex 0 0 0
      vertex 1 0 1
      vertex 0 0 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 1 1 1
      vertex 0 1 1
    endloop
  endfacet
  facet normal 0 1 0
    outer loop
      vertex 0 1 0
      vertex 1 1 0
      vertex 1 1 1
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 1 1
      vertex 0 1 0
    endloop
  endfacet
  facet normal -1 0 0
    outer loop
      vertex 0 0 0
      vertex 0 0 1
      vertex 0 1 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 1 0
      vertex 1 1 1
    endloop
  endfacet
  facet normal 1 0 0
    outer loop
      vertex 1 0 0
      vertex 1 1 1
      vertex 1 0 1
    endloop
  endfacet
endsolid cube"#;

fn make_env(agent_count: usize, max_steps: u32) -> SurfaceEnv {
    let mesh = parse_ascii_stl(CUBE_STL).unwrap();
    let placements = voxelize_mesh(&mesh, (100, 100, 100), 5.0);
    let coords: Vec<_> = placements.iter().map(|p| p.coord).collect();
    SurfaceEnv::from_filled_surface(&coords, agent_count, max_steps)
}

#[test]
fn surface_env_from_stl_completes() {
    let mut env = make_env(2, 500);
    env.reset(42);
    let mut done = false;
    for _ in 0..500 {
        let sr = env.step(SurfaceAction::StepAll);
        if sr.done {
            done = true;
            break;
        }
    }
    assert!(done, "episode did not complete within max_steps");
}

#[test]
fn surface_env_deterministic_across_runs() {
    // Resetting the same env twice with the same seed must yield identical assignments.
    let mut env = make_env(2, 500);
    let obs1 = env.reset(7);
    let snap1: Vec<_> = obs1.agents.iter().map(|a| (a.current, a.target)).collect();
    let obs2 = env.reset(7);
    let snap2: Vec<_> = obs2.agents.iter().map(|a| (a.current, a.target)).collect();
    assert_eq!(snap1, snap2);
}
