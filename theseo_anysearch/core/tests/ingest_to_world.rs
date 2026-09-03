use theseo_core::voxel::world::{World, WorldState};
use theseo_core::voxel::world::ingest::{parse_ascii_stl, voxelize_mesh};

const TRIANGLE_STL: &str = r#"solid tri
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid tri"#;

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

#[test]
fn round_trip_single_triangle() {
    let mesh = parse_ascii_stl(TRIANGLE_STL).unwrap();
    let (placements, dropped) = voxelize_mesh(&mesh, (100, 100, 100), 1.0);
    assert_eq!(dropped, 0);
    assert!(!placements.is_empty());

    let mut world = WorldState::new();
    for p in &placements {
        world.set_block(p.coord, p.block.clone()).unwrap();
    }
    assert_eq!(world.len(), placements.len());
    for p in &placements {
        assert!(world.is_filled(p.coord));
    }
}

#[test]
fn round_trip_closed_mesh_is_solid() {
    let mesh = parse_ascii_stl(CUBE_STL).unwrap();
    let (placements, dropped) = voxelize_mesh(&mesh, (100, 100, 100), 5.0);
    assert_eq!(dropped, 0);

    // Solid fill must produce at least 5³ = 125 voxels for this scale
    assert!(placements.len() >= 125);

    let mut world = WorldState::new();
    for p in &placements {
        world.set_block(p.coord, p.block.clone()).unwrap();
    }
    assert_eq!(world.len(), placements.len());
}
