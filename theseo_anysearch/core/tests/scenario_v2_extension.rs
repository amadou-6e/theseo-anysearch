use std::{path::Path, process::Command};
use theseo_core::voxel::{
    scenarios::{NativeScenarioV2, ScenarioInvocationV2},
    world::{Block, StorageCoord, WorldMutation, WorldState},
};

#[test]
fn independently_compiled_v2_extension_queries_the_real_world() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..");
    let manifest = root
        .join("usage")
        .join("experiments")
        .join("showcase")
        .join("scenario_world_query_v2")
        .join("extension")
        .join("Cargo.toml");
    let target = std::env::temp_dir().join(format!("anysearch-scenario-v2-{}", std::process::id()));
    let status = Command::new(env!("CARGO"))
        .args(["build", "--release", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TARGET_DIR", &target)
        .status()
        .expect("launch extension build");
    assert!(status.success(), "independent extension build failed");
    #[cfg(target_os = "windows")]
    let library = target
        .join("release")
        .join("anysearch_scenario_world_query_v2.dll");
    #[cfg(target_os = "linux")]
    let library = target
        .join("release")
        .join("libanysearch_scenario_world_query_v2.so");
    #[cfg(target_os = "macos")]
    let library = target
        .join("release")
        .join("libanysearch_scenario_world_query_v2.dylib");

    let extension = NativeScenarioV2::load(&library, "queried_route").expect("load v2 extension");
    let mut world = WorldState::new();
    world
        .set_block_value(StorageCoord { x: 4, y: 1, z: 1 }, Block::default())
        .unwrap();
    let result = extension
        .invoke(
            &world,
            &ScenarioInvocationV2 {
                seed: 42,
                episode_index: 3,
                grid_size: 32,
                scope: "training",
                action_mode: "discrete_26",
                action_offsets_json: "[]",
                previous_scenario_json: "null",
                curriculum_json: "{}",
                parameters_json: "{}",
                candidate_index_path: None,
                world_identity: "",
            },
        )
        .expect("invoke v2 extension");
    let value: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(value["scenario_id"], "queried-3");
    assert_eq!(value["goal"], serde_json::json!([3, 1, 1]));

    type ScenarioV1 = unsafe extern "C" fn(*const u8, usize, *mut u8, usize, *mut usize) -> i32;
    let loaded = unsafe { libloading::Library::new(&library) }.unwrap();
    let legacy: libloading::Symbol<ScenarioV1> =
        unsafe { loaded.get(b"anysearch_scenario_legacy_route_v1\0") }.unwrap();
    let input = br#"{"seed":42,"episode_index":7,"scope":"evaluation","grid_size":8,"filled_voxels":[],"action_mode":"discrete_26","action_offsets":[],"previous_scenario":null,"curriculum":{},"parameters":{}}"#;
    let mut output = vec![0u8; 4096];
    let mut written = 0;
    assert_eq!(
        unsafe {
            legacy(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut written,
            )
        },
        0
    );
    let legacy_value: serde_json::Value = serde_json::from_slice(&output[..written]).unwrap();
    assert_eq!(legacy_value["scenario_id"], "legacy-7");
}
