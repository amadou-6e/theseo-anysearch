//! Stable C ABI for scenario-v2 world queries.
//!
//! Coordinates are zero-based storage coordinates. Regions are minimum-inclusive and
//! maximum-exclusive. Every pointer is valid only for the duration of one scenario call.

use std::ffi::c_void;

pub const WORLD_QUERY_ABI_VERSION: u32 = 1;
pub const SCENARIO_ABI_VERSION_V2: u32 = 2;
pub const MAX_SCENARIO_OUTPUT_BYTES: usize = 1_048_576;
pub const MAX_REGION_RESULTS: usize = 1_000_000;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueryStatus {
    EmptyOrMiss = 0,
    BlockHit = 1,
    InvalidArgument = 2,
    OutOfBounds = 3,
    InsufficientBuffer = 4,
    StaleToken = 5,
    BackendFailure = 6,
    Unsupported = 7,
    HostFailure = 8,
}

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScenarioStatusV2 {
    Success = 0,
    InvalidArgument = 1,
    InvalidContext = 2,
    Panic = 3,
    SerializationFailure = 4,
    InsufficientBuffer = 5,
    OutputTooLarge = 6,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorldCoordV1 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorldBlockV1 {
    pub kind: u8,
    pub active: u8,
    pub reserved: [u8; 2],
    pub reward_weight: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorldRegionEntryV1 {
    pub coordinate: WorldCoordV1,
    pub block: WorldBlockV1,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorldRayHitV1 {
    pub coordinate: WorldCoordV1,
    pub block: WorldBlockV1,
    pub steps: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorldRayStepV1 {
    pub x: i8,
    pub y: i8,
    pub z: i8,
}

pub type PointQueryV1 = unsafe extern "C" fn(
    context: *mut c_void,
    call_token: u64,
    coordinate: WorldCoordV1,
    output: *mut WorldBlockV1,
) -> QueryStatus;

pub type RegionQueryV1 = unsafe extern "C" fn(
    context: *mut c_void,
    call_token: u64,
    minimum: WorldCoordV1,
    maximum_exclusive: WorldCoordV1,
    output: *mut WorldRegionEntryV1,
    output_capacity: usize,
    required_length: *mut usize,
) -> QueryStatus;

pub type RayQueryV1 = unsafe extern "C" fn(
    context: *mut c_void,
    call_token: u64,
    origin: WorldCoordV1,
    step: WorldRayStepV1,
    maximum_steps: u32,
    output: *mut WorldRayHitV1,
) -> QueryStatus;

pub type CountQueryV1 = unsafe extern "C" fn(
    context: *mut c_void,
    call_token: u64,
    minimum: WorldCoordV1,
    maximum_exclusive: WorldCoordV1,
    output: *mut u64,
) -> QueryStatus;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct WorldQueryApiV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub coordinate_size: u32,
    pub block_size: u32,
    pub region_entry_size: u32,
    pub ray_hit_size: u32,
    pub context: *mut c_void,
    pub call_token: u64,
    pub point: Option<PointQueryV1>,
    pub region: Option<RegionQueryV1>,
    pub ray: Option<RayQueryV1>,
    pub count_region: Option<CountQueryV1>,
}

#[repr(C)]
pub struct ScenarioContextV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub seed: u64,
    pub episode_index: u64,
    pub grid_size: u32,
    pub scope: *const u8,
    pub scope_len: usize,
    pub action_mode: *const u8,
    pub action_mode_len: usize,
    pub action_offsets_json: *const u8,
    pub action_offsets_json_len: usize,
    pub previous_scenario_json: *const u8,
    pub previous_scenario_json_len: usize,
    pub curriculum_json: *const u8,
    pub curriculum_json_len: usize,
    pub parameters_json: *const u8,
    pub parameters_json_len: usize,
    pub world: *const WorldQueryApiV1,
}

pub type ScenarioFunctionV2 = unsafe extern "C" fn(
    context: *const ScenarioContextV2,
    output: *mut u8,
    output_capacity: usize,
    required_length: *mut usize,
) -> ScenarioStatusV2;

const _: () = {
    assert!(std::mem::size_of::<WorldCoordV1>() == 12);
    assert!(std::mem::size_of::<WorldBlockV1>() == 8);
};
