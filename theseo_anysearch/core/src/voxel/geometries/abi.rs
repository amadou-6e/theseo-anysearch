//! Stable C ABI for inert geometry proposals.
//!
//! All pointers and callbacks are borrowed for one call only. Providers must not
//! retain them. Output is UTF-8 JSON matching the Python `GeometryProposal`.

use crate::voxel::scenarios::abi::{WorldCoordV1, WorldQueryApiV1};

pub const GEOMETRY_ABI_VERSION_V1: u32 = 1;
pub const MAX_GEOMETRY_OUTPUT_BYTES: usize = 16 * 1024 * 1024;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryStatusV1 {
    Success = 0,
    InvalidArgument = 1,
    InvalidContext = 2,
    Panic = 3,
    SerializationFailure = 4,
    InsufficientBuffer = 5,
    OutputTooLarge = 6,
    BudgetExceeded = 7,
    ProviderError = 8,
}

#[repr(C)]
pub struct GeometryContextV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub seed: u64,
    pub attempt: u32,
    pub extent: WorldCoordV1,
    pub parameters_json: *const u8,
    pub parameters_json_len: usize,
    pub task_json: *const u8,
    pub task_json_len: usize,
    pub world: *const WorldQueryApiV1,
}

pub type GeometryFunctionV1 = unsafe extern "C" fn(
    context: *const GeometryContextV1,
    output: *mut u8,
    output_capacity: usize,
    required_length: *mut usize,
) -> GeometryStatusV1;

const _: () = {
    assert!(std::mem::size_of::<WorldCoordV1>() == 12);
    assert!(std::mem::size_of::<GeometryStatusV1>() == 4);
};
