use std::ffi::c_char;
use std::ptr;

const ABI_VERSION: u32 = 1;
const REWARD: u64 = 1;
const TRAINING_METRICS: u64 = 2;
const EVALUATION_METRICS: u64 = 4;

#[repr(C)]
pub struct RewardContextV1 {
    pub abi_version: u32, pub struct_size: u32, pub step: u64, pub action_index: i32,
    pub previous_cursor: [i32; 3], pub cursor: [i32; 3], pub goal: [i32; 3],
    pub has_goal: u8, pub invalid_action: u8, pub collision: u8,
    pub terminated: u8, pub truncated: u8,
    pub previous_goal_distance: f64, pub goal_distance: f64, pub standard_reward: f64,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct RewardComponentV1 { pub name: [c_char; 64], pub value: f64 }
#[repr(C)]
pub struct RewardResultV1 {
    pub abi_version: u32, pub struct_size: u32, pub mode: u32, pub reward: f64,
    pub component_count: u32, pub components: [RewardComponentV1; 8],
}

#[no_mangle]
pub extern "C" fn anysearch_extension_abi_version() -> u32 { ABI_VERSION }
#[no_mangle]
pub extern "C" fn anysearch_extension_capabilities() -> u64 {
    REWARD | TRAINING_METRICS | EVALUATION_METRICS
}

#[no_mangle]
pub unsafe extern "C" fn anysearch_compute_reward_v1(
    context: *const RewardContextV1, result: *mut RewardResultV1,
) -> i32 {
    if context.is_null() || result.is_null() { return 1; }
    let context = &*context;
    if context.abi_version != ABI_VERSION { return 2; }
    let penalty = if context.collision != 0 { -0.02 } else { 0.0 };
    (*result).mode = 0;
    (*result).reward = penalty;
    (*result).component_count = 1;
    let name = b"native_collision\0";
    ptr::copy_nonoverlapping(name.as_ptr(), (*result).components[0].name.as_mut_ptr() as *mut u8, name.len());
    (*result).components[0].value = penalty;
    0
}

unsafe fn write_json(output: *mut u8, capacity: usize, length: *mut usize, value: &[u8]) -> i32 {
    if output.is_null() || length.is_null() || value.len() > capacity { return 1; }
    ptr::copy_nonoverlapping(value.as_ptr(), output, value.len());
    *length = value.len();
    0
}

#[no_mangle]
pub unsafe extern "C" fn anysearch_compute_training_metrics_v1(
    _input: *const u8, _input_len: usize, output: *mut u8,
    capacity: usize, length: *mut usize,
) -> i32 { write_json(output, capacity, length, br#"{"native_hook_active":1.0}"#) }

#[no_mangle]
pub unsafe extern "C" fn anysearch_compute_evaluation_metrics_v1(
    _input: *const u8, _input_len: usize, output: *mut u8,
    capacity: usize, length: *mut usize,
) -> i32 { write_json(output, capacity, length, br#"{"native_hook_active":1.0}"#) }
