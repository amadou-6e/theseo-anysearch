mod outcomes;
mod predicates;
mod rewards;

use std::ptr;

const ABI_VERSION: u32 = 2;
const REWARD: u64 = 1;
const TRAINING_METRICS: u64 = 2;
const EVALUATION_METRICS: u64 = 4;
const PREDICATE: u64 = 8;
const OUTCOME: u64 = 16;

#[no_mangle]
pub extern "C" fn anysearch_extension_abi_version() -> u32 {
    ABI_VERSION
}
#[no_mangle]
pub extern "C" fn anysearch_extension_capabilities() -> u64 {
    REWARD | TRAINING_METRICS | EVALUATION_METRICS | PREDICATE | OUTCOME
}

unsafe fn write_json(output: *mut u8, capacity: usize, length: *mut usize, value: &[u8]) -> i32 {
    if output.is_null() || length.is_null() || value.len() > capacity {
        return 1;
    }
    ptr::copy_nonoverlapping(value.as_ptr(), output, value.len());
    *length = value.len();
    0
}

#[no_mangle]
pub unsafe extern "C" fn anysearch_compute_training_metrics_v1(
    _input: *const u8,
    _input_len: usize,
    output: *mut u8,
    capacity: usize,
    length: *mut usize,
) -> i32 {
    write_json(output, capacity, length, br#"{"native_hook_active":1.0}"#)
}

#[no_mangle]
pub unsafe extern "C" fn anysearch_compute_evaluation_metrics_v1(
    _input: *const u8,
    _input_len: usize,
    output: *mut u8,
    capacity: usize,
    length: *mut usize,
) -> i32 {
    write_json(output, capacity, length, br#"{"native_hook_active":1.0}"#)
}
