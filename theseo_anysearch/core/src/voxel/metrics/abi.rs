pub const METRIC_BUFFER_SIZE: usize = 65_536;

pub type MetricFunctionV1 = unsafe extern "C" fn(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    capacity: usize,
    length: *mut usize,
) -> i32;
