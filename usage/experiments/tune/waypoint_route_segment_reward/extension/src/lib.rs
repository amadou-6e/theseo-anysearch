mod rewards;

use serde::Deserialize;
use std::ptr;

const ABI_VERSION: u32 = 2;
const REWARD: u64 = 1;
const EVALUATION_METRICS: u64 = 4;

#[derive(Deserialize)]
struct EvaluationContext {
    #[serde(default)]
    final_infos: Vec<FinalInfo>,
}

#[derive(Deserialize)]
struct FinalInfo {
    #[serde(default)]
    route_waypoints_reached: u64,
    #[serde(default)]
    route_waypoints_total: u64,
}

#[no_mangle]
pub extern "C" fn anysearch_extension_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn anysearch_extension_capabilities() -> u64 {
    REWARD | EVALUATION_METRICS
}

fn evaluation_metrics(input: &[u8]) -> Result<Vec<u8>, serde_json::Error> {
    let context: EvaluationContext = serde_json::from_slice(input)?;
    let waypoints_reached = context
        .final_infos
        .iter()
        .map(|info| info.route_waypoints_reached)
        .sum::<u64>();
    let waypoints_total = context
        .final_infos
        .iter()
        .map(|info| info.route_waypoints_total)
        .sum::<u64>();
    let completion_fraction = if waypoints_total == 0 {
        0.0
    } else {
        waypoints_reached as f64 / waypoints_total as f64
    };
    serde_json::to_vec(&serde_json::json!({
        "waypoints_reached": waypoints_reached as f64,
        "waypoint_completion_fraction": completion_fraction,
    }))
}

#[no_mangle]
pub unsafe extern "C" fn anysearch_compute_evaluation_metrics_v1(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    capacity: usize,
    length: *mut usize,
) -> i32 {
    if input.is_null() || output.is_null() || length.is_null() {
        return 1;
    }
    let input = std::slice::from_raw_parts(input, input_len);
    let value = match evaluation_metrics(input) {
        Ok(value) => value,
        Err(_) => return 2,
    };
    if value.len() > capacity {
        return 3;
    }
    ptr::copy_nonoverlapping(value.as_ptr(), output, value.len());
    *length = value.len();
    0
}

#[cfg(test)]
mod tests {
    use super::evaluation_metrics;

    #[test]
    fn aggregates_waypoint_progress_across_evaluation_episodes() {
        let output = evaluation_metrics(
            br#"{"final_infos":[{"route_waypoints_reached":15,"route_waypoints_total":150},{"route_waypoints_reached":150,"route_waypoints_total":150}]}"#,
        )
        .unwrap();
        let metrics: serde_json::Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(metrics["waypoints_reached"], 165.0);
        assert_eq!(metrics["waypoint_completion_fraction"], 0.55);
    }
}
