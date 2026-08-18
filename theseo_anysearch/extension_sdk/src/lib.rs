extern crate self as anysearch_extension;

use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub use anysearch_extension_macros::{
    anysearch_outcome, anysearch_predicate, anysearch_reward, anysearch_scenario,
};

pub const ABI_VERSION: u32 = 2;
pub const MAX_REWARD_COMPONENTS: usize = 8;
pub const MAX_COMPONENT_NAME_BYTES: usize = 63;

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ScenarioContext {
    pub seed: u64,
    pub episode_index: u64,
    pub scope: String,
    pub grid_size: u16,
    pub filled_voxels: Vec<[i32; 3]>,
    pub action_mode: String,
    pub action_offsets: Vec<[i32; 3]>,
    pub previous_scenario: Option<serde_json::Value>,
    pub curriculum: serde_json::Value,
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ScenarioResult {
    pub start: [i32; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub goal: Option<[i32; 3]>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub route: Vec<[i32; 3]>,
    pub scenario_id: String,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl ScenarioResult {
    pub fn goal(start: [i32; 3], goal: [i32; 3], scenario_id: impl Into<String>) -> Self {
        Self {
            start,
            goal: Some(goal),
            route: Vec::new(),
            scenario_id: scenario_id.into(),
            metadata: serde_json::Map::new(),
        }
    }

    pub fn route(start: [i32; 3], route: Vec<[i32; 3]>, scenario_id: impl Into<String>) -> Self {
        Self {
            start,
            goal: None,
            route,
            scenario_id: scenario_id.into(),
            metadata: serde_json::Map::new(),
        }
    }
}

/// Convert an ergonomic scenario function into the variable-length JSON ABI.
///
/// # Safety
/// All pointer/length pairs must identify readable or writable buffers.
pub unsafe fn export_scenario_v1(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_capacity: usize,
    output_len: *mut usize,
    function: fn(&ScenarioContext) -> ScenarioResult,
) -> i32 {
    if input.is_null() || output.is_null() || output_len.is_null() {
        return 1;
    }
    let context: ScenarioContext =
        match serde_json::from_slice(std::slice::from_raw_parts(input, input_len)) {
            Ok(value) => value,
            Err(_) => return 2,
        };
    let result = match catch_unwind(AssertUnwindSafe(|| function(&context))) {
        Ok(value) => value,
        Err(_) => return 3,
    };
    let encoded = match serde_json::to_vec(&result) {
        Ok(value) => value,
        Err(_) => return 4,
    };
    if encoded.len() > output_capacity {
        return 5;
    }
    std::ptr::copy_nonoverlapping(encoded.as_ptr(), output, encoded.len());
    *output_len = encoded.len();
    0
}

#[repr(C)]
pub struct RewardContextV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub step: u64,
    pub action_index: i32,
    pub previous_cursor: [i32; 3],
    pub cursor: [i32; 3],
    pub goal: [i32; 3],
    pub has_goal: u8,
    pub invalid_action: u8,
    pub collision: u8,
    pub goal_reached: u8,
    pub terminated: u8,
    pub truncated: u8,
    pub consecutive_collisions: u32,
    pub previous_goal_distance: f64,
    pub goal_distance: f64,
    pub standard_reward: f64,
    pub segment_step: u64,
    pub segment_length: u64,
    pub parameters_json: *const u8,
    pub parameters_json_len: usize,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RewardComponentV2 {
    pub name: [c_char; 64],
    pub value: f64,
}

#[repr(C)]
pub struct RewardResultV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub mode: u32,
    pub reward: f64,
    pub component_count: u32,
    pub components: [RewardComponentV2; MAX_REWARD_COMPONENTS],
}

#[derive(Clone, Debug)]
pub struct RewardContext {
    pub step: u64,
    pub action_index: i32,
    pub previous_cursor: [i32; 3],
    pub cursor: [i32; 3],
    pub goal: Option<[i32; 3]>,
    pub invalid_action: bool,
    pub collision: bool,
    pub goal_reached: bool,
    pub terminated: bool,
    pub truncated: bool,
    pub consecutive_collisions: u32,
    pub previous_goal_distance: f64,
    pub goal_distance: f64,
    pub standard_reward: f64,
    pub segment_step: u64,
    pub segment_length: u64,
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RewardMode {
    Add,
    Replace,
}

#[derive(Clone, Debug)]
pub struct RewardComponent {
    pub name: String,
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct RewardResult {
    pub reward: f64,
    pub components: Vec<RewardComponent>,
    pub mode: RewardMode,
}

impl RewardResult {
    pub fn add(reward: f64) -> Self {
        Self {
            reward,
            components: Vec::new(),
            mode: RewardMode::Add,
        }
    }

    pub fn replace(reward: f64) -> Self {
        Self {
            reward,
            components: Vec::new(),
            mode: RewardMode::Replace,
        }
    }

    pub fn with_component(mut self, name: impl Into<String>, value: f64) -> Self {
        self.components.push(RewardComponent {
            name: name.into(),
            value,
        });
        self
    }
}

impl From<&RewardContextV2> for RewardContext {
    fn from(value: &RewardContextV2) -> Self {
        let parameter_bytes = if value.parameters_json.is_null() {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(value.parameters_json, value.parameters_json_len) }
        };
        let parameters = if parameter_bytes.is_empty() {
            serde_json::Map::new()
        } else {
            serde_json::from_slice(parameter_bytes)
                .expect("AnySearch supplied invalid custom reward parameter JSON")
        };
        Self {
            step: value.step,
            action_index: value.action_index,
            previous_cursor: value.previous_cursor,
            cursor: value.cursor,
            goal: (value.has_goal != 0).then_some(value.goal),
            invalid_action: value.invalid_action != 0,
            collision: value.collision != 0,
            goal_reached: value.goal_reached != 0,
            terminated: value.terminated != 0,
            truncated: value.truncated != 0,
            consecutive_collisions: value.consecutive_collisions,
            previous_goal_distance: value.previous_goal_distance,
            goal_distance: value.goal_distance,
            standard_reward: value.standard_reward,
            segment_step: value.segment_step,
            segment_length: value.segment_length,
            parameters,
        }
    }
}

/// Convert one ergonomic reward function into the stable v2 ABI response.
///
/// # Safety
/// `context` and `result` must point to writable/readable v2 ABI structures.
pub unsafe fn export_reward_v2(
    context: *const RewardContextV2,
    result: *mut RewardResultV2,
    reward_function: fn(&RewardContext) -> RewardResult,
) -> i32 {
    if context.is_null() || result.is_null() {
        return 1;
    }
    let context = &*context;
    if context.abi_version != ABI_VERSION
        || context.struct_size as usize != std::mem::size_of::<RewardContextV2>()
    {
        return 2;
    }

    let computed = match catch_unwind(AssertUnwindSafe(|| {
        reward_function(&RewardContext::from(context))
    })) {
        Ok(value) => value,
        Err(_) => return 3,
    };
    if !computed.reward.is_finite() || computed.components.len() > MAX_REWARD_COMPONENTS {
        return 4;
    }

    let raw = &mut *result;
    raw.abi_version = ABI_VERSION;
    raw.struct_size = std::mem::size_of::<RewardResultV2>() as u32;
    raw.mode = match computed.mode {
        RewardMode::Add => 0,
        RewardMode::Replace => 1,
    };
    raw.reward = computed.reward;
    raw.component_count = computed.components.len() as u32;

    for (index, component) in computed.components.iter().enumerate() {
        if !component.value.is_finite()
            || component.name.is_empty()
            || component.name.len() > MAX_COMPONENT_NAME_BYTES
            || !component.name.is_ascii()
        {
            return 5;
        }
        raw.components[index].name = [0; 64];
        for (target, source) in raw.components[index]
            .name
            .iter_mut()
            .zip(component.name.as_bytes())
        {
            *target = *source as c_char;
        }
        raw.components[index].value = component.value;
    }
    0
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ActionHistoryEntryV2 {
    pub action_index: i32,
    pub previous_cursor: [i32; 3],
    pub cursor: [i32; 3],
    pub feasible: u8,
    pub collision: u8,
}

#[repr(C)]
pub struct PredicateContextV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub step: u64,
    pub grid_size: u16,
    pub action_index: i32,
    pub cursor: [i32; 3],
    pub destination: [i32; 3],
    pub goal: [i32; 3],
    pub has_goal: u8,
    pub valid_action: u8,
    pub destination_in_bounds: u8,
    pub destination_blocked: u8,
    pub observation_filled: usize,
    pub observation_steps_remaining: u32,
    pub observation_goal_distance: u32,
    pub has_observation_goal_distance: u8,
    pub history: *const ActionHistoryEntryV2,
    pub history_len: usize,
    pub parameters_json: *const u8,
    pub parameters_json_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PredicateResultV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub feasible: u8,
}

#[derive(Clone, Debug)]
pub struct ActionHistoryEntry {
    pub action_index: i32,
    pub previous_cursor: [i32; 3],
    pub cursor: [i32; 3],
    pub feasible: bool,
    pub collision: bool,
}

#[derive(Clone, Debug)]
pub struct PredicateContext {
    pub step: u64,
    pub grid_size: u16,
    pub action_index: i32,
    pub cursor: [i32; 3],
    pub destination: [i32; 3],
    pub goal: Option<[i32; 3]>,
    pub valid_action: bool,
    pub destination_in_bounds: bool,
    pub destination_blocked: bool,
    pub observation_filled: usize,
    pub observation_steps_remaining: u32,
    pub observation_goal_distance: Option<u32>,
    pub history: Vec<ActionHistoryEntry>,
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Copy, Debug)]
pub struct PredicateResult {
    pub feasible: bool,
}
impl PredicateResult {
    pub fn allow() -> Self {
        Self { feasible: true }
    }
    pub fn deny() -> Self {
        Self { feasible: false }
    }
}

#[repr(C)]
pub struct OutcomeContextV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub step: u64,
    pub grid_size: u16,
    pub action_index: i32,
    pub cursor: [i32; 3],
    pub destination: [i32; 3],
    pub goal: [i32; 3],
    pub has_goal: u8,
    pub history: *const ActionHistoryEntryV2,
    pub history_len: usize,
    pub parameters_json: *const u8,
    pub parameters_json_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OutcomeResultV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub set_cursor: u8,
    pub cursor: [i32; 3],
    pub place_voxel: u8,
    pub place_coord: [i32; 3],
    pub remove_voxel: u8,
    pub remove_coord: [i32; 3],
}

#[derive(Clone, Debug)]
pub struct OutcomeContext {
    pub step: u64,
    pub grid_size: u16,
    pub action_index: i32,
    pub cursor: [i32; 3],
    pub destination: [i32; 3],
    pub goal: Option<[i32; 3]>,
    pub history: Vec<ActionHistoryEntry>,
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Debug, Default)]
pub struct OutcomeMutations {
    pub cursor: Option<[i32; 3]>,
    pub place: Option<[i32; 3]>,
    pub remove: Option<[i32; 3]>,
}
impl OutcomeMutations {
    pub fn set_cursor(&mut self, coord: [i32; 3]) {
        self.cursor = Some(coord);
    }
    pub fn place_voxel(&mut self, coord: [i32; 3]) {
        self.place = Some(coord);
    }
    pub fn remove_voxel(&mut self, coord: [i32; 3]) {
        self.remove = Some(coord);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OutcomeResult;
impl OutcomeResult {
    pub fn applied() -> Self {
        Self
    }
}

fn parameters(pointer: *const u8, length: usize) -> serde_json::Map<String, serde_json::Value> {
    if pointer.is_null() || length == 0 {
        return serde_json::Map::new();
    }
    let bytes = unsafe { std::slice::from_raw_parts(pointer, length) };
    serde_json::from_slice(bytes)
        .expect("AnySearch supplied invalid action extension parameter JSON")
}
fn history(pointer: *const ActionHistoryEntryV2, length: usize) -> Vec<ActionHistoryEntry> {
    if pointer.is_null() || length == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(pointer, length) }
        .iter()
        .map(|entry| ActionHistoryEntry {
            action_index: entry.action_index,
            previous_cursor: entry.previous_cursor,
            cursor: entry.cursor,
            feasible: entry.feasible != 0,
            collision: entry.collision != 0,
        })
        .collect()
}

pub unsafe fn export_predicate_v2(
    context: *const PredicateContextV2,
    result: *mut PredicateResultV2,
    function: fn(&PredicateContext) -> PredicateResult,
) -> i32 {
    if context.is_null() || result.is_null() {
        return 1;
    }
    let raw = &*context;
    if raw.abi_version != ABI_VERSION
        || raw.struct_size as usize != std::mem::size_of::<PredicateContextV2>()
    {
        return 2;
    }
    let context = PredicateContext {
        step: raw.step,
        grid_size: raw.grid_size,
        action_index: raw.action_index,
        cursor: raw.cursor,
        destination: raw.destination,
        goal: (raw.has_goal != 0).then_some(raw.goal),
        valid_action: raw.valid_action != 0,
        destination_in_bounds: raw.destination_in_bounds != 0,
        destination_blocked: raw.destination_blocked != 0,
        observation_filled: raw.observation_filled,
        observation_steps_remaining: raw.observation_steps_remaining,
        observation_goal_distance: (raw.has_observation_goal_distance != 0)
            .then_some(raw.observation_goal_distance),
        history: history(raw.history, raw.history_len),
        parameters: parameters(raw.parameters_json, raw.parameters_json_len),
    };
    let computed = match catch_unwind(AssertUnwindSafe(|| function(&context))) {
        Ok(value) => value,
        Err(_) => return 3,
    };
    *result = PredicateResultV2 {
        abi_version: ABI_VERSION,
        struct_size: std::mem::size_of::<PredicateResultV2>() as u32,
        feasible: u8::from(computed.feasible),
    };
    0
}

pub unsafe fn export_outcome_v2(
    context: *const OutcomeContextV2,
    result: *mut OutcomeResultV2,
    function: fn(&OutcomeContext, &mut OutcomeMutations) -> OutcomeResult,
) -> i32 {
    if context.is_null() || result.is_null() {
        return 1;
    }
    let raw = &*context;
    if raw.abi_version != ABI_VERSION
        || raw.struct_size as usize != std::mem::size_of::<OutcomeContextV2>()
    {
        return 2;
    }
    let context = OutcomeContext {
        step: raw.step,
        grid_size: raw.grid_size,
        action_index: raw.action_index,
        cursor: raw.cursor,
        destination: raw.destination,
        goal: (raw.has_goal != 0).then_some(raw.goal),
        history: history(raw.history, raw.history_len),
        parameters: parameters(raw.parameters_json, raw.parameters_json_len),
    };
    let mut mutations = OutcomeMutations::default();
    if catch_unwind(AssertUnwindSafe(|| function(&context, &mut mutations))).is_err() {
        return 3;
    }
    *result = OutcomeResultV2 {
        abi_version: ABI_VERSION,
        struct_size: std::mem::size_of::<OutcomeResultV2>() as u32,
        set_cursor: u8::from(mutations.cursor.is_some()),
        cursor: mutations.cursor.unwrap_or([0; 3]),
        place_voxel: u8::from(mutations.place.is_some()),
        place_coord: mutations.place.unwrap_or([0; 3]),
        remove_voxel: u8::from(mutations.remove.is_some()),
        remove_coord: mutations.remove.unwrap_or([0; 3]),
    };
    0
}
#[cfg(test)]
mod tests {
    use super::*;

    #[anysearch_reward]
    fn collision_penalty(context: &RewardContext) -> RewardResult {
        let penalty = if context.collision { -0.25 } else { 0.0 };
        RewardResult::add(penalty).with_component("collision_penalty", penalty)
    }

    #[anysearch_predicate]
    fn deny_blocked(context: &PredicateContext) -> PredicateResult {
        if context.destination_blocked {
            PredicateResult::deny()
        } else {
            PredicateResult::allow()
        }
    }

    #[anysearch_outcome]
    fn place_destination(
        context: &OutcomeContext,
        mutations: &mut OutcomeMutations,
    ) -> OutcomeResult {
        mutations.set_cursor(context.destination);
        mutations.place_voxel(context.destination);
        OutcomeResult::applied()
    }

    #[anysearch_scenario]
    fn adjacent_scenario(context: &ScenarioContext) -> ScenarioResult {
        ScenarioResult::goal(
            [4, 4, 4],
            [5, 4, 4],
            format!("episode-{}", context.episode_index),
        )
    }

    #[test]
    fn scenario_attribute_exports_json_result() {
        let input = br#"{"seed":42,"episode_index":7,"scope":"evaluation","grid_size":8,"filled_voxels":[],"action_mode":"discrete_26","action_offsets":[[1,0,0]],"previous_scenario":null,"curriculum":{},"parameters":{}}"#;
        let mut output = [0_u8; 1024];
        let mut output_len = 0_usize;
        let status = unsafe {
            anysearch_scenario_adjacent_scenario_v1(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut output_len,
            )
        };
        assert_eq!(status, 0);
        let value: serde_json::Value = serde_json::from_slice(&output[..output_len]).unwrap();
        assert_eq!(value["scenario_id"], "episode-7");
        assert_eq!(value["goal"], serde_json::json!([5, 4, 4]));
    }

    #[test]
    fn predicate_and_outcome_attributes_export_v2_symbols() {
        let raw_history = [ActionHistoryEntryV2 {
            action_index: 2,
            feasible: 0,
            collision: 1,
            ..Default::default()
        }];
        let predicate_context = PredicateContextV2 {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<PredicateContextV2>() as u32,
            step: 4,
            grid_size: 8,
            action_index: 2,
            cursor: [2, 2, 2],
            destination: [3, 2, 2],
            goal: [6, 6, 6],
            has_goal: 1,
            valid_action: 1,
            destination_in_bounds: 1,
            destination_blocked: 1,
            observation_filled: 3,
            observation_steps_remaining: 6,
            observation_goal_distance: 11,
            has_observation_goal_distance: 1,
            history: raw_history.as_ptr(),
            history_len: raw_history.len(),
            parameters_json: std::ptr::null(),
            parameters_json_len: 0,
        };
        let mut predicate_result = PredicateResultV2 {
            abi_version: 0,
            struct_size: 0,
            feasible: 1,
        };
        assert_eq!(
            unsafe {
                anysearch_predicate_deny_blocked_v2(&predicate_context, &mut predicate_result)
            },
            0
        );
        assert_eq!(predicate_result.feasible, 0);

        let outcome_context = OutcomeContextV2 {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<OutcomeContextV2>() as u32,
            step: 4,
            grid_size: 8,
            action_index: 2,
            cursor: [2, 2, 2],
            destination: [3, 2, 2],
            goal: [6, 6, 6],
            has_goal: 1,
            history: raw_history.as_ptr(),
            history_len: raw_history.len(),
            parameters_json: std::ptr::null(),
            parameters_json_len: 0,
        };
        let mut outcome_result = OutcomeResultV2 {
            abi_version: 0,
            struct_size: 0,
            set_cursor: 0,
            cursor: [0; 3],
            place_voxel: 0,
            place_coord: [0; 3],
            remove_voxel: 0,
            remove_coord: [0; 3],
        };
        assert_eq!(
            unsafe {
                anysearch_outcome_place_destination_v2(&outcome_context, &mut outcome_result)
            },
            0
        );
        assert_eq!(outcome_result.cursor, [3, 2, 2]);
        assert_eq!(outcome_result.place_coord, [3, 2, 2]);
    }
    #[test]
    fn attribute_generates_and_encodes_v2_export() {
        let parameters = br#"{"scale":2.5}"#;
        let context = RewardContextV2 {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<RewardContextV2>() as u32,
            step: 3,
            action_index: 4,
            previous_cursor: [1, 1, 1],
            cursor: [1, 1, 1],
            goal: [5, 5, 5],
            has_goal: 1,
            invalid_action: 0,
            collision: 1,
            goal_reached: 0,
            terminated: 0,
            truncated: 0,
            consecutive_collisions: 2,
            previous_goal_distance: 7.0,
            goal_distance: 7.0,
            standard_reward: -0.01,
            segment_step: 3,
            segment_length: 7,
            parameters_json: parameters.as_ptr(),
            parameters_json_len: parameters.len(),
        };
        assert_eq!(
            RewardContext::from(&context).parameters["scale"].as_f64(),
            Some(2.5)
        );
        let mut result: RewardResultV2 = unsafe { std::mem::zeroed() };

        let status = unsafe { anysearch_reward_collision_penalty_v2(&context, &mut result) };

        assert_eq!(status, 0);
        assert_eq!(result.abi_version, ABI_VERSION);
        assert_eq!(result.mode, 0);
        assert_eq!(result.reward, -0.25);
        assert_eq!(result.component_count, 1);
        assert_eq!(result.components[0].value, -0.25);
        let name = result.components[0]
            .name
            .iter()
            .take_while(|byte| **byte != 0)
            .map(|byte| *byte as u8)
            .collect::<Vec<_>>();
        assert_eq!(name, b"collision_penalty");
    }
}
