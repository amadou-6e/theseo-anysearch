extern crate self as anysearch_extension;

use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub use anysearch_extension_macros::anysearch_reward;

pub const ABI_VERSION: u32 = 2;
pub const MAX_REWARD_COMPONENTS: usize = 8;
pub const MAX_COMPONENT_NAME_BYTES: usize = 63;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[anysearch_reward]
    fn collision_penalty(context: &RewardContext) -> RewardResult {
        let penalty = if context.collision { -0.25 } else { 0.0 };
        RewardResult::add(penalty).with_component("collision_penalty", penalty)
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
