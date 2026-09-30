use std::ffi::c_char;

use crate::voxel::common::ABI_VERSION;

pub const MAX_COMPONENTS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy)]
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

impl Default for RewardComponentV2 {
    fn default() -> Self {
        Self {
            name: [0; 64],
            value: 0.0,
        }
    }
}

#[repr(C)]
pub struct RewardResultV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub mode: u32,
    pub reward: f64,
    pub component_count: u32,
    pub components: [RewardComponentV2; MAX_COMPONENTS],
}

impl Default for RewardResultV2 {
    fn default() -> Self {
        Self {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            mode: 0,
            reward: 0.0,
            component_count: 0,
            components: [RewardComponentV2::default(); MAX_COMPONENTS],
        }
    }
}
