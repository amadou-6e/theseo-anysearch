use crate::voxel::{actions::ActionHistoryEntryV2, common::ABI_VERSION};

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

impl Default for PredicateResultV2 {
    fn default() -> Self {
        Self {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            feasible: 1,
        }
    }
}
