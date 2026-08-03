use crate::voxel::{actions::ActionHistoryEntryV2, common::ABI_VERSION};

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

impl Default for OutcomeResultV2 {
    fn default() -> Self {
        Self {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            set_cursor: 0,
            cursor: [0; 3],
            place_voxel: 0,
            place_coord: [0; 3],
            remove_voxel: 0,
            remove_coord: [0; 3],
        }
    }
}
