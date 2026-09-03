#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ActionHistoryEntryV2 {
    pub action_index: i32,
    pub previous_cursor: [i32; 3],
    pub cursor: [i32; 3],
    pub feasible: u8,
    pub collision: u8,
}
