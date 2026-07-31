use std::ffi::c_char;

#[repr(C)]
pub struct RewardContextV1 {
    pub abi_version: u32, pub struct_size: u32, pub step: u64, pub action_index: i32,
    pub previous_cursor: [i32; 3], pub cursor: [i32; 3], pub goal: [i32; 3],
    pub has_goal: u8, pub invalid_action: u8, pub collision: u8,
    pub goal_reached: u8, pub terminated: u8, pub truncated: u8,
    pub consecutive_collisions: u32,
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

pub fn native_collision(context: &RewardContextV1, result: &mut RewardResultV1) -> i32 {
    let penalty = if context.collision != 0 { -0.02 } else { 0.0 };
    result.mode = 0;
    result.reward = penalty;
    result.component_count = 1;
    let name = b"native_collision\0";
    for (target, source) in result.components[0].name.iter_mut().zip(name.iter()) {
        *target = *source as c_char;
    }
    result.components[0].value = penalty;
    0
}
