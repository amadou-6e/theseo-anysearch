use crate::voxel::outcomes::OutcomeResultV2;

pub fn apply(action_index: i32, destination: [i32; 3]) -> OutcomeResultV2 {
    OutcomeResultV2 {
        place_voxel: u8::from(action_index != 26),
        place_coord: destination,
        ..OutcomeResultV2::default()
    }
}
