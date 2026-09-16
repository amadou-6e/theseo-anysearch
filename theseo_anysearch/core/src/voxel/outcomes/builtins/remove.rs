use crate::voxel::outcomes::OutcomeResultV2;

pub fn apply(previous_cursor: [i32; 3]) -> OutcomeResultV2 {
    OutcomeResultV2 {
        remove_voxel: 1,
        remove_coord: previous_cursor,
        ..OutcomeResultV2::default()
    }
}
