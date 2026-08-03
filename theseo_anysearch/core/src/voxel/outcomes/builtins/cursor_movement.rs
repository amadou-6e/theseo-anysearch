use crate::voxel::outcomes::OutcomeResultV2;

pub fn apply(destination: [i32; 3]) -> OutcomeResultV2 {
    OutcomeResultV2 {
        set_cursor: 1,
        cursor: destination,
        ..OutcomeResultV2::default()
    }
}
