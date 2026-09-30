/// Fixed lexicographic ordering for the 26 neighboring voxel movements.
///
/// Single- and multi-agent environments share this table because the action
/// index is part of the public environment contract.
pub(crate) const OFFSETS_26: [(i32, i32, i32); 26] = {
    let mut offsets = [(0, 0, 0); 26];
    let mut index = 0;
    let mut dx = -1;
    while dx <= 1 {
        let mut dy = -1;
        while dy <= 1 {
            let mut dz = -1;
            while dz <= 1 {
                if dx != 0 || dy != 0 || dz != 0 {
                    offsets[index] = (dx, dy, dz);
                    index += 1;
                }
                dz += 1;
            }
            dy += 1;
        }
        dx += 1;
    }
    offsets
};

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::OFFSETS_26;

    #[test]
    fn offsets_are_unique_lexicographic_neighbors() {
        assert_eq!(OFFSETS_26.first(), Some(&(-1, -1, -1)));
        assert_eq!(OFFSETS_26.last(), Some(&(1, 1, 1)));
        assert!(!OFFSETS_26.contains(&(0, 0, 0)));
        assert_eq!(OFFSETS_26.iter().copied().collect::<HashSet<_>>().len(), 26);
    }
}
