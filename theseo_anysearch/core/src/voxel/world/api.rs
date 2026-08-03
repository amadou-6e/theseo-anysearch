use super::{Block, BlockUpdate};
use crate::voxel::world::state::Coord;

#[derive(Debug, Clone)]
pub enum WorldError {
    OutOfBounds(Coord),
    NotFound(Coord),
}

pub trait World {
    fn set_block(&mut self, coord: Coord, block: Block) -> Result<(), WorldError>;
    fn remove_block(&mut self, coord: Coord) -> Result<(), WorldError>;
    fn update_block(&mut self, coord: Coord, update: BlockUpdate) -> Result<(), WorldError>;
    fn get_block(&self, coord: Coord) -> Option<&Block>;
}

#[cfg(test)]
mod tests {
    use crate::voxel::world::WorldState;

    #[test]
    fn in_bounds_true() {
        assert!(WorldState::in_bounds((0, 0, 0)));
        assert!(WorldState::in_bounds((999, 999, 999)));
    }

    #[test]
    fn in_bounds_false() {
        assert!(!WorldState::in_bounds((1000, 0, 0)));
        assert!(!WorldState::in_bounds((0, 0, 1000)));
    }
}
