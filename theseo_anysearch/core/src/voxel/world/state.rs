use std::collections::HashMap;

use super::{
    api::{World, WorldError},
    block::{Block, BlockUpdate},
};

pub const WORLD_SIZE: u16 = 1000;
pub const WORLD_CENTER: (u16, u16, u16) = (500, 500, 500);
pub type Coord = (u16, u16, u16);

#[derive(Clone, Debug)]
pub struct WorldState {
    filled: HashMap<Coord, Block>,
}

impl WorldState {
    pub fn new() -> Self {
        Self {
            filled: HashMap::new(),
        }
    }

    pub fn with_default_cube() -> Self {
        let mut world = Self::new();
        for z in 495u16..505u16 {
            for y in 495u16..505u16 {
                for x in 495u16..505u16 {
                    let _ = world.set_block((x, y, z), Block::default());
                }
            }
        }
        world
    }

    pub fn index(x: usize, y: usize, z: usize) -> usize {
        x + y * WORLD_SIZE as usize + z * WORLD_SIZE as usize * WORLD_SIZE as usize
    }

    pub fn in_bounds(coord: Coord) -> bool {
        coord.0 < WORLD_SIZE && coord.1 < WORLD_SIZE && coord.2 < WORLD_SIZE
    }

    pub fn is_filled(&self, coord: Coord) -> bool {
        self.filled.contains_key(&coord)
    }

    pub fn is_blocking(&self, coord: Coord) -> bool {
        self.filled.get(&coord).is_some_and(|block| block.active)
    }

    pub fn set(&mut self, coord: Coord, filled: bool) {
        if filled {
            let _ = self.set_block(coord, Block::default());
        } else {
            let _ = self.remove_block(coord);
        }
    }

    pub fn toggle(&mut self, coord: Coord) {
        if self.is_filled(coord) {
            let _ = self.remove_block(coord);
        } else {
            let _ = self.set_block(coord, Block::default());
        }
    }

    pub fn len(&self) -> usize {
        self.filled.len()
    }

    pub fn iter_filled(&self) -> impl Iterator<Item = &Coord> {
        self.filled.keys()
    }

    pub fn clear(&mut self) {
        self.filled.clear();
    }

    pub fn estimated_sparse_bytes(&self) -> usize {
        self.filled.len() * std::mem::size_of::<(Coord, Block)>()
    }

    pub fn dense_world_bytes() -> usize {
        WORLD_SIZE as usize * WORLD_SIZE as usize * WORLD_SIZE as usize
    }
}

impl World for WorldState {
    fn set_block(&mut self, coord: Coord, block: Block) -> Result<(), WorldError> {
        if !Self::in_bounds(coord) {
            return Err(WorldError::OutOfBounds(coord));
        }
        self.filled.insert(coord, block);
        Ok(())
    }

    fn remove_block(&mut self, coord: Coord) -> Result<(), WorldError> {
        if !Self::in_bounds(coord) {
            return Err(WorldError::OutOfBounds(coord));
        }
        if self.filled.remove(&coord).is_none() {
            return Err(WorldError::NotFound(coord));
        }
        Ok(())
    }

    fn update_block(&mut self, coord: Coord, update: BlockUpdate) -> Result<(), WorldError> {
        if !Self::in_bounds(coord) {
            return Err(WorldError::OutOfBounds(coord));
        }
        let block = self
            .filled
            .get_mut(&coord)
            .ok_or(WorldError::NotFound(coord))?;
        update.apply_to(block);
        Ok(())
    }

    fn get_block(&self, coord: Coord) -> Option<&Block> {
        self.filled.get(&coord)
    }
}

impl Default for WorldState {
    fn default() -> Self {
        Self::with_default_cube()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::world::{Block, BlockUpdate, World, WorldError};

    #[test]
    fn set_block_inserts() {
        let mut world = WorldState::new();
        world.set_block((1, 2, 3), Block::default()).unwrap();
        assert!(world.is_filled((1, 2, 3)));
    }

    #[test]
    fn set_block_replaces() {
        let mut world = WorldState::new();
        world.set_block((1, 2, 3), Block::default()).unwrap();
        world.set_block((1, 2, 3), Block::default()).unwrap();
        assert_eq!(world.len(), 1);
    }

    #[test]
    fn remove_block_clears() {
        let mut world = WorldState::new();
        world.set_block((1, 2, 3), Block::default()).unwrap();
        world.remove_block((1, 2, 3)).unwrap();
        assert!(!world.is_filled((1, 2, 3)));
    }

    #[test]
    fn remove_block_not_found() {
        let mut world = WorldState::new();
        let err = world.remove_block((1, 2, 3)).unwrap_err();
        assert!(matches!(err, WorldError::NotFound(_)));
    }

    #[test]
    fn update_block_mutates() {
        let mut world = WorldState::new();
        world.set_block((1, 2, 3), Block::default()).unwrap();
        world
            .update_block(
                (1, 2, 3),
                BlockUpdate {
                    kind: Some(7),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(world.get_block((1, 2, 3)).unwrap().kind, 7);
    }

    #[test]
    fn update_block_not_found() {
        let mut world = WorldState::new();
        let err = world
            .update_block((1, 2, 3), BlockUpdate::default())
            .unwrap_err();
        assert!(matches!(err, WorldError::NotFound(_)));
    }

    #[test]
    fn out_of_bounds_set() {
        let mut world = WorldState::new();
        let err = world.set_block((1000, 0, 0), Block::default()).unwrap_err();
        assert!(matches!(err, WorldError::OutOfBounds(_)));
    }

    #[test]
    fn out_of_bounds_remove() {
        let mut world = WorldState::new();
        let err = world.remove_block((0, 1000, 0)).unwrap_err();
        assert!(matches!(err, WorldError::OutOfBounds(_)));
    }

    #[test]
    fn toggle_fill() {
        let mut world = WorldState::new();
        world.toggle((5, 5, 5));
        assert!(world.is_filled((5, 5, 5)));
        world.toggle((5, 5, 5));
        assert!(!world.is_filled((5, 5, 5)));
    }

    #[test]
    fn iter_filled_count() {
        let mut world = WorldState::new();
        world.set_block((1, 1, 1), Block::default()).unwrap();
        world.set_block((2, 2, 2), Block::default()).unwrap();
        assert_eq!(world.iter_filled().count(), world.len());
    }

    #[test]
    fn estimated_sparse_bytes() {
        let mut world = WorldState::new();
        world.set_block((1, 1, 1), Block::default()).unwrap();
        assert_eq!(
            world.estimated_sparse_bytes(),
            world.len() * std::mem::size_of::<(Coord, Block)>()
        );
    }

    #[test]
    fn default_world_has_cube() {
        assert_eq!(WorldState::default().len(), 1000);
    }

    #[test]
    fn empty_world_is_empty() {
        assert_eq!(WorldState::new().len(), 0);
    }
}
