use super::{
    api::{World, WorldError},
    block::{Block, BlockUpdate},
    BoundedRegion, ChunkedWorld, HashMapWorld, InMemoryResidentGuard, StorageCoord,
    WorldAccessError, WorldExtent, WorldMutation, WorldRead, WorldResidency,
};

pub const WORLD_SIZE: u16 = 1000;
pub const WORLD_CENTER: (u16, u16, u16) = (500, 500, 500);
pub type Coord = (u16, u16, u16);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldBackendKind {
    HashMap,
    Chunked,
}

#[derive(Clone, Debug)]
enum WorldBackend {
    HashMap(HashMapWorld),
    Chunked(ChunkedWorld),
}

#[derive(Clone, Debug)]
pub struct WorldState {
    backend: WorldBackend,
}

impl WorldState {
    pub fn new() -> Self {
        Self::new_chunked(32)
    }

    pub fn new_hashmap() -> Self {
        Self {
            backend: WorldBackend::HashMap(HashMapWorld::new(Self::legacy_extent())),
        }
    }

    pub fn new_chunked(chunk_size: u32) -> Self {
        let backend = ChunkedWorld::new(Self::legacy_extent(), WorldExtent::cubic(chunk_size))
            .expect("positive legacy chunk size must fit the address space");
        Self {
            backend: WorldBackend::Chunked(backend),
        }
    }

    const fn legacy_extent() -> WorldExtent {
        WorldExtent::cubic(WORLD_SIZE as u32)
    }

    pub const fn backend_kind(&self) -> WorldBackendKind {
        match self.backend {
            WorldBackend::HashMap(_) => WorldBackendKind::HashMap,
            WorldBackend::Chunked(_) => WorldBackendKind::Chunked,
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
        self.get_block(coord).is_some()
    }

    pub fn is_blocking(&self, coord: Coord) -> bool {
        self.get_block(coord).is_some_and(|block| block.active)
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
        self.read_backend().block_count() as usize
    }

    pub fn iter_filled(&self) -> impl Iterator<Item = Coord> {
        let extent = Self::legacy_extent();
        let region = BoundedRegion::new(
            StorageCoord { x: 0, y: 0, z: 0 },
            StorageCoord {
                x: extent.x,
                y: extent.y,
                z: extent.z,
            },
            extent,
        )
        .expect("legacy world extent is non-empty");
        self.read_backend()
            .blocks_in_region(region)
            .expect("full legacy region is valid")
            .into_iter()
            .map(|(coordinate, _)| {
                (
                    coordinate.x as u16,
                    coordinate.y as u16,
                    coordinate.z as u16,
                )
            })
    }

    pub fn clear(&mut self) {
        self.mutation_backend().clear_blocks();
    }

    pub fn estimated_sparse_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<(Coord, Block)>()
    }

    pub fn dense_world_bytes() -> usize {
        WORLD_SIZE as usize * WORLD_SIZE as usize * WORLD_SIZE as usize
    }

    fn storage(coord: Coord) -> StorageCoord {
        StorageCoord {
            x: u32::from(coord.0),
            y: u32::from(coord.1),
            z: u32::from(coord.2),
        }
    }

    fn read_backend(&self) -> &dyn WorldRead {
        match &self.backend {
            WorldBackend::HashMap(world) => world,
            WorldBackend::Chunked(world) => world,
        }
    }

    fn mutation_backend(&mut self) -> &mut dyn WorldMutation {
        match &mut self.backend {
            WorldBackend::HashMap(world) => world,
            WorldBackend::Chunked(world) => world,
        }
    }
}

impl WorldRead for WorldState {
    fn extent(&self) -> WorldExtent {
        self.read_backend().extent()
    }

    fn get_block_value(&self, coord: StorageCoord) -> Result<Option<Block>, WorldAccessError> {
        self.read_backend().get_block_value(coord)
    }

    fn blocks_in_region(
        &self,
        region: BoundedRegion,
    ) -> Result<Vec<(StorageCoord, Block)>, WorldAccessError> {
        self.read_backend().blocks_in_region(region)
    }

    fn block_count(&self) -> u64 {
        self.read_backend().block_count()
    }
}

impl WorldMutation for WorldState {
    fn set_block_value(
        &mut self,
        coord: StorageCoord,
        block: Block,
    ) -> Result<Option<Block>, WorldAccessError> {
        self.mutation_backend().set_block_value(coord, block)
    }

    fn remove_block_value(
        &mut self,
        coord: StorageCoord,
    ) -> Result<Option<Block>, WorldAccessError> {
        self.mutation_backend().remove_block_value(coord)
    }

    fn update_block_value(
        &mut self,
        coord: StorageCoord,
        update: BlockUpdate,
    ) -> Result<Block, WorldAccessError> {
        self.mutation_backend().update_block_value(coord, update)
    }

    fn clear_blocks(&mut self) {
        self.mutation_backend().clear_blocks();
    }
}

impl WorldResidency for WorldState {
    type Guard<'a> = InMemoryResidentGuard<'a>;

    fn is_region_resident(&self, region: BoundedRegion) -> bool {
        match &self.backend {
            WorldBackend::HashMap(world) => world.is_region_resident(region),
            WorldBackend::Chunked(world) => world.is_region_resident(region),
        }
    }

    fn pin_region(&self, region: BoundedRegion) -> Result<Self::Guard<'_>, WorldAccessError> {
        match &self.backend {
            WorldBackend::HashMap(world) => world.pin_region(region),
            WorldBackend::Chunked(world) => world.pin_region(region),
        }
    }
}

impl World for WorldState {
    fn set_block(&mut self, coord: Coord, block: Block) -> Result<(), WorldError> {
        if !Self::in_bounds(coord) {
            return Err(WorldError::OutOfBounds(coord));
        }
        self.mutation_backend()
            .set_block_value(Self::storage(coord), block)
            .expect("validated legacy coordinate must fit backend");
        Ok(())
    }

    fn remove_block(&mut self, coord: Coord) -> Result<(), WorldError> {
        if !Self::in_bounds(coord) {
            return Err(WorldError::OutOfBounds(coord));
        }
        if self
            .mutation_backend()
            .remove_block_value(Self::storage(coord))
            .expect("validated legacy coordinate must fit backend")
            .is_none()
        {
            return Err(WorldError::NotFound(coord));
        }
        Ok(())
    }

    fn update_block(&mut self, coord: Coord, update: BlockUpdate) -> Result<(), WorldError> {
        if !Self::in_bounds(coord) {
            return Err(WorldError::OutOfBounds(coord));
        }
        self.mutation_backend()
            .update_block_value(Self::storage(coord), update)
            .map_err(|_| WorldError::NotFound(coord))?;
        Ok(())
    }

    fn get_block(&self, coord: Coord) -> Option<Block> {
        if !Self::in_bounds(coord) {
            return None;
        }
        self.read_backend()
            .get_block_value(Self::storage(coord))
            .expect("validated legacy coordinate must fit backend")
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

    fn backends() -> [WorldState; 2] {
        [WorldState::new_hashmap(), WorldState::new_chunked(16)]
    }

    #[test]
    fn mutations_match_across_backends() {
        for mut world in backends() {
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
            assert_eq!(world.len(), 1);
            world.remove_block((1, 2, 3)).unwrap();
            assert_eq!(world.len(), 0);
        }
    }

    #[test]
    fn errors_match_across_backends() {
        for mut world in backends() {
            assert!(matches!(
                world.remove_block((1, 2, 3)),
                Err(WorldError::NotFound(_))
            ));
            assert!(matches!(
                world.set_block((1000, 0, 0), Block::default()),
                Err(WorldError::OutOfBounds(_))
            ));
        }
    }

    #[test]
    fn legacy_helpers_match_across_backends() {
        for mut world in backends() {
            world.toggle((5, 5, 5));
            assert!(world.is_filled((5, 5, 5)));
            assert!(world.is_blocking((5, 5, 5)));
            world.set_block((1, 1, 1), Block::default()).unwrap();
            world.set_block((1, 1, 1), Block::default()).unwrap();
            assert_eq!(world.len(), 2);
            assert_eq!(world.iter_filled().count(), 2);
            assert_eq!(
                world.estimated_sparse_bytes(),
                world.len() * std::mem::size_of::<(Coord, Block)>()
            );
            world.toggle((5, 5, 5));
            assert!(!world.is_filled((5, 5, 5)));
            world.clear();
            assert_eq!(world.len(), 0);
        }
    }

    #[test]
    fn default_world_has_cube() {
        assert_eq!(WorldState::default().len(), 1000);
    }
}
