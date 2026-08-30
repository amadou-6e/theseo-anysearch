use super::{
    api::{World, WorldError},
    block::{Block, BlockUpdate},
    BoundedRegion, ChunkedWorld, DiskBackedWorld, DiskCacheMetrics, DiskResidentGuard,
    HashMapWorld, InMemoryResidentGuard, PrefetchRequest, StorageCoord, WorldAccessError,
    WorldExtent, WorldMutation, WorldRead, WorldResidency,
};
use std::{collections::HashMap, path::Path, sync::Arc};

pub const WORLD_SIZE: u16 = 1000;
pub const WORLD_CENTER: (u16, u16, u16) = (500, 500, 500);
pub type Coord = (u16, u16, u16);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldBackendKind {
    HashMap,
    Chunked,
    DiskBacked,
}

#[derive(Clone, Debug)]
enum WorldBackend {
    HashMap(HashMapWorld),
    Chunked(ChunkedWorld),
    DiskBacked(DiskBackedWorld),
}

#[derive(Debug)]
pub enum WorldResidentGuard<'a> {
    InMemory(InMemoryResidentGuard<'a>),
    Disk(DiskResidentGuard),
}

/// Cheaply cloneable, immutable ownership of base geometry.
///
/// A handle can be shared by any number of environments. Mutation is only
/// available through a [`WorldState`]'s private episode overlay.
#[derive(Clone, Debug)]
pub struct WorldHandle {
    backend: Arc<WorldBackend>,
}

#[derive(Clone, Debug)]
enum OverlayEntry {
    Block(Block),
    Tombstone,
}

#[derive(Clone, Debug)]
pub struct WorldState {
    base: WorldHandle,
    overlay: HashMap<StorageCoord, OverlayEntry>,
}

impl WorldState {
    pub fn new() -> Self {
        Self::new_chunked(32)
    }

    pub fn new_hashmap() -> Self {
        Self {
            base: WorldHandle::new(WorldBackend::HashMap(HashMapWorld::new(
                Self::legacy_extent(),
            ))),
            overlay: HashMap::new(),
        }
    }

    pub fn new_chunked(chunk_size: u32) -> Self {
        let backend = ChunkedWorld::new(Self::legacy_extent(), WorldExtent::cubic(chunk_size))
            .expect("positive legacy chunk size must fit the address space");
        Self {
            base: WorldHandle::new(WorldBackend::Chunked(backend)),
            overlay: HashMap::new(),
        }
    }

    const fn legacy_extent() -> WorldExtent {
        WorldExtent::cubic(WORLD_SIZE as u32)
    }

    pub fn backend_kind(&self) -> WorldBackendKind {
        match self.base.backend.as_ref() {
            WorldBackend::HashMap(_) => WorldBackendKind::HashMap,
            WorldBackend::Chunked(_) => WorldBackendKind::Chunked,
            WorldBackend::DiskBacked(_) => WorldBackendKind::DiskBacked,
        }
    }

    pub fn from_compiled_pack(
        root: &Path,
        maximum_decoded_bytes: usize,
    ) -> Result<Self, WorldAccessError> {
        Ok(Self {
            base: WorldHandle::new(WorldBackend::DiskBacked(DiskBackedWorld::open(
                root,
                maximum_decoded_bytes,
            )?)),
            overlay: HashMap::new(),
        })
    }

    pub fn prefetch_region(&self, region: BoundedRegion) -> Result<(), WorldAccessError> {
        match self.base.backend.as_ref() {
            WorldBackend::DiskBacked(world) => world.prefetch_region(region),
            _ => Ok(()),
        }
    }

    pub fn disk_cache_metrics(&self) -> Option<DiskCacheMetrics> {
        match self.base.backend.as_ref() {
            WorldBackend::DiskBacked(world) => Some(world.metrics()),
            _ => None,
        }
    }

    pub fn request_prefetch_region(&self, region: BoundedRegion) -> Option<PrefetchRequest> {
        match self.base.backend.as_ref() {
            WorldBackend::DiskBacked(world) => Some(world.request_prefetch(region)),
            _ => None,
        }
    }

    pub fn with_default_cube() -> Self {
        let mut world = Self::new();
        world.replace_base_blocks((495u16..505u16).flat_map(|z| {
            (495u16..505u16)
                .flat_map(move |y| (495u16..505u16).map(move |x| ((x, y, z), Block::default())))
        }));
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
        self.block_count() as usize
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
        self.blocks_in_region(region)
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
        self.overlay.clear();
    }

    pub fn from_handle(base: WorldHandle) -> Self {
        Self {
            base,
            overlay: HashMap::new(),
        }
    }

    pub fn world_handle(&self) -> WorldHandle {
        self.base.clone()
    }

    /// Replace the immutable base and discard episode-local mutations.
    pub fn replace_base(&mut self, base: WorldHandle) {
        self.base = base;
        self.overlay.clear();
    }

    /// Build immutable base geometry once, outside the reset path.
    pub fn replace_base_blocks<I>(&mut self, blocks: I)
    where
        I: IntoIterator<Item = (Coord, Block)>,
    {
        let mut backend = match self.backend_kind() {
            WorldBackendKind::HashMap => {
                WorldBackend::HashMap(HashMapWorld::new(Self::legacy_extent()))
            }
            WorldBackendKind::Chunked => {
                let chunk_shape = match self.base.backend.as_ref() {
                    WorldBackend::Chunked(world) => world.chunk_shape(),
                    WorldBackend::HashMap(_) => unreachable!("backend kind was checked"),
                    WorldBackend::DiskBacked(_) => unreachable!("backend kind was checked"),
                };
                WorldBackend::Chunked(
                    ChunkedWorld::new(Self::legacy_extent(), chunk_shape)
                        .expect("positive legacy chunk size must fit the address space"),
                )
            }
            WorldBackendKind::DiskBacked => WorldBackend::Chunked(
                ChunkedWorld::new(self.extent(), WorldExtent::cubic(32))
                    .expect("compiled extent and chunk shape must be valid"),
            ),
        };
        for (coord, block) in blocks {
            if Self::in_bounds(coord) {
                backend
                    .mutation()
                    .set_block_value(Self::storage(coord), block)
                    .expect("validated legacy coordinate must fit backend");
            }
        }
        self.replace_base(WorldHandle::new(backend));
    }

    pub fn estimated_sparse_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<(Coord, Block)>()
    }

    /// Approximate memory owned by episode-local block overrides and tombstones.
    pub fn estimated_overlay_bytes(&self) -> usize {
        self.overlay.len() * std::mem::size_of::<(StorageCoord, OverlayEntry)>()
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
        self.base.read_backend()
    }
}

impl WorldBackend {
    fn read(&self) -> &dyn WorldRead {
        match self {
            Self::HashMap(world) => world,
            Self::Chunked(world) => world,
            Self::DiskBacked(world) => world,
        }
    }

    fn mutation(&mut self) -> &mut dyn WorldMutation {
        match self {
            Self::HashMap(world) => world,
            Self::Chunked(world) => world,
            Self::DiskBacked(_) => panic!("immutable disk backend cannot be mutated directly"),
        }
    }
}

impl WorldHandle {
    fn new(backend: WorldBackend) -> Self {
        Self {
            backend: Arc::new(backend),
        }
    }

    fn read_backend(&self) -> &dyn WorldRead {
        self.backend.read()
    }
}

impl WorldRead for WorldHandle {
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

impl WorldResidency for WorldHandle {
    type Guard<'a> = WorldResidentGuard<'a>;

    fn is_region_resident(&self, region: BoundedRegion) -> bool {
        match self.backend.as_ref() {
            WorldBackend::HashMap(world) => world.is_region_resident(region),
            WorldBackend::Chunked(world) => world.is_region_resident(region),
            WorldBackend::DiskBacked(world) => world.is_region_resident(region),
        }
    }

    fn pin_region(&self, region: BoundedRegion) -> Result<Self::Guard<'_>, WorldAccessError> {
        match self.backend.as_ref() {
            WorldBackend::HashMap(world) => {
                world.pin_region(region).map(WorldResidentGuard::InMemory)
            }
            WorldBackend::Chunked(world) => {
                world.pin_region(region).map(WorldResidentGuard::InMemory)
            }
            WorldBackend::DiskBacked(world) => {
                world.pin_region(region).map(WorldResidentGuard::Disk)
            }
        }
    }
}

impl WorldRead for WorldState {
    fn extent(&self) -> WorldExtent {
        self.read_backend().extent()
    }

    fn get_block_value(&self, coord: StorageCoord) -> Result<Option<Block>, WorldAccessError> {
        if !self.extent().contains_storage(coord) {
            return Err(WorldAccessError::OutOfBounds(coord));
        }
        match self.overlay.get(&coord) {
            Some(OverlayEntry::Block(block)) => Ok(Some(block.clone())),
            Some(OverlayEntry::Tombstone) => Ok(None),
            None => self.read_backend().get_block_value(coord),
        }
    }

    fn blocks_in_region(
        &self,
        region: BoundedRegion,
    ) -> Result<Vec<(StorageCoord, Block)>, WorldAccessError> {
        let mut resolved = self
            .read_backend()
            .blocks_in_region(region)?
            .into_iter()
            .collect::<HashMap<_, _>>();
        for (coord, entry) in self
            .overlay
            .iter()
            .filter(|(coord, _)| region.contains(**coord))
        {
            match entry {
                OverlayEntry::Block(block) => {
                    resolved.insert(*coord, block.clone());
                }
                OverlayEntry::Tombstone => {
                    resolved.remove(coord);
                }
            }
        }
        let mut blocks = resolved.into_iter().collect::<Vec<_>>();
        blocks.sort_by_key(|(coord, _)| coord.global_key());
        Ok(blocks)
    }

    fn block_count(&self) -> u64 {
        let mut count = self.read_backend().block_count();
        for (coord, entry) in &self.overlay {
            let in_base = self
                .read_backend()
                .get_block_value(*coord)
                .expect("overlay coordinates are validated")
                .is_some();
            match (in_base, entry) {
                (false, OverlayEntry::Block(_)) => count += 1,
                (true, OverlayEntry::Tombstone) => count -= 1,
                _ => {}
            }
        }
        count
    }
}

impl WorldMutation for WorldState {
    fn set_block_value(
        &mut self,
        coord: StorageCoord,
        block: Block,
    ) -> Result<Option<Block>, WorldAccessError> {
        if !self.extent().contains_storage(coord) {
            return Err(WorldAccessError::OutOfBounds(coord));
        }
        let previous = self.get_block_value(coord)?;
        self.overlay.insert(coord, OverlayEntry::Block(block));
        Ok(previous)
    }

    fn remove_block_value(
        &mut self,
        coord: StorageCoord,
    ) -> Result<Option<Block>, WorldAccessError> {
        if !self.extent().contains_storage(coord) {
            return Err(WorldAccessError::OutOfBounds(coord));
        }
        let previous = self.get_block_value(coord)?;
        if previous.is_none() {
            return Ok(None);
        }
        if self.read_backend().get_block_value(coord)?.is_some() {
            self.overlay.insert(coord, OverlayEntry::Tombstone);
        } else {
            self.overlay.remove(&coord);
        }
        Ok(previous)
    }

    fn update_block_value(
        &mut self,
        coord: StorageCoord,
        update: BlockUpdate,
    ) -> Result<Block, WorldAccessError> {
        let mut block = self
            .get_block_value(coord)?
            .ok_or(WorldAccessError::NotFound(coord))?;
        update.apply_to(&mut block);
        self.overlay
            .insert(coord, OverlayEntry::Block(block.clone()));
        Ok(block)
    }

    fn clear_blocks(&mut self) {
        self.overlay.clear();
    }
}

impl WorldResidency for WorldState {
    type Guard<'a> = WorldResidentGuard<'a>;

    fn is_region_resident(&self, region: BoundedRegion) -> bool {
        self.base.is_region_resident(region)
    }

    fn pin_region(&self, region: BoundedRegion) -> Result<Self::Guard<'_>, WorldAccessError> {
        self.base.pin_region(region)
    }
}

impl World for WorldState {
    fn set_block(&mut self, coord: Coord, block: Block) -> Result<(), WorldError> {
        if !Self::in_bounds(coord) {
            return Err(WorldError::OutOfBounds(coord));
        }
        self.set_block_value(Self::storage(coord), block)
            .expect("validated legacy coordinate must fit backend");
        Ok(())
    }

    fn remove_block(&mut self, coord: Coord) -> Result<(), WorldError> {
        if !Self::in_bounds(coord) {
            return Err(WorldError::OutOfBounds(coord));
        }
        if self
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
        self.update_block_value(Self::storage(coord), update)
            .map_err(|_| WorldError::NotFound(coord))?;
        Ok(())
    }

    fn get_block(&self, coord: Coord) -> Option<Block> {
        if !Self::in_bounds(coord) {
            return None;
        }
        self.get_block_value(Self::storage(coord))
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
    use crate::voxel::world::{Block, BlockUpdate, GridRay, World, WorldError, WorldRead};

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

    fn base_world() -> WorldState {
        let mut world = WorldState::new();
        world.replace_base_blocks([
            ((1, 1, 1), Block::default()),
            (
                (2, 2, 2),
                Block {
                    kind: 2,
                    ..Block::default()
                },
            ),
        ]);
        world
    }

    #[test]
    fn shared_base_has_isolated_overlays() {
        let source = base_world();
        let handle = source.world_handle();
        let mut first = WorldState::from_handle(handle.clone());
        let second = WorldState::from_handle(handle);

        first.set_block((3, 3, 3), Block::default()).unwrap();
        first.remove_block((1, 1, 1)).unwrap();

        assert!(second.get_block((1, 1, 1)).is_some());
        assert!(second.get_block((3, 3, 3)).is_none());
    }

    #[test]
    fn reset_clears_overlay_without_replacing_base() {
        let mut world = base_world();
        let original_base = world.base.backend.clone();
        world.set_block((3, 3, 3), Block::default()).unwrap();
        world.remove_block((1, 1, 1)).unwrap();

        world.clear();

        assert!(Arc::ptr_eq(&original_base, &world.base.backend));
        assert!(world.overlay.is_empty());
        assert!(world.get_block((1, 1, 1)).is_some());
        assert!(world.get_block((3, 3, 3)).is_none());
    }

    #[test]
    fn insert_update_tombstone_and_reinsert_resolve_correctly() {
        let mut world = base_world();
        world
            .set_block(
                (1, 1, 1),
                Block {
                    kind: 7,
                    ..Block::default()
                },
            )
            .unwrap();
        assert_eq!(world.get_block((1, 1, 1)).unwrap().kind, 7);
        assert_eq!(world.len(), 2);

        world.remove_block((1, 1, 1)).unwrap();
        assert!(world.get_block((1, 1, 1)).is_none());
        assert_eq!(world.len(), 1);

        world
            .set_block(
                (1, 1, 1),
                Block {
                    kind: 9,
                    ..Block::default()
                },
            )
            .unwrap();
        world
            .update_block(
                (1, 1, 1),
                BlockUpdate {
                    kind: Some(10),
                    ..BlockUpdate::default()
                },
            )
            .unwrap();
        assert_eq!(world.get_block((1, 1, 1)).unwrap().kind, 10);
        assert_eq!(world.len(), 2);

        world.set_block((4, 4, 4), Block::default()).unwrap();
        assert_eq!(world.len(), 3);
        world.remove_block((4, 4, 4)).unwrap();
        assert_eq!(world.len(), 2);
        assert!(!world
            .overlay
            .contains_key(&StorageCoord { x: 4, y: 4, z: 4 }));
    }

    #[test]
    fn regional_queries_merge_without_duplicates() {
        let mut world = base_world();
        world
            .set_block(
                (1, 1, 1),
                Block {
                    kind: 7,
                    ..Block::default()
                },
            )
            .unwrap();
        world.remove_block((2, 2, 2)).unwrap();
        world.set_block((3, 3, 3), Block::default()).unwrap();
        let region = BoundedRegion::new(
            StorageCoord { x: 0, y: 0, z: 0 },
            StorageCoord { x: 5, y: 5, z: 5 },
            WorldState::legacy_extent(),
        )
        .unwrap();

        let blocks = world.blocks_in_region(region).unwrap();
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].0, StorageCoord { x: 1, y: 1, z: 1 });
        assert_eq!(blocks[0].1.kind, 7);
        assert_eq!(blocks[1].0, StorageCoord { x: 3, y: 3, z: 3 });
    }

    #[test]
    fn raycast_observes_overlay_first_view() {
        let mut world = base_world();
        world.remove_block((1, 1, 1)).unwrap();
        world.set_block((3, 3, 3), Block::default()).unwrap();
        let ray = GridRay::new(StorageCoord { x: 0, y: 0, z: 0 }, [1, 1, 1], 5).unwrap();

        let hit = world.raycast(ray).unwrap().unwrap();
        assert_eq!(hit.coordinate, StorageCoord { x: 2, y: 2, z: 2 });
        world.remove_block((2, 2, 2)).unwrap();
        let hit = world.raycast(ray).unwrap().unwrap();
        assert_eq!(hit.coordinate, StorageCoord { x: 3, y: 3, z: 3 });
    }
}
