use std::collections::HashMap;

use super::{
    regional::{
        validate_coordinate, BoundedRegion, InMemoryResidentGuard, WorldAccessError, WorldMutation,
        WorldRead, WorldResidency,
    },
    Block, BlockUpdate, StorageCoord, WorldExtent,
};

type ChunkKey = (u32, u32, u32);
type LocalKey = (u32, u32, u32);

#[derive(Clone, Debug, Default)]
struct RegularChunk {
    blocks: HashMap<LocalKey, Block>,
}

#[derive(Clone, Debug)]
pub struct ChunkedWorld {
    extent: WorldExtent,
    chunk_shape: WorldExtent,
    chunks: HashMap<ChunkKey, RegularChunk>,
    block_count: u64,
}

impl ChunkedWorld {
    pub fn new(extent: WorldExtent, chunk_shape: WorldExtent) -> Result<Self, WorldAccessError> {
        let count = chunk_shape
            .voxel_count()
            .ok_or(WorldAccessError::ChunkShapeExceedsAddressSpace(chunk_shape))?;
        if count == 0 {
            return Err(WorldAccessError::ChunkShapeExceedsAddressSpace(chunk_shape));
        }
        Ok(Self {
            extent,
            chunk_shape,
            chunks: HashMap::new(),
            block_count: 0,
        })
    }

    fn keys(&self, coord: StorageCoord) -> (ChunkKey, LocalKey) {
        (
            (
                coord.x / self.chunk_shape.x,
                coord.y / self.chunk_shape.y,
                coord.z / self.chunk_shape.z,
            ),
            (
                coord.x % self.chunk_shape.x,
                coord.y % self.chunk_shape.y,
                coord.z % self.chunk_shape.z,
            ),
        )
    }

    pub fn resident_chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

impl WorldRead for ChunkedWorld {
    fn extent(&self) -> WorldExtent {
        self.extent
    }

    fn get_block_value(&self, coord: StorageCoord) -> Result<Option<Block>, WorldAccessError> {
        validate_coordinate(self.extent, coord)?;
        let (chunk, local) = self.keys(coord);
        Ok(self
            .chunks
            .get(&chunk)
            .and_then(|chunk| chunk.blocks.get(&local))
            .cloned())
    }

    fn blocks_in_region(
        &self,
        region: BoundedRegion,
    ) -> Result<Vec<(StorageCoord, Block)>, WorldAccessError> {
        BoundedRegion::new(region.minimum, region.maximum_exclusive, self.extent)?;
        let minimum_chunk = self.keys(region.minimum).0;
        let maximum = StorageCoord {
            x: region.maximum_exclusive.x - 1,
            y: region.maximum_exclusive.y - 1,
            z: region.maximum_exclusive.z - 1,
        };
        let maximum_chunk = self.keys(maximum).0;
        let mut blocks = Vec::new();
        for chunk_z in minimum_chunk.2..=maximum_chunk.2 {
            for chunk_y in minimum_chunk.1..=maximum_chunk.1 {
                for chunk_x in minimum_chunk.0..=maximum_chunk.0 {
                    let chunk_key = (chunk_x, chunk_y, chunk_z);
                    let Some(chunk) = self.chunks.get(&chunk_key) else {
                        continue;
                    };
                    for (&(local_x, local_y, local_z), block) in &chunk.blocks {
                        let coordinate = StorageCoord {
                            x: chunk_x * self.chunk_shape.x + local_x,
                            y: chunk_y * self.chunk_shape.y + local_y,
                            z: chunk_z * self.chunk_shape.z + local_z,
                        };
                        if region.contains(coordinate) {
                            blocks.push((coordinate, block.clone()));
                        }
                    }
                }
            }
        }
        blocks.sort_by_key(|(coordinate, _)| coordinate.global_key());
        Ok(blocks)
    }

    fn block_count(&self) -> u64 {
        self.block_count
    }
}

impl WorldMutation for ChunkedWorld {
    fn set_block_value(
        &mut self,
        coord: StorageCoord,
        block: Block,
    ) -> Result<Option<Block>, WorldAccessError> {
        validate_coordinate(self.extent, coord)?;
        let (chunk_key, local_key) = self.keys(coord);
        let previous = self
            .chunks
            .entry(chunk_key)
            .or_default()
            .blocks
            .insert(local_key, block);
        if previous.is_none() {
            self.block_count += 1;
        }
        Ok(previous)
    }

    fn remove_block_value(
        &mut self,
        coord: StorageCoord,
    ) -> Result<Option<Block>, WorldAccessError> {
        validate_coordinate(self.extent, coord)?;
        let (chunk_key, local_key) = self.keys(coord);
        let Some(chunk) = self.chunks.get_mut(&chunk_key) else {
            return Ok(None);
        };
        let removed = chunk.blocks.remove(&local_key);
        let empty = chunk.blocks.is_empty();
        if removed.is_some() {
            self.block_count -= 1;
        }
        if empty {
            self.chunks.remove(&chunk_key);
        }
        Ok(removed)
    }

    fn update_block_value(
        &mut self,
        coord: StorageCoord,
        update: BlockUpdate,
    ) -> Result<Block, WorldAccessError> {
        validate_coordinate(self.extent, coord)?;
        let (chunk_key, local_key) = self.keys(coord);
        let block = self
            .chunks
            .get_mut(&chunk_key)
            .and_then(|chunk| chunk.blocks.get_mut(&local_key))
            .ok_or(WorldAccessError::NotFound(coord))?;
        update.apply_to(block);
        Ok(block.clone())
    }

    fn clear_blocks(&mut self) {
        self.chunks.clear();
        self.block_count = 0;
    }
}

impl WorldResidency for ChunkedWorld {
    type Guard<'a> = InMemoryResidentGuard<'a>;

    fn is_region_resident(&self, region: BoundedRegion) -> bool {
        BoundedRegion::new(region.minimum, region.maximum_exclusive, self.extent).is_ok()
    }

    fn pin_region(&self, region: BoundedRegion) -> Result<Self::Guard<'_>, WorldAccessError> {
        let region = BoundedRegion::new(region.minimum, region.maximum_exclusive, self.extent)?;
        Ok(InMemoryResidentGuard::new(region))
    }
}
