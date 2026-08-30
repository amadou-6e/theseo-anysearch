use std::collections::HashMap;

use super::{
    regional::{
        validate_coordinate, BoundedRegion, InMemoryResidentGuard, WorldAccessError, WorldMutation,
        WorldRead, WorldResidency,
    },
    Block, BlockUpdate, StorageCoord, WorldExtent,
};

#[derive(Clone, Debug)]
pub struct HashMapWorld {
    extent: WorldExtent,
    blocks: HashMap<(u32, u32, u32), Block>,
}

impl HashMapWorld {
    pub fn new(extent: WorldExtent) -> Self {
        Self {
            extent,
            blocks: HashMap::new(),
        }
    }
}

impl WorldRead for HashMapWorld {
    fn extent(&self) -> WorldExtent {
        self.extent
    }

    fn get_block_value(&self, coord: StorageCoord) -> Result<Option<Block>, WorldAccessError> {
        validate_coordinate(self.extent, coord)?;
        Ok(self.blocks.get(&coord.global_key()).cloned())
    }

    fn blocks_in_region(
        &self,
        region: BoundedRegion,
    ) -> Result<Vec<(StorageCoord, Block)>, WorldAccessError> {
        BoundedRegion::new(region.minimum, region.maximum_exclusive, self.extent)?;
        let mut blocks = self
            .blocks
            .iter()
            .filter_map(|(&(x, y, z), block)| {
                let coordinate = StorageCoord { x, y, z };
                region
                    .contains(coordinate)
                    .then(|| (coordinate, block.clone()))
            })
            .collect::<Vec<_>>();
        blocks.sort_by_key(|(coordinate, _)| coordinate.global_key());
        Ok(blocks)
    }

    fn block_count(&self) -> u64 {
        self.blocks.len() as u64
    }
}

impl WorldMutation for HashMapWorld {
    fn set_block_value(
        &mut self,
        coord: StorageCoord,
        block: Block,
    ) -> Result<Option<Block>, WorldAccessError> {
        validate_coordinate(self.extent, coord)?;
        Ok(self.blocks.insert(coord.global_key(), block))
    }

    fn remove_block_value(
        &mut self,
        coord: StorageCoord,
    ) -> Result<Option<Block>, WorldAccessError> {
        validate_coordinate(self.extent, coord)?;
        Ok(self.blocks.remove(&coord.global_key()))
    }

    fn update_block_value(
        &mut self,
        coord: StorageCoord,
        update: BlockUpdate,
    ) -> Result<Block, WorldAccessError> {
        validate_coordinate(self.extent, coord)?;
        let block = self
            .blocks
            .get_mut(&coord.global_key())
            .ok_or(WorldAccessError::NotFound(coord))?;
        update.apply_to(block);
        Ok(block.clone())
    }

    fn clear_blocks(&mut self) {
        self.blocks.clear();
    }
}

impl WorldResidency for HashMapWorld {
    type Guard<'a> = InMemoryResidentGuard<'a>;

    fn is_region_resident(&self, region: BoundedRegion) -> bool {
        BoundedRegion::new(region.minimum, region.maximum_exclusive, self.extent).is_ok()
    }

    fn pin_region(&self, region: BoundedRegion) -> Result<Self::Guard<'_>, WorldAccessError> {
        let region = BoundedRegion::new(region.minimum, region.maximum_exclusive, self.extent)?;
        Ok(InMemoryResidentGuard::new(region))
    }
}
