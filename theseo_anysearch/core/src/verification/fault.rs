use crate::voxel::world::{
    Block, BlockUpdate, BoundedRegion, GridRay, RayHit, StorageCoord, WorldAccessError,
    WorldMutation, WorldRead,
};
use std::cell::Cell;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FaultOperation {
    Point,
    Region,
    Ray,
    Set,
    Remove,
    Update,
}

#[derive(Clone, Copy, Debug)]
pub struct FaultRule {
    pub operation: FaultOperation,
    pub fail_on_call: usize,
    pub error: WorldAccessError,
}

#[derive(Clone, Debug)]
pub struct FaultInjectedWorld<W> {
    inner: W,
    rules: Vec<FaultRule>,
    calls: [Cell<usize>; 6],
}

impl<W> FaultInjectedWorld<W> {
    pub fn new(inner: W, rules: Vec<FaultRule>) -> Self {
        Self {
            inner,
            rules,
            calls: std::array::from_fn(|_| Cell::new(0)),
        }
    }

    pub fn into_inner(self) -> W {
        self.inner
    }

    fn index(operation: FaultOperation) -> usize {
        match operation {
            FaultOperation::Point => 0,
            FaultOperation::Region => 1,
            FaultOperation::Ray => 2,
            FaultOperation::Set => 3,
            FaultOperation::Remove => 4,
            FaultOperation::Update => 5,
        }
    }

    fn failure(&self, operation: FaultOperation) -> Option<WorldAccessError> {
        let index = Self::index(operation);
        let call = self.calls[index].get() + 1;
        self.calls[index].set(call);
        self.rules
            .iter()
            .find(|rule| rule.operation == operation && rule.fail_on_call == call)
            .map(|rule| rule.error)
    }
}

impl<W: WorldRead> WorldRead for FaultInjectedWorld<W> {
    fn extent(&self) -> crate::voxel::world::WorldExtent {
        self.inner.extent()
    }

    fn get_block_value(&self, coord: StorageCoord) -> Result<Option<Block>, WorldAccessError> {
        if let Some(error) = self.failure(FaultOperation::Point) {
            return Err(error);
        }
        self.inner.get_block_value(coord)
    }

    fn blocks_in_region(
        &self,
        region: BoundedRegion,
    ) -> Result<Vec<(StorageCoord, Block)>, WorldAccessError> {
        if let Some(error) = self.failure(FaultOperation::Region) {
            return Err(error);
        }
        self.inner.blocks_in_region(region)
    }

    fn block_count(&self) -> u64 {
        self.inner.block_count()
    }

    fn raycast(&self, ray: GridRay) -> Result<Option<RayHit>, WorldAccessError> {
        if let Some(error) = self.failure(FaultOperation::Ray) {
            return Err(error);
        }
        self.inner.raycast(ray)
    }
}

impl<W: WorldRead + WorldMutation> WorldMutation for FaultInjectedWorld<W> {
    fn set_block_value(
        &mut self,
        coord: StorageCoord,
        block: Block,
    ) -> Result<Option<Block>, WorldAccessError> {
        if let Some(error) = self.failure(FaultOperation::Set) {
            return Err(error);
        }
        self.inner.set_block_value(coord, block)
    }

    fn remove_block_value(
        &mut self,
        coord: StorageCoord,
    ) -> Result<Option<Block>, WorldAccessError> {
        if let Some(error) = self.failure(FaultOperation::Remove) {
            return Err(error);
        }
        self.inner.remove_block_value(coord)
    }

    fn update_block_value(
        &mut self,
        coord: StorageCoord,
        update: BlockUpdate,
    ) -> Result<Block, WorldAccessError> {
        if let Some(error) = self.failure(FaultOperation::Update) {
            return Err(error);
        }
        self.inner.update_block_value(coord, update)
    }

    fn clear_blocks(&mut self) {
        self.inner.clear_blocks();
    }
}
