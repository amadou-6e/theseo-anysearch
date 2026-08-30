use std::marker::PhantomData;

use super::{Block, BlockUpdate, StorageCoord, WorldExtent};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BoundedRegion {
    pub minimum: StorageCoord,
    pub maximum_exclusive: StorageCoord,
}

impl BoundedRegion {
    pub fn new(
        minimum: StorageCoord,
        maximum_exclusive: StorageCoord,
        extent: WorldExtent,
    ) -> Result<Self, WorldAccessError> {
        let valid = minimum.x < maximum_exclusive.x
            && minimum.y < maximum_exclusive.y
            && minimum.z < maximum_exclusive.z
            && maximum_exclusive.x <= extent.x
            && maximum_exclusive.y <= extent.y
            && maximum_exclusive.z <= extent.z;
        if !valid {
            return Err(WorldAccessError::InvalidRegion {
                minimum,
                maximum_exclusive,
            });
        }
        Ok(Self {
            minimum,
            maximum_exclusive,
        })
    }

    pub const fn contains(self, coord: StorageCoord) -> bool {
        coord.x >= self.minimum.x
            && coord.y >= self.minimum.y
            && coord.z >= self.minimum.z
            && coord.x < self.maximum_exclusive.x
            && coord.y < self.maximum_exclusive.y
            && coord.z < self.maximum_exclusive.z
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GridRay {
    pub origin: StorageCoord,
    pub step: [i8; 3],
    pub max_steps: u32,
}

impl GridRay {
    pub fn new(
        origin: StorageCoord,
        step: [i8; 3],
        max_steps: u32,
    ) -> Result<Self, WorldAccessError> {
        if step == [0, 0, 0] || step.iter().any(|value| !(-1..=1).contains(value)) {
            return Err(WorldAccessError::InvalidRayStep(step));
        }
        Ok(Self {
            origin,
            step,
            max_steps,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RayHit {
    pub coordinate: StorageCoord,
    pub block: Block,
    pub steps: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldAccessError {
    OutOfBounds(StorageCoord),
    NotFound(StorageCoord),
    InvalidRegion {
        minimum: StorageCoord,
        maximum_exclusive: StorageCoord,
    },
    InvalidRayStep([i8; 3]),
    ChunkShapeExceedsAddressSpace(WorldExtent),
}

pub trait WorldRead {
    fn extent(&self) -> WorldExtent;
    fn get_block_value(&self, coord: StorageCoord) -> Result<Option<Block>, WorldAccessError>;
    fn blocks_in_region(
        &self,
        region: BoundedRegion,
    ) -> Result<Vec<(StorageCoord, Block)>, WorldAccessError>;
    fn block_count(&self) -> u64;

    fn raycast(&self, ray: GridRay) -> Result<Option<RayHit>, WorldAccessError> {
        if ray.step == [0, 0, 0] || ray.step.iter().any(|value| !(-1..=1).contains(value)) {
            return Err(WorldAccessError::InvalidRayStep(ray.step));
        }
        if !self.extent().contains_storage(ray.origin) {
            return Err(WorldAccessError::OutOfBounds(ray.origin));
        }
        let mut coordinate = ray.origin;
        for steps in 0..=ray.max_steps {
            if let Some(block) = self.get_block_value(coordinate)? {
                return Ok(Some(RayHit {
                    coordinate,
                    block,
                    steps,
                }));
            }
            let Some(x) = coordinate.x.checked_add_signed(i32::from(ray.step[0])) else {
                return Ok(None);
            };
            let Some(y) = coordinate.y.checked_add_signed(i32::from(ray.step[1])) else {
                return Ok(None);
            };
            let Some(z) = coordinate.z.checked_add_signed(i32::from(ray.step[2])) else {
                return Ok(None);
            };
            coordinate = StorageCoord { x, y, z };
            if !self.extent().contains_storage(coordinate) {
                return Ok(None);
            }
        }
        Ok(None)
    }
}

pub trait WorldMutation {
    fn set_block_value(
        &mut self,
        coord: StorageCoord,
        block: Block,
    ) -> Result<Option<Block>, WorldAccessError>;
    fn remove_block_value(
        &mut self,
        coord: StorageCoord,
    ) -> Result<Option<Block>, WorldAccessError>;
    fn update_block_value(
        &mut self,
        coord: StorageCoord,
        update: BlockUpdate,
    ) -> Result<Block, WorldAccessError>;
    fn clear_blocks(&mut self);
}

pub trait WorldResidency {
    type Guard<'a>
    where
        Self: 'a;

    fn is_region_resident(&self, region: BoundedRegion) -> bool;
    fn pin_region(&self, region: BoundedRegion) -> Result<Self::Guard<'_>, WorldAccessError>;
}

#[derive(Debug)]
pub struct InMemoryResidentGuard<'a> {
    region: BoundedRegion,
    _owner: PhantomData<&'a ()>,
}

impl<'a> InMemoryResidentGuard<'a> {
    pub(crate) const fn new(region: BoundedRegion) -> Self {
        Self {
            region,
            _owner: PhantomData,
        }
    }

    pub const fn region(&self) -> BoundedRegion {
        self.region
    }
}

pub(crate) fn validate_coordinate(
    extent: WorldExtent,
    coord: StorageCoord,
) -> Result<(), WorldAccessError> {
    if extent.contains_storage(coord) {
        Ok(())
    } else {
        Err(WorldAccessError::OutOfBounds(coord))
    }
}
