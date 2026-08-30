use crate::voxel::world::{
    Block, BlockUpdate, BoundedRegion, GridRay, RayHit, StorageCoord, WorldAccessError,
    WorldMutation, WorldRead,
};

#[derive(Clone, Debug)]
pub enum Mutation {
    Set(StorageCoord, Block),
    Remove(StorageCoord),
    Update(StorageCoord, BlockUpdate),
    Clear,
}

#[derive(Clone, Debug, PartialEq)]
pub enum MutationResult {
    Block(Result<Option<Block>, WorldAccessError>),
    Updated(Result<Block, WorldAccessError>),
    Cleared,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReadSnapshot {
    pub points: Vec<Result<Option<Block>, WorldAccessError>>,
    pub regions: Vec<Result<Vec<(StorageCoord, Block)>, WorldAccessError>>,
    pub rays: Vec<Result<Option<RayHit>, WorldAccessError>>,
    pub count: u64,
}

#[derive(Clone, Debug)]
pub struct ReadProbe {
    pub points: Vec<StorageCoord>,
    pub regions: Vec<BoundedRegion>,
    pub rays: Vec<GridRay>,
}

pub fn capture_read<W: WorldRead>(world: &W, probe: &ReadProbe) -> ReadSnapshot {
    ReadSnapshot {
        points: probe
            .points
            .iter()
            .map(|coordinate| world.get_block_value(*coordinate))
            .collect(),
        regions: probe
            .regions
            .iter()
            .map(|region| world.blocks_in_region(*region))
            .collect(),
        rays: probe.rays.iter().map(|ray| world.raycast(*ray)).collect(),
        count: world.block_count(),
    }
}

pub fn apply_mutation<W: WorldRead + WorldMutation>(
    world: &mut W,
    mutation: Mutation,
) -> MutationResult {
    match mutation {
        Mutation::Set(coordinate, block) => {
            MutationResult::Block(world.set_block_value(coordinate, block))
        }
        Mutation::Remove(coordinate) => MutationResult::Block(world.remove_block_value(coordinate)),
        Mutation::Update(coordinate, update) => {
            MutationResult::Updated(world.update_block_value(coordinate, update))
        }
        Mutation::Clear => {
            world.clear_blocks();
            MutationResult::Cleared
        }
    }
}

pub fn compare_exact<W: WorldRead, O: WorldRead>(
    oracle: &W,
    candidate: &O,
    probe: &ReadProbe,
) -> Result<(), String> {
    let expected = capture_read(oracle, probe);
    let actual = capture_read(candidate, probe);
    (expected == actual)
        .then_some(())
        .ok_or_else(|| format!("backend mismatch\nexpected: {expected:#?}\nactual: {actual:#?}"))
}
