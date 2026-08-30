use crate::voxel::scenarios::{
    abi::{
        QueryStatus, WorldBlockV1, WorldCoordV1, WorldRayHitV1, WorldRayStepV1, WorldRegionEntryV1,
    },
    WorldQueryScope,
};
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

/// Capture the same valid probe through the scenario-v2 C callback table.
pub fn capture_abi_read<W: WorldRead>(
    world: &W,
    probe: &ReadProbe,
    token: u64,
) -> Result<ReadSnapshot, String> {
    let scope = WorldQueryScope::enter(world, world.extent(), token)
        .map_err(|status| format!("failed to enter query scope: {status:?}"))?;
    let api = scope.api();
    let point = api.point.ok_or("point callback is missing")?;
    let region = api.region.ok_or("region callback is missing")?;
    let ray = api.ray.ok_or("ray callback is missing")?;

    let points = probe
        .points
        .iter()
        .map(|coordinate| {
            let mut output = WorldBlockV1::default();
            let status = unsafe {
                point(
                    api.context,
                    api.call_token,
                    ffi_coord(*coordinate),
                    &mut output,
                )
            };
            match status {
                QueryStatus::BlockHit => Ok(Some(block(output))),
                QueryStatus::EmptyOrMiss => Ok(None),
                other => Err(format!("point callback failed: {other:?}")),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    let regions = probe
        .regions
        .iter()
        .map(|bounds| {
            let mut required = 0usize;
            let status = unsafe {
                region(
                    api.context,
                    api.call_token,
                    ffi_coord(bounds.minimum),
                    ffi_coord(bounds.maximum_exclusive),
                    std::ptr::null_mut(),
                    0,
                    &mut required,
                )
            };
            if !matches!(
                status,
                QueryStatus::InsufficientBuffer | QueryStatus::EmptyOrMiss
            ) {
                return Err(format!("region sizing callback failed: {status:?}"));
            }
            let mut output = vec![WorldRegionEntryV1::default(); required];
            let status = unsafe {
                region(
                    api.context,
                    api.call_token,
                    ffi_coord(bounds.minimum),
                    ffi_coord(bounds.maximum_exclusive),
                    output.as_mut_ptr(),
                    output.len(),
                    &mut required,
                )
            };
            if !matches!(status, QueryStatus::BlockHit | QueryStatus::EmptyOrMiss) {
                return Err(format!("region callback failed: {status:?}"));
            }
            Ok(output
                .into_iter()
                .map(|entry| (storage(entry.coordinate), block(entry.block)))
                .collect())
        })
        .collect::<Result<Vec<_>, String>>()?;
    let rays = probe
        .rays
        .iter()
        .map(|query| {
            let mut output = WorldRayHitV1::default();
            let status = unsafe {
                ray(
                    api.context,
                    api.call_token,
                    ffi_coord(query.origin),
                    WorldRayStepV1 {
                        x: query.step[0],
                        y: query.step[1],
                        z: query.step[2],
                    },
                    query.max_steps,
                    &mut output,
                )
            };
            match status {
                QueryStatus::BlockHit => Ok(Some(RayHit {
                    coordinate: storage(output.coordinate),
                    block: block(output.block),
                    steps: output.steps,
                })),
                QueryStatus::EmptyOrMiss => Ok(None),
                other => Err(format!("ray callback failed: {other:?}")),
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(ReadSnapshot {
        points: points.into_iter().map(Ok).collect(),
        regions: regions.into_iter().map(Ok).collect(),
        rays: rays.into_iter().map(Ok).collect(),
        count: world.block_count(),
    })
}

fn ffi_coord(value: StorageCoord) -> WorldCoordV1 {
    WorldCoordV1 {
        x: value.x,
        y: value.y,
        z: value.z,
    }
}

fn storage(value: WorldCoordV1) -> StorageCoord {
    StorageCoord {
        x: value.x,
        y: value.y,
        z: value.z,
    }
}

fn block(value: WorldBlockV1) -> Block {
    Block {
        kind: value.kind,
        active: value.active != 0,
        reward_weight: value.reward_weight,
    }
}
