//! Call-scoped implementation of the world-query callback table.
use super::abi::{
    QueryStatus, WorldBlockV1, WorldCoordV1, WorldQueryApiV1, WorldRayHitV1, WorldRayStepV1,
    WorldRegionEntryV1, MAX_REGION_RESULTS, WORLD_QUERY_ABI_VERSION,
};
use crate::voxel::world::{
    Block, BoundedRegion, GridRay, StorageCoord, WorldAccessError, WorldExtent, WorldRead,
};
use std::{
    cell::RefCell,
    ffi::c_void,
    panic::{catch_unwind, AssertUnwindSafe},
};

struct ActiveCall {
    token: u64,
    world: *const (dyn WorldRead + 'static),
    in_callback: bool,
    extent: WorldExtent,
}
thread_local! { static ACTIVE_CALL: RefCell<Option<ActiveCall>> = const { RefCell::new(None) }; }

/// Owns a single-thread invocation. Dropping it invalidates copied tokens and tables.
pub struct WorldQueryScope {
    token: u64,
}
impl WorldQueryScope {
    pub fn enter(
        world: &dyn WorldRead,
        extent: WorldExtent,
        token: u64,
    ) -> Result<Self, QueryStatus> {
        if token == 0 {
            return Err(QueryStatus::InvalidArgument);
        }
        ACTIVE_CALL.with(|slot| {
            let mut active = slot.borrow_mut();
            if active.is_some() {
                return Err(QueryStatus::HostFailure);
            }
            // Internal lifetime erasure is bounded by this guard and never crosses the ABI.
            let pointer = unsafe {
                std::mem::transmute::<*const dyn WorldRead, *const (dyn WorldRead + 'static)>(world)
            };
            *active = Some(ActiveCall {
                token,
                world: pointer,
                in_callback: false,
                extent,
            });
            Ok(Self { token })
        })
    }
    pub fn api(&self) -> WorldQueryApiV1 {
        WorldQueryApiV1 {
            abi_version: WORLD_QUERY_ABI_VERSION,
            struct_size: std::mem::size_of::<WorldQueryApiV1>() as u32,
            coordinate_size: std::mem::size_of::<WorldCoordV1>() as u32,
            block_size: std::mem::size_of::<WorldBlockV1>() as u32,
            region_entry_size: std::mem::size_of::<WorldRegionEntryV1>() as u32,
            ray_hit_size: std::mem::size_of::<WorldRayHitV1>() as u32,
            context: self.token as usize as *mut c_void,
            call_token: self.token,
            point: Some(point),
            region: Some(region),
            ray: Some(ray),
            count_region: Some(count_region),
        }
    }
}
impl Drop for WorldQueryScope {
    fn drop(&mut self) {
        ACTIVE_CALL.with(|slot| {
            if slot
                .borrow()
                .as_ref()
                .is_some_and(|call| call.token == self.token)
            {
                *slot.borrow_mut() = None;
            }
        });
    }
}

struct BoundedWorld<'a> {
    inner: &'a dyn WorldRead,
    extent: WorldExtent,
}
impl WorldRead for BoundedWorld<'_> {
    fn extent(&self) -> WorldExtent {
        self.extent
    }
    fn get_block_value(&self, coordinate: StorageCoord) -> Result<Option<Block>, WorldAccessError> {
        if !self.extent.contains_storage(coordinate) {
            return Err(WorldAccessError::OutOfBounds(coordinate));
        }
        self.inner.get_block_value(coordinate)
    }
    fn blocks_in_region(
        &self,
        region: BoundedRegion,
    ) -> Result<Vec<(StorageCoord, Block)>, WorldAccessError> {
        BoundedRegion::new(region.minimum, region.maximum_exclusive, self.extent)?;
        self.inner.blocks_in_region(region)
    }
    fn block_count(&self) -> u64 {
        self.inner.block_count()
    }
}

fn with_world<T>(
    context: *mut c_void,
    token: u64,
    op: impl FnOnce(&dyn WorldRead) -> Result<T, QueryStatus>,
) -> Result<T, QueryStatus> {
    if context.is_null() || context as usize as u64 != token {
        return Err(QueryStatus::StaleToken);
    }
    ACTIVE_CALL.with(|slot| {
        let (pointer, extent) = {
            let mut active = slot.borrow_mut();
            let call = active.as_mut().ok_or(QueryStatus::StaleToken)?;
            if call.token != token {
                return Err(QueryStatus::StaleToken);
            }
            if call.in_callback {
                return Err(QueryStatus::HostFailure);
            }
            call.in_callback = true;
            (call.world, call.extent)
        };
        let bounded = BoundedWorld {
            inner: unsafe { &*pointer },
            extent,
        };
        let result = catch_unwind(AssertUnwindSafe(|| op(&bounded)));
        if let Some(call) = slot.borrow_mut().as_mut() {
            call.in_callback = false;
        }
        match result {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    })
}
fn coord(v: WorldCoordV1) -> StorageCoord {
    StorageCoord {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}
fn ffi_coord(v: StorageCoord) -> WorldCoordV1 {
    WorldCoordV1 {
        x: v.x,
        y: v.y,
        z: v.z,
    }
}
fn ffi_block(v: Block) -> WorldBlockV1 {
    WorldBlockV1 {
        kind: v.kind,
        active: u8::from(v.active),
        reserved: [0; 2],
        reward_weight: v.reward_weight,
    }
}
fn map_error(e: WorldAccessError) -> QueryStatus {
    match e {
        WorldAccessError::OutOfBounds(_) => QueryStatus::OutOfBounds,
        WorldAccessError::BackendFailure => QueryStatus::BackendFailure,
        WorldAccessError::Unsupported => QueryStatus::Unsupported,
        _ => QueryStatus::InvalidArgument,
    }
}
fn guarded(op: impl FnOnce() -> Result<QueryStatus, QueryStatus>) -> QueryStatus {
    match catch_unwind(AssertUnwindSafe(op)) {
        Ok(Ok(s)) => s,
        Ok(Err(s)) => s,
        Err(_) => QueryStatus::HostFailure,
    }
}

unsafe extern "C" fn point(
    context: *mut c_void,
    token: u64,
    coordinate: WorldCoordV1,
    output: *mut WorldBlockV1,
) -> QueryStatus {
    guarded(|| {
        if output.is_null() {
            return Err(QueryStatus::InvalidArgument);
        }
        with_world(context, token, |world| {
            match world
                .get_block_value(coord(coordinate))
                .map_err(map_error)?
            {
                Some(block) => {
                    unsafe { output.write(ffi_block(block)) };
                    Ok(QueryStatus::BlockHit)
                }
                None => Ok(QueryStatus::EmptyOrMiss),
            }
        })
    })
}
fn checked_region(
    world: &dyn WorldRead,
    min: WorldCoordV1,
    max: WorldCoordV1,
) -> Result<BoundedRegion, QueryStatus> {
    if min.x >= max.x || min.y >= max.y || min.z >= max.z {
        return Err(QueryStatus::InvalidArgument);
    }
    let extent = world.extent();
    if max.x > extent.x || max.y > extent.y || max.z > extent.z {
        return Err(QueryStatus::OutOfBounds);
    }
    let dx = max
        .x
        .checked_sub(min.x)
        .ok_or(QueryStatus::InvalidArgument)? as usize;
    let dy = max
        .y
        .checked_sub(min.y)
        .ok_or(QueryStatus::InvalidArgument)? as usize;
    let dz = max
        .z
        .checked_sub(min.z)
        .ok_or(QueryStatus::InvalidArgument)? as usize;
    dx.checked_mul(dy)
        .and_then(|n| n.checked_mul(dz))
        .and_then(|n| n.checked_mul(std::mem::size_of::<WorldRegionEntryV1>()))
        .ok_or(QueryStatus::InvalidArgument)?;
    BoundedRegion::new(coord(min), coord(max), extent).map_err(map_error)
}
unsafe extern "C" fn region(
    context: *mut c_void,
    token: u64,
    min: WorldCoordV1,
    max: WorldCoordV1,
    output: *mut WorldRegionEntryV1,
    capacity: usize,
    required: *mut usize,
) -> QueryStatus {
    guarded(|| {
        if required.is_null()
            || (capacity != 0 && output.is_null())
            || capacity
                .checked_mul(std::mem::size_of::<WorldRegionEntryV1>())
                .is_none()
        {
            return Err(QueryStatus::InvalidArgument);
        }
        with_world(context, token, |world| {
            let values = world
                .blocks_in_region(checked_region(world, min, max)?)
                .map_err(map_error)?;
            if values.len() > MAX_REGION_RESULTS {
                return Err(QueryStatus::InvalidArgument);
            }
            unsafe { required.write(values.len()) };
            if capacity < values.len() {
                return Ok(QueryStatus::InsufficientBuffer);
            }
            for (index, (c, b)) in values.iter().enumerate() {
                unsafe {
                    output.add(index).write(WorldRegionEntryV1 {
                        coordinate: ffi_coord(*c),
                        block: ffi_block(b.clone()),
                    })
                }
            }
            Ok(if values.is_empty() {
                QueryStatus::EmptyOrMiss
            } else {
                QueryStatus::BlockHit
            })
        })
    })
}
unsafe extern "C" fn ray(
    context: *mut c_void,
    token: u64,
    origin: WorldCoordV1,
    step: WorldRayStepV1,
    max_steps: u32,
    output: *mut WorldRayHitV1,
) -> QueryStatus {
    guarded(|| {
        if output.is_null() {
            return Err(QueryStatus::InvalidArgument);
        }
        with_world(context, token, |world| {
            match world
                .raycast(
                    GridRay::new(coord(origin), [step.x, step.y, step.z], max_steps)
                        .map_err(map_error)?,
                )
                .map_err(map_error)?
            {
                Some(hit) => {
                    unsafe {
                        output.write(WorldRayHitV1 {
                            coordinate: ffi_coord(hit.coordinate),
                            block: ffi_block(hit.block),
                            steps: hit.steps,
                        })
                    };
                    Ok(QueryStatus::BlockHit)
                }
                None => Ok(QueryStatus::EmptyOrMiss),
            }
        })
    })
}
unsafe extern "C" fn count_region(
    context: *mut c_void,
    token: u64,
    min: WorldCoordV1,
    max: WorldCoordV1,
    output: *mut u64,
) -> QueryStatus {
    guarded(|| {
        if output.is_null() {
            return Err(QueryStatus::InvalidArgument);
        }
        with_world(context, token, |world| {
            let count = world
                .blocks_in_region(checked_region(world, min, max)?)
                .map_err(map_error)?
                .len() as u64;
            unsafe { output.write(count) };
            Ok(if count == 0 {
                QueryStatus::EmptyOrMiss
            } else {
                QueryStatus::BlockHit
            })
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::world::{Block, WorldExtent, WorldMutation, WorldResidency, WorldState};

    fn populated() -> WorldState {
        let mut world = WorldState::new();
        world
            .set_block_value(StorageCoord { x: 2, y: 3, z: 4 }, Block::default())
            .unwrap();
        world
    }
    #[test]
    fn point_hit_empty_and_stale_are_distinct() {
        let world = populated();
        let scope = WorldQueryScope::enter(&world, world.extent(), 7).unwrap();
        let api = scope.api();
        let mut block = WorldBlockV1::default();
        let f = api.point.unwrap();
        assert_eq!(
            unsafe {
                f(
                    api.context,
                    7,
                    WorldCoordV1 { x: 2, y: 3, z: 4 },
                    &mut block,
                )
            },
            QueryStatus::BlockHit
        );
        assert_eq!(
            unsafe {
                f(
                    api.context,
                    7,
                    WorldCoordV1 { x: 1, y: 1, z: 1 },
                    &mut block,
                )
            },
            QueryStatus::EmptyOrMiss
        );
        drop(scope);
        assert_eq!(
            unsafe {
                f(
                    api.context,
                    7,
                    WorldCoordV1 { x: 1, y: 1, z: 1 },
                    &mut block,
                )
            },
            QueryStatus::StaleToken
        )
    }
    #[test]
    fn logical_extent_hides_larger_backend_coordinates() {
        let world = populated();
        let scope = WorldQueryScope::enter(&world, WorldExtent::cubic(32), 70).unwrap();
        let api = scope.api();
        let mut out = WorldBlockV1::default();
        assert_eq!(
            unsafe {
                api.point.unwrap()(
                    api.context,
                    70,
                    WorldCoordV1 { x: 40, y: 1, z: 1 },
                    &mut out,
                )
            },
            QueryStatus::OutOfBounds
        );
    }
    #[test]
    fn region_negotiates_exact_length_and_rejects_bad_buffers() {
        let world = populated();
        let scope = WorldQueryScope::enter(&world, world.extent(), 8).unwrap();
        let api = scope.api();
        let f = api.region.unwrap();
        let min = WorldCoordV1 { x: 0, y: 0, z: 0 };
        let max = WorldCoordV1 { x: 5, y: 5, z: 5 };
        let mut required = 0;
        assert_eq!(
            unsafe {
                f(
                    api.context,
                    8,
                    min,
                    max,
                    std::ptr::null_mut(),
                    0,
                    &mut required,
                )
            },
            QueryStatus::InsufficientBuffer
        );
        assert_eq!(required, 1);
        let mut out = WorldRegionEntryV1::default();
        assert_eq!(
            unsafe { f(api.context, 8, min, max, &mut out, 1, &mut required) },
            QueryStatus::BlockHit
        );
        assert_eq!(out.coordinate, WorldCoordV1 { x: 2, y: 3, z: 4 });
        assert_eq!(
            unsafe {
                f(
                    api.context,
                    8,
                    min,
                    max,
                    std::ptr::null_mut(),
                    1,
                    &mut required,
                )
            },
            QueryStatus::InvalidArgument
        );
        assert_eq!(
            unsafe {
                f(
                    api.context,
                    8,
                    min,
                    max,
                    &mut out,
                    usize::MAX,
                    &mut required,
                )
            },
            QueryStatus::InvalidArgument
        );
        assert_eq!(
            unsafe {
                f(
                    api.context,
                    8,
                    min,
                    WorldCoordV1 {
                        x: 1001,
                        y: 5,
                        z: 5,
                    },
                    &mut out,
                    1,
                    &mut required,
                )
            },
            QueryStatus::OutOfBounds
        );
        assert_eq!(
            unsafe { f(api.context, 8, max, min, &mut out, 1, &mut required) },
            QueryStatus::InvalidArgument
        );
    }
    #[test]
    fn ray_hit_and_miss_are_distinct() {
        let world = populated();
        let scope = WorldQueryScope::enter(&world, world.extent(), 9).unwrap();
        let api = scope.api();
        let f = api.ray.unwrap();
        let mut out = WorldRayHitV1::default();
        assert_eq!(
            unsafe {
                f(
                    api.context,
                    9,
                    WorldCoordV1 { x: 0, y: 3, z: 4 },
                    WorldRayStepV1 { x: 1, y: 0, z: 0 },
                    5,
                    &mut out,
                )
            },
            QueryStatus::BlockHit
        );
        assert_eq!(out.steps, 2);
        assert_eq!(
            unsafe {
                f(
                    api.context,
                    9,
                    WorldCoordV1 { x: 0, y: 0, z: 0 },
                    WorldRayStepV1 { x: 1, y: 0, z: 0 },
                    5,
                    &mut out,
                )
            },
            QueryStatus::EmptyOrMiss
        )
    }
    struct FailedWorld;
    impl WorldRead for FailedWorld {
        fn extent(&self) -> WorldExtent {
            WorldExtent::cubic(8)
        }
        fn get_block_value(&self, _: StorageCoord) -> Result<Option<Block>, WorldAccessError> {
            Err(WorldAccessError::BackendFailure)
        }
        fn blocks_in_region(
            &self,
            _: BoundedRegion,
        ) -> Result<Vec<(StorageCoord, Block)>, WorldAccessError> {
            Err(WorldAccessError::BackendFailure)
        }
        fn block_count(&self) -> u64 {
            0
        }
    }
    #[test]
    fn backend_failure_is_not_empty() {
        let world = FailedWorld;
        let scope = WorldQueryScope::enter(&world, world.extent(), 10).unwrap();
        let api = scope.api();
        let mut out = WorldBlockV1::default();
        assert_eq!(
            unsafe { api.point.unwrap()(api.context, 10, WorldCoordV1::default(), &mut out) },
            QueryStatus::BackendFailure
        )
    }
    #[test]
    fn null_context_and_output_are_rejected() {
        let world = populated();
        let scope = WorldQueryScope::enter(&world, world.extent(), 11).unwrap();
        let api = scope.api();
        assert_eq!(
            unsafe {
                api.point.unwrap()(
                    std::ptr::null_mut(),
                    11,
                    WorldCoordV1::default(),
                    std::ptr::null_mut(),
                )
            },
            QueryStatus::InvalidArgument
        );
        let mut out = WorldBlockV1::default();
        assert_eq!(
            unsafe { api.point.unwrap()(api.context, 12, WorldCoordV1::default(), &mut out) },
            QueryStatus::StaleToken
        )
    }
    struct PanicWorld;
    impl WorldRead for PanicWorld {
        fn extent(&self) -> WorldExtent {
            WorldExtent::cubic(8)
        }
        fn get_block_value(&self, _: StorageCoord) -> Result<Option<Block>, WorldAccessError> {
            panic!("backend panic")
        }
        fn blocks_in_region(
            &self,
            _: BoundedRegion,
        ) -> Result<Vec<(StorageCoord, Block)>, WorldAccessError> {
            panic!("backend panic")
        }
        fn block_count(&self) -> u64 {
            0
        }
    }
    #[test]
    fn backend_panic_does_not_cross_callback() {
        let world = PanicWorld;
        let scope = WorldQueryScope::enter(&world, world.extent(), 13).unwrap();
        let api = scope.api();
        let mut out = WorldBlockV1::default();
        assert_eq!(
            unsafe { api.point.unwrap()(api.context, 13, WorldCoordV1::default(), &mut out) },
            QueryStatus::HostFailure
        )
    }

    #[test]
    fn region_results_match_before_and_after_residency_pin() {
        let world = populated();
        let bounded = BoundedRegion::new(
            StorageCoord { x: 0, y: 0, z: 0 },
            StorageCoord { x: 8, y: 8, z: 8 },
            world.extent(),
        )
        .unwrap();
        let collect = |token| {
            let scope = WorldQueryScope::enter(&world, world.extent(), token).unwrap();
            let api = scope.api();
            let mut required = 0;
            let f = api.region.unwrap();
            assert_eq!(
                unsafe {
                    f(
                        api.context,
                        token,
                        WorldCoordV1 { x: 0, y: 0, z: 0 },
                        WorldCoordV1 { x: 8, y: 8, z: 8 },
                        std::ptr::null_mut(),
                        0,
                        &mut required,
                    )
                },
                QueryStatus::InsufficientBuffer
            );
            let mut values = vec![WorldRegionEntryV1::default(); required];
            assert_eq!(
                unsafe {
                    f(
                        api.context,
                        token,
                        WorldCoordV1 { x: 0, y: 0, z: 0 },
                        WorldCoordV1 { x: 8, y: 8, z: 8 },
                        values.as_mut_ptr(),
                        values.len(),
                        &mut required,
                    )
                },
                QueryStatus::BlockHit
            );
            values
        };
        let cold = collect(20);
        let _guard = world.pin_region(bounded).unwrap();
        let hot = collect(21);
        assert_eq!(cold, hot);
    }
}
