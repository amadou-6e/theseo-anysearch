pub mod api;
pub mod block;
pub mod chunked;
pub mod coordinates;
pub mod hashmap_backend;
pub mod ingest;
pub mod regional;
pub mod state;

#[cfg(test)]
mod regional_tests;

pub use api::{World, WorldError};
pub use block::{
    Block, BlockUpdate, BLOCK_KIND_BOUNDARY, BLOCK_KIND_FILLED, BLOCK_KIND_GOAL,
    BLOCK_KIND_OCCUPIED, BLOCK_KIND_START,
};
pub use chunked::ChunkedWorld;
pub use coordinates::{
    storage_to_task, task_to_storage, CoordinateError, StorageCoord, TaskCoord, WorldExtent,
    WORLD_SCHEMA_VERSION,
};
pub use hashmap_backend::HashMapWorld;
pub use regional::{
    BoundedRegion, GridRay, InMemoryResidentGuard, RayHit, WorldAccessError, WorldMutation,
    WorldRead, WorldResidency,
};
pub use state::{Coord, WorldBackendKind, WorldHandle, WorldState, WORLD_CENTER, WORLD_SIZE};
