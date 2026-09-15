//! Versioned native procedural-geometry provider ABI.

pub mod abi;
mod loader;

pub use loader::{GeometryInvocationV1, NativeGeometryV1};
