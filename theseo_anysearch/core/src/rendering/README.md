# Rendering primitives

This module contains environment-independent software-rendering primitives:

- `projection.rs` projects 3D coordinates through a configured camera;
- `raster.rs` draws depth-aware squares and voxel cubes into image buffers.

Voxel episode tracing and training-video generation live under
`crate::voxel::rendering` because their data contracts depend on voxel coordinates,
voxelized geometry, and surface-navigation episodes. Python argument and error conversion
remains in `bridge/python/rendering.rs`.