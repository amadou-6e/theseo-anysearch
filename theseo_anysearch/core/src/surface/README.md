# Surface environment

This module owns the surface-navigation environment family:

- surface observations and actions;
- per-agent surface state;
- exterior-shell extraction and navigation;
- the `Environment` implementation for `SurfaceEnv`.

It consumes voxel coordinates and geometry from `crate::voxel::world`, but it is a distinct
environment family and does not own voxel storage, sampling, rewards, predicates, or outcomes.