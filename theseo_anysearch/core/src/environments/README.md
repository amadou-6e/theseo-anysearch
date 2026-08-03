# Environment contracts

This module contains only environment-family-neutral transition contracts:

- `Environment` defines `reset` and `step`;
- `StepResult` carries an observation, reward, and termination flag.

Concrete environment families own their implementations. Voxel environments live under
`crate::voxel`; the surface-navigation environment lives under `crate::surface`.