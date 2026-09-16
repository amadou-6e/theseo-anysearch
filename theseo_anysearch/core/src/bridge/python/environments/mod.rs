pub mod multi_voxel;
pub mod surface;
pub mod voxel;

use pyo3::prelude::*;

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<voxel::PyVoxelObservation>()?;
    module.add_class::<voxel::PyStepResultVoxel>()?;
    module.add_class::<voxel::PyVoxelEnv>()?;
    module.add_class::<multi_voxel::PyMultiVoxelObs>()?;
    module.add_class::<multi_voxel::PyMultiVoxelStepResult>()?;
    module.add_class::<multi_voxel::PyMultiVoxelEnv>()?;
    module.add_class::<surface::PyAgentState>()?;
    module.add_class::<surface::PySurfaceObservation>()?;
    module.add_class::<surface::PyStepResultSurface>()?;
    module.add_class::<surface::PySurfaceEnv>()?;
    Ok(())
}
