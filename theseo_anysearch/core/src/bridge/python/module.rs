use pyo3::prelude::*;

use super::{environments, geometry, rendering, sampling};

pub fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    geometry::register(module)?;
    rendering::register(module)?;
    environments::register(module)?;
    sampling::register(module)?;
    Ok(())
}

#[pymodule]
pub fn theseo_core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register(module)
}
