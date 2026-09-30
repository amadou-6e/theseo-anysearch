use std::path::Path;

use libloading::Library;

use super::version::validate_library_version;

pub fn load_library(path: &Path, kind: &str) -> Result<Library, String> {
    let library = unsafe { Library::new(path) }
        .map_err(|error| format!("cannot load {kind} library {}: {error}", path.display()))?;
    validate_library_version(&library, kind)?;
    Ok(library)
}
