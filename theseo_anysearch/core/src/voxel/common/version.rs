use libloading::{Library, Symbol};

pub const ABI_VERSION: u32 = 2;

pub fn validate_library_version(library: &Library, kind: &str) -> Result<(), String> {
    let version = unsafe {
        let symbol: Symbol<unsafe extern "C" fn() -> u32> = library
            .get(b"anysearch_extension_abi_version\0")
            .map_err(|error| format!("missing ABI version export: {error}"))?;
        symbol()
    };
    if version != ABI_VERSION {
        return Err(format!(
            "native {kind} ABI {version}, expected {ABI_VERSION}"
        ));
    }
    Ok(())
}
