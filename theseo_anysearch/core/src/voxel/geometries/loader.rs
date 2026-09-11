use super::abi::{
    GeometryContextV1, GeometryFunctionV1, GeometryStatusV1, GEOMETRY_ABI_VERSION_V1,
    MAX_GEOMETRY_OUTPUT_BYTES,
};
use crate::voxel::{
    common::{load_library, validate_name},
    scenarios::{abi::WorldCoordV1, WorldQueryScope},
    world::{WorldExtent, WorldRead},
};
use libloading::{Library, Symbol};
use std::{
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

pub struct GeometryInvocationV1<'a> {
    pub seed: u64,
    pub attempt: u32,
    pub extent: WorldExtent,
    pub parameters_json: &'a str,
    pub task_json: &'a str,
}

pub struct NativeGeometryV1 {
    _library: Library,
    function: GeometryFunctionV1,
    name: String,
}

impl NativeGeometryV1 {
    pub fn load(path: &Path, name: &str) -> Result<Self, String> {
        validate_name("geometry", name)?;
        let library = load_library(path, "geometry")?;
        let symbol_name = format!("anysearch_geometry_{name}_v1\0");
        let function = unsafe {
            let symbol: Symbol<GeometryFunctionV1> = library
                .get(symbol_name.as_bytes())
                .map_err(|error| format!("geometry {name:?} v1 is not exported: {error}"))?;
            *symbol
        };
        Ok(Self {
            _library: library,
            function,
            name: name.to_owned(),
        })
    }

    pub fn invoke(
        &self,
        world: &dyn WorldRead,
        input: &GeometryInvocationV1<'_>,
    ) -> Result<String, String> {
        let call = |output: *mut u8,
                    capacity: usize,
                    required: &mut usize|
         -> Result<GeometryStatusV1, String> {
            let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed).max(1);
            let scope = WorldQueryScope::enter(world, input.extent, token).map_err(|status| {
                format!("could not establish geometry query scope: {status:?}")
            })?;
            let api = scope.api();
            let context = GeometryContextV1 {
                abi_version: GEOMETRY_ABI_VERSION_V1,
                struct_size: std::mem::size_of::<GeometryContextV1>() as u32,
                seed: input.seed,
                attempt: input.attempt,
                extent: WorldCoordV1 {
                    x: input.extent.x,
                    y: input.extent.y,
                    z: input.extent.z,
                },
                parameters_json: input.parameters_json.as_ptr(),
                parameters_json_len: input.parameters_json.len(),
                task_json: input.task_json.as_ptr(),
                task_json_len: input.task_json.len(),
                world: &api,
            };
            Ok(unsafe { (self.function)(&context, output, capacity, required) })
        };
        let mut required = 0;
        let first = call(std::ptr::null_mut(), 0, &mut required)?;
        if first != GeometryStatusV1::InsufficientBuffer {
            return Err(format!(
                "native geometry {:?} length negotiation returned {first:?}",
                self.name
            ));
        }
        if required == 0 || required > MAX_GEOMETRY_OUTPUT_BYTES {
            return Err(format!(
                "native geometry {:?} requested invalid output length {required}",
                self.name
            ));
        }
        let mut output = vec![0; required];
        let mut written = 0;
        let second = call(output.as_mut_ptr(), output.len(), &mut written)?;
        if second != GeometryStatusV1::Success || written > output.len() {
            return Err(format!(
                "native geometry {:?} returned {second:?} with length {written}",
                self.name
            ));
        }
        output.truncate(written);
        String::from_utf8(output)
            .map_err(|_| format!("native geometry {:?} returned non-UTF-8 JSON", self.name))
    }

    /// Invoke the provider twice for the same immutable context and reject
    /// output that cannot be reproduced. Both invocations share the already
    /// loaded library handle, so determinism enforcement does not reload the
    /// extension between calls.
    pub fn invoke_deterministic(
        &self,
        world: &dyn WorldRead,
        input: &GeometryInvocationV1<'_>,
    ) -> Result<String, String> {
        let first = self.invoke(world, input)?;
        let second = self.invoke(world, input)?;
        verify_deterministic_output(&self.name, first, second)
    }
}

fn verify_deterministic_output(
    name: &str,
    first: String,
    second: String,
) -> Result<String, String> {
    if first != second {
        return Err(format!(
            "native geometry {name:?} is nondeterministic for a fixed context"
        ));
    }
    Ok(first)
}

#[cfg(test)]
mod tests {
    use super::verify_deterministic_output;

    #[test]
    fn fixed_output_is_accepted() {
        assert_eq!(
            verify_deterministic_output("wall", "proposal".into(), "proposal".into()),
            Ok("proposal".into())
        );
    }

    #[test]
    fn changed_output_is_rejected_with_provider_name() {
        let error = verify_deterministic_output("wall", "first".into(), "second".into())
            .expect_err("changed output must fail");
        assert!(error.contains("wall"));
        assert!(error.contains("nondeterministic"));
    }
}
