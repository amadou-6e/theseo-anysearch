use super::{
    abi::{
        ScenarioContextV2, ScenarioFunctionV2, ScenarioStatusV2, MAX_SCENARIO_OUTPUT_BYTES,
        SCENARIO_ABI_VERSION_V2,
    },
    query::WorldQueryScope,
};
use crate::voxel::{
    common::{load_library, validate_name},
    world::{WorldExtent, WorldRead},
};
use libloading::{Library, Symbol};
use std::{
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_TOKEN: AtomicU64 = AtomicU64::new(1);

/// Owned reset metadata. No filled-voxel array is serialized into this context.
pub struct ScenarioInvocationV2<'a> {
    pub seed: u64,
    pub episode_index: u64,
    pub grid_size: u32,
    pub scope: &'a str,
    pub action_mode: &'a str,
    pub action_offsets_json: &'a str,
    pub previous_scenario_json: &'a str,
    pub curriculum_json: &'a str,
    pub parameters_json: &'a str,
}

/// Loaded v2 scenario extension. The library remains resident while its function is callable.
pub struct NativeScenarioV2 {
    _library: Library,
    function: ScenarioFunctionV2,
    name: String,
}
impl NativeScenarioV2 {
    pub fn load(path: &Path, name: &str) -> Result<Self, String> {
        validate_name("scenario", name)?;
        let library = load_library(path, "scenario")?;
        let symbol_name = format!("anysearch_scenario_{name}_v2\0");
        let function = unsafe {
            let symbol: Symbol<ScenarioFunctionV2> = library
                .get(symbol_name.as_bytes())
                .map_err(|error| format!("scenario {name:?} v2 is not exported: {error}"))?;
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
        input: &ScenarioInvocationV2<'_>,
    ) -> Result<String, String> {
        let bytes = |value: &str| (value.as_ptr(), value.len());
        let (scope_ptr, scope_len) = bytes(input.scope);
        let (mode_ptr, mode_len) = bytes(input.action_mode);
        let (offsets_ptr, offsets_len) = bytes(input.action_offsets_json);
        let (previous_ptr, previous_len) = bytes(input.previous_scenario_json);
        let (curriculum_ptr, curriculum_len) = bytes(input.curriculum_json);
        let (parameters_ptr, parameters_len) = bytes(input.parameters_json);
        let call = |output: *mut u8,
                    capacity: usize,
                    required: &mut usize|
         -> Result<ScenarioStatusV2, String> {
            let token = NEXT_TOKEN.fetch_add(1, Ordering::Relaxed).max(1);
            if input.grid_size == 0 {
                return Err("scenario grid_size must be positive".to_owned());
            }
            let query_scope =
                WorldQueryScope::enter(world, WorldExtent::cubic(input.grid_size), token).map_err(
                    |status| format!("could not establish scenario query scope: {status:?}"),
                )?;
            let api = query_scope.api();
            let context = ScenarioContextV2 {
                abi_version: SCENARIO_ABI_VERSION_V2,
                struct_size: std::mem::size_of::<ScenarioContextV2>() as u32,
                seed: input.seed,
                episode_index: input.episode_index,
                grid_size: input.grid_size,
                scope: scope_ptr,
                scope_len,
                action_mode: mode_ptr,
                action_mode_len: mode_len,
                action_offsets_json: offsets_ptr,
                action_offsets_json_len: offsets_len,
                previous_scenario_json: previous_ptr,
                previous_scenario_json_len: previous_len,
                curriculum_json: curriculum_ptr,
                curriculum_json_len: curriculum_len,
                parameters_json: parameters_ptr,
                parameters_json_len: parameters_len,
                world: &api,
            };
            Ok(unsafe { (self.function)(&context, output, capacity, required) })
        };
        let mut required = 0usize;
        let first = call(std::ptr::null_mut(), 0, &mut required)?;
        if first != ScenarioStatusV2::InsufficientBuffer {
            return Err(format!(
                "native scenario {:?} length negotiation returned {first:?}",
                self.name
            ));
        }
        if required == 0 || required > MAX_SCENARIO_OUTPUT_BYTES {
            return Err(format!("native scenario {:?} requested invalid output length {required} (maximum {MAX_SCENARIO_OUTPUT_BYTES})",self.name));
        }
        let mut output = vec![0u8; required];
        let mut written = 0usize;
        let second = call(output.as_mut_ptr(), output.len(), &mut written)?;
        if second != ScenarioStatusV2::Success {
            return Err(format!(
                "native scenario {:?} returned {second:?}",
                self.name
            ));
        }
        if written > output.len() {
            return Err(format!(
                "native scenario {:?} reported output beyond its negotiated buffer",
                self.name
            ));
        }
        output.truncate(written);
        String::from_utf8(output)
            .map_err(|_| format!("native scenario {:?} returned non-UTF-8 JSON", self.name))
    }
}
