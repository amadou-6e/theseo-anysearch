use std::path::Path;

use libloading::{Library, Symbol};

use crate::voxel::common::{load_library, validate_name, validate_parameters, ABI_VERSION};

use super::abi::{OutcomeContextV2, OutcomeResultV2};

type OutcomeFunctionV2 = unsafe extern "C" fn(*const OutcomeContextV2, *mut OutcomeResultV2) -> i32;

pub struct NativeOutcomeExtension {
    _library: Library,
    function: OutcomeFunctionV2,
    name: String,
    parameters_json: Vec<u8>,
}

impl NativeOutcomeExtension {
    pub fn load(path: &Path, name: &str, parameters_json: String) -> Result<Self, String> {
        validate_name("outcome", name)?;
        validate_parameters("outcome", &parameters_json)?;
        let library = load_library(path, "outcome")?;
        let symbol_name = format!("anysearch_outcome_{name}_v2\0");
        let function = unsafe {
            let symbol: Symbol<OutcomeFunctionV2> = library
                .get(symbol_name.as_bytes())
                .map_err(|error| format!("outcome {name:?} is not exported: {error}"))?;
            *symbol
        };
        Ok(Self {
            _library: library,
            function,
            name: name.to_owned(),
            parameters_json: parameters_json.into_bytes(),
        })
    }

    pub fn evaluate(&self, mut context: OutcomeContextV2) -> Result<OutcomeResultV2, String> {
        context.parameters_json = self.parameters_json.as_ptr();
        context.parameters_json_len = self.parameters_json.len();
        let mut result = OutcomeResultV2::default();
        let status = unsafe { (self.function)(&context, &mut result) };
        if status != 0 {
            return Err(format!(
                "native outcome {:?} returned status {status}",
                self.name
            ));
        }
        if result.abi_version != ABI_VERSION
            || result.struct_size as usize != std::mem::size_of::<OutcomeResultV2>()
        {
            return Err(format!(
                "native outcome {:?} returned an incompatible result",
                self.name
            ));
        }
        Ok(result)
    }
}
