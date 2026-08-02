use std::path::Path;

use libloading::{Library, Symbol};

use crate::voxel::common::{load_library, validate_name, validate_parameters, ABI_VERSION};

use super::abi::{PredicateContextV2, PredicateResultV2};

type PredicateFunctionV2 =
    unsafe extern "C" fn(*const PredicateContextV2, *mut PredicateResultV2) -> i32;

pub struct NativePredicateExtension {
    _library: Library,
    function: PredicateFunctionV2,
    name: String,
    parameters_json: Vec<u8>,
}

impl NativePredicateExtension {
    pub fn load(path: &Path, name: &str, parameters_json: String) -> Result<Self, String> {
        validate_name("predicate", name)?;
        validate_parameters("predicate", &parameters_json)?;
        let library = load_library(path, "predicate")?;
        let symbol_name = format!("anysearch_predicate_{name}_v2\0");
        let function = unsafe {
            let symbol: Symbol<PredicateFunctionV2> = library
                .get(symbol_name.as_bytes())
                .map_err(|error| format!("predicate {name:?} is not exported: {error}"))?;
            *symbol
        };
        Ok(Self {
            _library: library,
            function,
            name: name.to_owned(),
            parameters_json: parameters_json.into_bytes(),
        })
    }

    pub fn evaluate(&self, mut context: PredicateContextV2) -> Result<bool, String> {
        context.parameters_json = self.parameters_json.as_ptr();
        context.parameters_json_len = self.parameters_json.len();
        let mut result = PredicateResultV2::default();
        let status = unsafe { (self.function)(&context, &mut result) };
        if status != 0 {
            return Err(format!(
                "native predicate {:?} returned status {status}",
                self.name
            ));
        }
        if result.abi_version != ABI_VERSION
            || result.struct_size as usize != std::mem::size_of::<PredicateResultV2>()
        {
            return Err(format!(
                "native predicate {:?} returned an incompatible result",
                self.name
            ));
        }
        Ok(result.feasible != 0)
    }
}
