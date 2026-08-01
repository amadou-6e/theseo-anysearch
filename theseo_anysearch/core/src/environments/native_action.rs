use libloading::{Library, Symbol};
use std::path::Path;

use super::native_reward::ABI_VERSION;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ActionHistoryEntryV2 {
    pub action_index: i32,
    pub previous_cursor: [i32; 3],
    pub cursor: [i32; 3],
    pub feasible: u8,
    pub collision: u8,
}

#[repr(C)]
pub struct PredicateContextV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub step: u64,
    pub grid_size: u16,
    pub action_index: i32,
    pub cursor: [i32; 3],
    pub destination: [i32; 3],
    pub goal: [i32; 3],
    pub has_goal: u8,
    pub valid_action: u8,
    pub destination_in_bounds: u8,
    pub destination_blocked: u8,
    pub observation_filled: usize,
    pub observation_steps_remaining: u32,
    pub observation_goal_distance: u32,
    pub has_observation_goal_distance: u8,
    pub history: *const ActionHistoryEntryV2,
    pub history_len: usize,
    pub parameters_json: *const u8,
    pub parameters_json_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PredicateResultV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub feasible: u8,
}

impl Default for PredicateResultV2 {
    fn default() -> Self {
        Self {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            feasible: 1,
        }
    }
}

#[repr(C)]
pub struct OutcomeContextV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub step: u64,
    pub grid_size: u16,
    pub action_index: i32,
    pub cursor: [i32; 3],
    pub destination: [i32; 3],
    pub goal: [i32; 3],
    pub has_goal: u8,
    pub history: *const ActionHistoryEntryV2,
    pub history_len: usize,
    pub parameters_json: *const u8,
    pub parameters_json_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OutcomeResultV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub set_cursor: u8,
    pub cursor: [i32; 3],
    pub place_voxel: u8,
    pub place_coord: [i32; 3],
    pub remove_voxel: u8,
    pub remove_coord: [i32; 3],
}

impl Default for OutcomeResultV2 {
    fn default() -> Self {
        Self {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            set_cursor: 0,
            cursor: [0; 3],
            place_voxel: 0,
            place_coord: [0; 3],
            remove_voxel: 0,
            remove_coord: [0; 3],
        }
    }
}

type PredicateFunctionV2 =
    unsafe extern "C" fn(*const PredicateContextV2, *mut PredicateResultV2) -> i32;
type OutcomeFunctionV2 = unsafe extern "C" fn(*const OutcomeContextV2, *mut OutcomeResultV2) -> i32;

fn validate_name(kind: &str, name: &str) -> Result<(), String> {
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(format!("invalid {kind} name {name:?}"));
    }
    Ok(())
}

fn validate_parameters(kind: &str, parameters_json: &str) -> Result<(), String> {
    serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(parameters_json)
        .map(|_| ())
        .map_err(|error| format!("invalid {kind} parameters: {error}"))
}

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
        let library = unsafe { Library::new(path) }.map_err(|error| {
            format!("cannot load predicate library {}: {error}", path.display())
        })?;
        validate_library_version(&library)?;
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
        let library = unsafe { Library::new(path) }
            .map_err(|error| format!("cannot load outcome library {}: {error}", path.display()))?;
        validate_library_version(&library)?;
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

fn validate_library_version(library: &Library) -> Result<(), String> {
    let version = unsafe {
        let symbol: Symbol<unsafe extern "C" fn() -> u32> = library
            .get(b"anysearch_extension_abi_version\0")
            .map_err(|error| format!("missing ABI version export: {error}"))?;
        symbol()
    };
    if version != ABI_VERSION {
        return Err(format!(
            "native action ABI {version}, expected {ABI_VERSION}"
        ));
    }
    Ok(())
}
