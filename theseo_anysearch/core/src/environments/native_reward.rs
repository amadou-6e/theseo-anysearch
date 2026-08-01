use libloading::{Library, Symbol};
use std::collections::HashMap;
use std::ffi::c_char;
use std::path::Path;

pub const ABI_VERSION: u32 = 2;
pub const MAX_COMPONENTS: usize = 8;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RewardContextV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub step: u64,
    pub action_index: i32,
    pub previous_cursor: [i32; 3],
    pub cursor: [i32; 3],
    pub goal: [i32; 3],
    pub has_goal: u8,
    pub invalid_action: u8,
    pub collision: u8,
    pub goal_reached: u8,
    pub terminated: u8,
    pub truncated: u8,
    pub consecutive_collisions: u32,
    pub previous_goal_distance: f64,
    pub goal_distance: f64,
    pub standard_reward: f64,
    pub parameters_json: *const u8,
    pub parameters_json_len: usize,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RewardComponentV2 {
    pub name: [c_char; 64],
    pub value: f64,
}

impl Default for RewardComponentV2 {
    fn default() -> Self {
        Self {
            name: [0; 64],
            value: 0.0,
        }
    }
}

#[repr(C)]
pub struct RewardResultV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub mode: u32,
    pub reward: f64,
    pub component_count: u32,
    pub components: [RewardComponentV2; MAX_COMPONENTS],
}

impl Default for RewardResultV2 {
    fn default() -> Self {
        Self {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<Self>() as u32,
            mode: 0,
            reward: 0.0,
            component_count: 0,
            components: [RewardComponentV2::default(); MAX_COMPONENTS],
        }
    }
}

type RewardFunctionV2 = unsafe extern "C" fn(*const RewardContextV2, *mut RewardResultV2) -> i32;

pub struct NativeRewardExtension {
    _library: Library,
    function: RewardFunctionV2,
    pub name: String,
    parameters_json: Vec<u8>,
}

impl NativeRewardExtension {
    pub fn load(path: &Path, name: &str, parameters_json: String) -> Result<Self, String> {
        if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
            return Err(format!("invalid reward name {name:?}"));
        }
        serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&parameters_json)
            .map_err(|error| format!("invalid custom reward parameters: {error}"))?;
        let library = unsafe { Library::new(path) }.map_err(|error| {
            format!(
                "cannot load native reward library {}: {error}",
                path.display()
            )
        })?;
        let version = unsafe {
            let symbol: Symbol<unsafe extern "C" fn() -> u32> = library
                .get(b"anysearch_extension_abi_version\0")
                .map_err(|error| format!("missing ABI version export: {error}"))?;
            symbol()
        };
        if version != ABI_VERSION {
            return Err(format!(
                "native reward ABI {version}, expected {ABI_VERSION}"
            ));
        }
        let symbol_name = format!("anysearch_reward_{name}_v2\0");
        let function = unsafe {
            let symbol: Symbol<RewardFunctionV2> = library
                .get(symbol_name.as_bytes())
                .map_err(|error| format!("reward {name:?} is not exported: {error}"))?;
            *symbol
        };
        Ok(Self {
            _library: library,
            function,
            name: name.to_owned(),
            parameters_json: parameters_json.into_bytes(),
        })
    }

    pub fn compute(
        &self,
        mut context: RewardContextV2,
        built_in: &HashMap<String, f32>,
    ) -> Result<(f32, HashMap<String, f32>), String> {
        context.parameters_json = self.parameters_json.as_ptr();
        context.parameters_json_len = self.parameters_json.len();
        let mut result = RewardResultV2::default();
        let status = unsafe { (self.function)(&context, &mut result) };
        if status != 0 {
            return Err(format!(
                "native reward {:?} returned status {status}",
                self.name
            ));
        }
        if result.mode > 1 || result.component_count as usize > MAX_COMPONENTS {
            return Err("native reward returned an invalid mode or component count".to_owned());
        }
        if !result.reward.is_finite() {
            return Err("native reward returned a non-finite value".to_owned());
        }
        let mut custom = HashMap::new();
        for component in result
            .components
            .iter()
            .take(result.component_count as usize)
        {
            let bytes: Vec<u8> = component
                .name
                .iter()
                .map(|value| *value as u8)
                .take_while(|value| *value != 0)
                .collect();
            let name =
                String::from_utf8(bytes).map_err(|_| "reward component name is not UTF-8")?;
            if name.is_empty() || !name.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'_') {
                return Err(format!("invalid reward component name {name:?}"));
            }
            if built_in.contains_key(&name)
                || custom
                    .insert(name.clone(), component.value as f32)
                    .is_some()
            {
                return Err(format!("duplicate reward component {name:?}"));
            }
            if !component.value.is_finite() {
                return Err(format!("reward component {name:?} is not finite"));
            }
        }
        let component_sum: f64 = custom.values().map(|value| f64::from(*value)).sum();
        if !custom.is_empty() && (component_sum - result.reward).abs() > 1e-6 {
            return Err("native reward components do not sum to its reward".to_owned());
        }
        if custom.is_empty() {
            custom.insert(self.name.clone(), result.reward as f32);
        }
        if result.mode == 1 {
            Ok((result.reward as f32, custom))
        } else {
            let mut breakdown = built_in.clone();
            breakdown.extend(custom);
            Ok((
                built_in.values().sum::<f32>() + result.reward as f32,
                breakdown,
            ))
        }
    }
}
