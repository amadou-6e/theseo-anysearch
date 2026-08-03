use std::path::Path;

use libloading::{Library, Symbol};

use crate::voxel::common::{load_library, validate_name, validate_parameters};

use super::{
    abi::{RewardContextV2, RewardResultV2, MAX_COMPONENTS},
    breakdown::RewardBreakdown,
};

type RewardFunctionV2 = unsafe extern "C" fn(*const RewardContextV2, *mut RewardResultV2) -> i32;

pub struct NativeRewardExtension {
    _library: Library,
    function: RewardFunctionV2,
    pub name: String,
    parameters_json: Vec<u8>,
}

impl NativeRewardExtension {
    pub fn load(path: &Path, name: &str, parameters_json: String) -> Result<Self, String> {
        validate_name("reward", name)?;
        validate_parameters("reward", &parameters_json)?;
        let library = load_library(path, "reward")?;
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
        built_in: &RewardBreakdown,
    ) -> Result<(f32, RewardBreakdown), String> {
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
        let mut custom = RewardBreakdown::new();
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
