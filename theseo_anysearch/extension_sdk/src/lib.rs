extern crate self as anysearch_extension;

use std::ffi::c_char;
use std::panic::{catch_unwind, AssertUnwindSafe};

pub use anysearch_environment_rule_contract::{
    EnvironmentRuleMetadata, RuleKind, RuleReference, RULE_METADATA_SCHEMA_VERSION,
};
pub use anysearch_extension_macros::{
    anysearch_outcome, anysearch_predicate, anysearch_reward, anysearch_scenario,
    anysearch_scenario_v2,
};

pub const ABI_VERSION: u32 = 2;
pub const MAX_REWARD_COMPONENTS: usize = 8;
pub const MAX_COMPONENT_NAME_BYTES: usize = 63;
pub const MAX_RULE_METADATA_BYTES: usize = 65_536;

/// Serialize rule metadata through the stable variable-length JSON ABI.
///
/// # Safety
/// `output` must reference `output_capacity` writable bytes and
/// `required_length` must be a valid writable pointer.
pub unsafe fn export_rule_metadata_v1(
    output: *mut u8,
    output_capacity: usize,
    required_length: *mut usize,
    metadata: &EnvironmentRuleMetadata,
) -> i32 {
    if required_length.is_null() {
        return 1;
    }
    if metadata.validate().is_err() {
        return 2;
    }
    let encoded = match serde_json::to_vec(metadata) {
        Ok(value) if value.len() <= MAX_RULE_METADATA_BYTES => value,
        _ => return 2,
    };
    *required_length = encoded.len();
    if output.is_null() || output_capacity < encoded.len() {
        return 3;
    }
    std::ptr::copy_nonoverlapping(encoded.as_ptr(), output, encoded.len());
    0
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ScenarioContext {
    pub seed: u64,
    pub episode_index: u64,
    pub scope: String,
    pub grid_size: u16,
    pub filled_voxels: Vec<[i32; 3]>,
    pub action_mode: String,
    pub action_offsets: Vec<[i32; 3]>,
    pub previous_scenario: Option<serde_json::Value>,
    pub curriculum: serde_json::Value,
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct ScenarioResult {
    pub start: [i32; 3],
    #[serde(skip_serializing_if = "Option::is_none")]
    pub goal: Option<[i32; 3]>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub route: Vec<[i32; 3]>,
    pub scenario_id: String,
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl ScenarioResult {
    pub fn goal(start: [i32; 3], goal: [i32; 3], scenario_id: impl Into<String>) -> Self {
        Self {
            start,
            goal: Some(goal),
            route: Vec::new(),
            scenario_id: scenario_id.into(),
            metadata: serde_json::Map::new(),
        }
    }

    pub fn route(start: [i32; 3], route: Vec<[i32; 3]>, scenario_id: impl Into<String>) -> Self {
        Self {
            start,
            goal: None,
            route,
            scenario_id: scenario_id.into(),
            metadata: serde_json::Map::new(),
        }
    }
}

/// Convert an ergonomic scenario function into the variable-length JSON ABI.
///
/// # Safety
/// All pointer/length pairs must identify readable or writable buffers.
pub unsafe fn export_scenario_v1(
    input: *const u8,
    input_len: usize,
    output: *mut u8,
    output_capacity: usize,
    output_len: *mut usize,
    function: fn(&ScenarioContext) -> ScenarioResult,
) -> i32 {
    if input.is_null() || output.is_null() || output_len.is_null() {
        return 1;
    }
    let context: ScenarioContext =
        match serde_json::from_slice(std::slice::from_raw_parts(input, input_len)) {
            Ok(value) => value,
            Err(_) => return 2,
        };
    let result = match catch_unwind(AssertUnwindSafe(|| function(&context))) {
        Ok(value) => value,
        Err(_) => return 3,
    };
    let encoded = match serde_json::to_vec(&result) {
        Ok(value) => value,
        Err(_) => return 4,
    };
    if encoded.len() > output_capacity {
        return 5;
    }
    std::ptr::copy_nonoverlapping(encoded.as_ptr(), output, encoded.len());
    *output_len = encoded.len();
    0
}

pub const WORLD_QUERY_ABI_VERSION: u32 = 1;
pub const SCENARIO_ABI_VERSION_V2: u32 = 2;
pub const MAX_SCENARIO_OUTPUT_BYTES: usize = 1_048_576;

#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueryStatus {
    EmptyOrMiss = 0,
    BlockHit = 1,
    InvalidArgument = 2,
    OutOfBounds = 3,
    InsufficientBuffer = 4,
    StaleToken = 5,
    BackendFailure = 6,
    Unsupported = 7,
    HostFailure = 8,
}
#[repr(i32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScenarioStatusV2 {
    Success = 0,
    InvalidArgument = 1,
    InvalidContext = 2,
    Panic = 3,
    SerializationFailure = 4,
    InsufficientBuffer = 5,
    OutputTooLarge = 6,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorldCoordV1 {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorldBlockV1 {
    pub kind: u8,
    pub active: u8,
    pub reserved: [u8; 2],
    pub reward_weight: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorldRegionEntryV1 {
    pub coordinate: WorldCoordV1,
    pub block: WorldBlockV1,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WorldRayHitV1 {
    pub coordinate: WorldCoordV1,
    pub block: WorldBlockV1,
    pub steps: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorldRayStepV1 {
    pub x: i8,
    pub y: i8,
    pub z: i8,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CandidateV1 {
    pub position: WorldCoordV1,
    pub kind: u8,
    pub reserved: [u8; 3],
    pub quality: f32,
    pub region: WorldCoordV1,
}
pub type CandidateQueryV1 = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    u64,
    u8,
    u64,
    u64,
    WorldCoordV1,
    u32,
    f32,
    u32,
    *mut CandidateV1,
    usize,
    *mut usize,
) -> QueryStatus;
pub type PointQueryV1 = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    u64,
    WorldCoordV1,
    *mut WorldBlockV1,
) -> QueryStatus;
pub type RegionQueryV1 = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    u64,
    WorldCoordV1,
    WorldCoordV1,
    *mut WorldRegionEntryV1,
    usize,
    *mut usize,
) -> QueryStatus;
pub type RayQueryV1 = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    u64,
    WorldCoordV1,
    WorldRayStepV1,
    u32,
    *mut WorldRayHitV1,
) -> QueryStatus;
pub type CountQueryV1 = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    u64,
    WorldCoordV1,
    WorldCoordV1,
    *mut u64,
) -> QueryStatus;
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WorldQueryApiV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub coordinate_size: u32,
    pub block_size: u32,
    pub region_entry_size: u32,
    pub ray_hit_size: u32,
    pub context: *mut std::ffi::c_void,
    pub call_token: u64,
    pub point: Option<PointQueryV1>,
    pub region: Option<RegionQueryV1>,
    pub ray: Option<RayQueryV1>,
    pub count_region: Option<CountQueryV1>,
    pub sample_candidates: Option<CandidateQueryV1>,
}
#[repr(C)]
pub struct ScenarioContextV2Raw {
    pub abi_version: u32,
    pub struct_size: u32,
    pub seed: u64,
    pub episode_index: u64,
    pub grid_size: u32,
    pub scope: *const u8,
    pub scope_len: usize,
    pub action_mode: *const u8,
    pub action_mode_len: usize,
    pub action_offsets_json: *const u8,
    pub action_offsets_json_len: usize,
    pub previous_scenario_json: *const u8,
    pub previous_scenario_json_len: usize,
    pub curriculum_json: *const u8,
    pub curriculum_json_len: usize,
    pub parameters_json: *const u8,
    pub parameters_json_len: usize,
    pub world: *const WorldQueryApiV1,
    pub extent: WorldCoordV1,
    pub world_identity: *const u8,
    pub world_identity_len: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorldQueryError {
    InvalidArgument,
    OutOfBounds,
    InsufficientBuffer,
    StaleToken,
    BackendFailure,
    Unsupported,
    HostFailure,
}
fn query_error(status: QueryStatus) -> WorldQueryError {
    match status {
        QueryStatus::InvalidArgument => WorldQueryError::InvalidArgument,
        QueryStatus::OutOfBounds => WorldQueryError::OutOfBounds,
        QueryStatus::InsufficientBuffer => WorldQueryError::InsufficientBuffer,
        QueryStatus::StaleToken => WorldQueryError::StaleToken,
        QueryStatus::BackendFailure => WorldQueryError::BackendFailure,
        QueryStatus::Unsupported => WorldQueryError::Unsupported,
        _ => WorldQueryError::HostFailure,
    }
}
pub struct WorldQuery<'a> {
    api: &'a WorldQueryApiV1,
}
impl<'a> WorldQuery<'a> {
    fn validate(api: &'a WorldQueryApiV1) -> Result<Self, WorldQueryError> {
        if api.abi_version != WORLD_QUERY_ABI_VERSION
            || (api.struct_size as usize) < std::mem::size_of::<WorldQueryApiV1>()
            || api.coordinate_size as usize != std::mem::size_of::<WorldCoordV1>()
            || api.block_size as usize != std::mem::size_of::<WorldBlockV1>()
            || api.region_entry_size as usize != std::mem::size_of::<WorldRegionEntryV1>()
            || api.ray_hit_size as usize != std::mem::size_of::<WorldRayHitV1>()
        {
            return Err(WorldQueryError::Unsupported);
        }
        Ok(Self { api })
    }
    pub fn point(&self, coordinate: [u32; 3]) -> Result<Option<WorldBlockV1>, WorldQueryError> {
        let function = self.api.point.ok_or(WorldQueryError::Unsupported)?;
        let mut output = WorldBlockV1::default();
        match unsafe {
            function(
                self.api.context,
                self.api.call_token,
                coordinate.into(),
                &mut output,
            )
        } {
            QueryStatus::EmptyOrMiss => Ok(None),
            QueryStatus::BlockHit => Ok(Some(output)),
            status => Err(query_error(status)),
        }
    }
    pub fn region(
        &self,
        minimum: [u32; 3],
        maximum_exclusive: [u32; 3],
    ) -> Result<Vec<WorldRegionEntryV1>, WorldQueryError> {
        let function = self.api.region.ok_or(WorldQueryError::Unsupported)?;
        let mut required = 0;
        let status = unsafe {
            function(
                self.api.context,
                self.api.call_token,
                minimum.into(),
                maximum_exclusive.into(),
                std::ptr::null_mut(),
                0,
                &mut required,
            )
        };
        if status != QueryStatus::InsufficientBuffer
            && !(status == QueryStatus::EmptyOrMiss && required == 0)
        {
            return Err(query_error(status));
        }
        let mut output = vec![WorldRegionEntryV1::default(); required];
        let status = unsafe {
            function(
                self.api.context,
                self.api.call_token,
                minimum.into(),
                maximum_exclusive.into(),
                output.as_mut_ptr(),
                output.len(),
                &mut required,
            )
        };
        match status {
            QueryStatus::EmptyOrMiss | QueryStatus::BlockHit => {
                output.truncate(required);
                Ok(output)
            }
            status => Err(query_error(status)),
        }
    }
    pub fn ray(
        &self,
        origin: [u32; 3],
        step: [i8; 3],
        maximum_steps: u32,
    ) -> Result<Option<WorldRayHitV1>, WorldQueryError> {
        let function = self.api.ray.ok_or(WorldQueryError::Unsupported)?;
        let mut output = WorldRayHitV1::default();
        match unsafe {
            function(
                self.api.context,
                self.api.call_token,
                origin.into(),
                WorldRayStepV1 {
                    x: step[0],
                    y: step[1],
                    z: step[2],
                },
                maximum_steps,
                &mut output,
            )
        } {
            QueryStatus::EmptyOrMiss => Ok(None),
            QueryStatus::BlockHit => Ok(Some(output)),
            status => Err(query_error(status)),
        }
    }
    pub fn count(
        &self,
        minimum: [u32; 3],
        maximum_exclusive: [u32; 3],
    ) -> Result<u64, WorldQueryError> {
        let function = self.api.count_region.ok_or(WorldQueryError::Unsupported)?;
        let mut output = 0;
        match unsafe {
            function(
                self.api.context,
                self.api.call_token,
                minimum.into(),
                maximum_exclusive.into(),
                &mut output,
            )
        } {
            QueryStatus::EmptyOrMiss | QueryStatus::BlockHit => Ok(output),
            status => Err(query_error(status)),
        }
    }

    pub fn candidates(
        &self,
        kind: u8,
        seed: u64,
        stream: u64,
        near: Option<([u32; 3], u32)>,
        minimum_quality: f32,
        maximum_results: u32,
    ) -> Result<Vec<CandidateV1>, WorldQueryError> {
        let function = self
            .api
            .sample_candidates
            .ok_or(WorldQueryError::Unsupported)?;
        let (near, radius) = near.map_or(([0, 0, 0], u32::MAX), |value| value);
        let mut required = 0;
        let status = unsafe {
            function(
                self.api.context,
                self.api.call_token,
                kind,
                seed,
                stream,
                near.into(),
                radius,
                minimum_quality,
                maximum_results,
                std::ptr::null_mut(),
                0,
                &mut required,
            )
        };
        if status != QueryStatus::InsufficientBuffer
            && !(status == QueryStatus::EmptyOrMiss && required == 0)
        {
            return Err(query_error(status));
        }
        let mut output = vec![CandidateV1::default(); required];
        let status = unsafe {
            function(
                self.api.context,
                self.api.call_token,
                kind,
                seed,
                stream,
                near.into(),
                radius,
                minimum_quality,
                maximum_results,
                output.as_mut_ptr(),
                output.len(),
                &mut required,
            )
        };
        match status {
            QueryStatus::EmptyOrMiss | QueryStatus::BlockHit => {
                output.truncate(required);
                Ok(output)
            }
            status => Err(query_error(status)),
        }
    }
}
impl From<[u32; 3]> for WorldCoordV1 {
    fn from(v: [u32; 3]) -> Self {
        Self {
            x: v[0],
            y: v[1],
            z: v[2],
        }
    }
}

pub struct ScenarioContextV2<'a> {
    pub seed: u64,
    pub episode_index: u64,
    pub grid_size: u32,
    pub scope: &'a str,
    pub action_mode: &'a str,
    pub action_offsets: serde_json::Value,
    pub previous_scenario: serde_json::Value,
    pub curriculum: serde_json::Value,
    pub parameters: serde_json::Value,
    pub world: WorldQuery<'a>,
    pub extent: [u32; 3],
    pub world_identity: &'a str,
}
unsafe fn text<'a>(pointer: *const u8, length: usize) -> Result<&'a str, ScenarioStatusV2> {
    if pointer.is_null() && length != 0 {
        return Err(ScenarioStatusV2::InvalidContext);
    }
    let bytes = if length == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(pointer, length)
    };
    std::str::from_utf8(bytes).map_err(|_| ScenarioStatusV2::InvalidContext)
}
unsafe fn context_v2<'a>(
    raw: *const ScenarioContextV2Raw,
) -> Result<ScenarioContextV2<'a>, ScenarioStatusV2> {
    if raw.is_null() {
        return Err(ScenarioStatusV2::InvalidArgument);
    }
    let raw = &*raw;
    if raw.abi_version != SCENARIO_ABI_VERSION_V2
        || (raw.struct_size as usize) < std::mem::size_of::<ScenarioContextV2Raw>()
        || raw.world.is_null()
    {
        return Err(ScenarioStatusV2::InvalidContext);
    }
    let parse = |p, l| {
        serde_json::from_str(unsafe { text(p, l) }?).map_err(|_| ScenarioStatusV2::InvalidContext)
    };
    Ok(ScenarioContextV2 {
        seed: raw.seed,
        episode_index: raw.episode_index,
        grid_size: raw.grid_size,
        scope: text(raw.scope, raw.scope_len)?,
        action_mode: text(raw.action_mode, raw.action_mode_len)?,
        action_offsets: parse(raw.action_offsets_json, raw.action_offsets_json_len)?,
        previous_scenario: parse(raw.previous_scenario_json, raw.previous_scenario_json_len)?,
        curriculum: parse(raw.curriculum_json, raw.curriculum_json_len)?,
        parameters: parse(raw.parameters_json, raw.parameters_json_len)?,
        world: WorldQuery::validate(&*raw.world).map_err(|_| ScenarioStatusV2::InvalidContext)?,
        extent: [raw.extent.x, raw.extent.y, raw.extent.z],
        world_identity: text(raw.world_identity, raw.world_identity_len)?,
    })
}

/// Export a safe scenario-v2 function with bounded two-call length negotiation.
pub unsafe fn export_scenario_v2(
    context: *const ScenarioContextV2Raw,
    output: *mut u8,
    capacity: usize,
    required: *mut usize,
    function: for<'a> fn(&ScenarioContextV2<'a>) -> ScenarioResult,
) -> ScenarioStatusV2 {
    if required.is_null() || (capacity != 0 && output.is_null()) {
        return ScenarioStatusV2::InvalidArgument;
    }
    let encoded = match catch_unwind(AssertUnwindSafe(|| {
        let context = context_v2(context)?;
        serde_json::to_vec(&function(&context)).map_err(|_| ScenarioStatusV2::SerializationFailure)
    })) {
        Ok(Ok(v)) => v,
        Ok(Err(s)) => return s,
        Err(_) => return ScenarioStatusV2::Panic,
    };
    if encoded.len() > MAX_SCENARIO_OUTPUT_BYTES {
        return ScenarioStatusV2::OutputTooLarge;
    }
    required.write(encoded.len());
    if capacity < encoded.len() {
        return ScenarioStatusV2::InsufficientBuffer;
    }
    std::ptr::copy_nonoverlapping(encoded.as_ptr(), output, encoded.len());
    ScenarioStatusV2::Success
}

#[repr(C)]
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
    pub segment_step: u64,
    pub segment_length: u64,
    pub parameters_json: *const u8,
    pub parameters_json_len: usize,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RewardComponentV2 {
    pub name: [c_char; 64],
    pub value: f64,
}

#[repr(C)]
pub struct RewardResultV2 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub mode: u32,
    pub reward: f64,
    pub component_count: u32,
    pub components: [RewardComponentV2; MAX_REWARD_COMPONENTS],
}

#[derive(Clone, Debug)]
pub struct RewardContext {
    pub step: u64,
    pub action_index: i32,
    pub previous_cursor: [i32; 3],
    pub cursor: [i32; 3],
    pub goal: Option<[i32; 3]>,
    pub invalid_action: bool,
    pub collision: bool,
    pub goal_reached: bool,
    pub terminated: bool,
    pub truncated: bool,
    pub consecutive_collisions: u32,
    pub previous_goal_distance: f64,
    pub goal_distance: f64,
    pub standard_reward: f64,
    pub segment_step: u64,
    pub segment_length: u64,
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RewardMode {
    Add,
    Replace,
}

#[derive(Clone, Debug)]
pub struct RewardComponent {
    pub name: String,
    pub value: f64,
}

#[derive(Clone, Debug)]
pub struct RewardResult {
    pub reward: f64,
    pub components: Vec<RewardComponent>,
    pub mode: RewardMode,
}

impl RewardResult {
    pub fn add(reward: f64) -> Self {
        Self {
            reward,
            components: Vec::new(),
            mode: RewardMode::Add,
        }
    }

    pub fn replace(reward: f64) -> Self {
        Self {
            reward,
            components: Vec::new(),
            mode: RewardMode::Replace,
        }
    }

    pub fn with_component(mut self, name: impl Into<String>, value: f64) -> Self {
        self.components.push(RewardComponent {
            name: name.into(),
            value,
        });
        self
    }
}

impl From<&RewardContextV2> for RewardContext {
    fn from(value: &RewardContextV2) -> Self {
        let parameter_bytes = if value.parameters_json.is_null() {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(value.parameters_json, value.parameters_json_len) }
        };
        let parameters = if parameter_bytes.is_empty() {
            serde_json::Map::new()
        } else {
            serde_json::from_slice(parameter_bytes)
                .expect("AnySearch supplied invalid custom reward parameter JSON")
        };
        Self {
            step: value.step,
            action_index: value.action_index,
            previous_cursor: value.previous_cursor,
            cursor: value.cursor,
            goal: (value.has_goal != 0).then_some(value.goal),
            invalid_action: value.invalid_action != 0,
            collision: value.collision != 0,
            goal_reached: value.goal_reached != 0,
            terminated: value.terminated != 0,
            truncated: value.truncated != 0,
            consecutive_collisions: value.consecutive_collisions,
            previous_goal_distance: value.previous_goal_distance,
            goal_distance: value.goal_distance,
            standard_reward: value.standard_reward,
            segment_step: value.segment_step,
            segment_length: value.segment_length,
            parameters,
        }
    }
}

/// Convert one ergonomic reward function into the stable v2 ABI response.
///
/// # Safety
/// `context` and `result` must point to writable/readable v2 ABI structures.
pub unsafe fn export_reward_v2(
    context: *const RewardContextV2,
    result: *mut RewardResultV2,
    reward_function: fn(&RewardContext) -> RewardResult,
) -> i32 {
    if context.is_null() || result.is_null() {
        return 1;
    }
    let context = &*context;
    if context.abi_version != ABI_VERSION
        || context.struct_size as usize != std::mem::size_of::<RewardContextV2>()
    {
        return 2;
    }

    let computed = match catch_unwind(AssertUnwindSafe(|| {
        reward_function(&RewardContext::from(context))
    })) {
        Ok(value) => value,
        Err(_) => return 3,
    };
    if !computed.reward.is_finite() || computed.components.len() > MAX_REWARD_COMPONENTS {
        return 4;
    }

    let raw = &mut *result;
    raw.abi_version = ABI_VERSION;
    raw.struct_size = std::mem::size_of::<RewardResultV2>() as u32;
    raw.mode = match computed.mode {
        RewardMode::Add => 0,
        RewardMode::Replace => 1,
    };
    raw.reward = computed.reward;
    raw.component_count = computed.components.len() as u32;

    for (index, component) in computed.components.iter().enumerate() {
        if !component.value.is_finite()
            || component.name.is_empty()
            || component.name.len() > MAX_COMPONENT_NAME_BYTES
            || !component.name.is_ascii()
        {
            return 5;
        }
        raw.components[index].name = [0; 64];
        for (target, source) in raw.components[index]
            .name
            .iter_mut()
            .zip(component.name.as_bytes())
        {
            *target = *source as c_char;
        }
        raw.components[index].value = component.value;
    }
    0
}

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

#[derive(Clone, Debug)]
pub struct ActionHistoryEntry {
    pub action_index: i32,
    pub previous_cursor: [i32; 3],
    pub cursor: [i32; 3],
    pub feasible: bool,
    pub collision: bool,
}

#[derive(Clone, Debug)]
pub struct PredicateContext {
    pub step: u64,
    pub grid_size: u16,
    pub action_index: i32,
    pub cursor: [i32; 3],
    pub destination: [i32; 3],
    pub goal: Option<[i32; 3]>,
    pub valid_action: bool,
    pub destination_in_bounds: bool,
    pub destination_blocked: bool,
    pub observation_filled: usize,
    pub observation_steps_remaining: u32,
    pub observation_goal_distance: Option<u32>,
    pub history: Vec<ActionHistoryEntry>,
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Copy, Debug)]
pub struct PredicateResult {
    pub feasible: bool,
}
impl PredicateResult {
    pub fn allow() -> Self {
        Self { feasible: true }
    }
    pub fn deny() -> Self {
        Self { feasible: false }
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

#[derive(Clone, Debug)]
pub struct OutcomeContext {
    pub step: u64,
    pub grid_size: u16,
    pub action_index: i32,
    pub cursor: [i32; 3],
    pub destination: [i32; 3],
    pub goal: Option<[i32; 3]>,
    pub history: Vec<ActionHistoryEntry>,
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

#[derive(Clone, Debug, Default)]
pub struct OutcomeMutations {
    pub cursor: Option<[i32; 3]>,
    pub place: Option<[i32; 3]>,
    pub remove: Option<[i32; 3]>,
}
impl OutcomeMutations {
    pub fn set_cursor(&mut self, coord: [i32; 3]) {
        self.cursor = Some(coord);
    }
    pub fn place_voxel(&mut self, coord: [i32; 3]) {
        self.place = Some(coord);
    }
    pub fn remove_voxel(&mut self, coord: [i32; 3]) {
        self.remove = Some(coord);
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OutcomeResult;
impl OutcomeResult {
    pub fn applied() -> Self {
        Self
    }
}

fn parameters(pointer: *const u8, length: usize) -> serde_json::Map<String, serde_json::Value> {
    if pointer.is_null() || length == 0 {
        return serde_json::Map::new();
    }
    let bytes = unsafe { std::slice::from_raw_parts(pointer, length) };
    serde_json::from_slice(bytes)
        .expect("AnySearch supplied invalid action extension parameter JSON")
}
fn history(pointer: *const ActionHistoryEntryV2, length: usize) -> Vec<ActionHistoryEntry> {
    if pointer.is_null() || length == 0 {
        return Vec::new();
    }
    unsafe { std::slice::from_raw_parts(pointer, length) }
        .iter()
        .map(|entry| ActionHistoryEntry {
            action_index: entry.action_index,
            previous_cursor: entry.previous_cursor,
            cursor: entry.cursor,
            feasible: entry.feasible != 0,
            collision: entry.collision != 0,
        })
        .collect()
}

pub unsafe fn export_predicate_v2(
    context: *const PredicateContextV2,
    result: *mut PredicateResultV2,
    function: fn(&PredicateContext) -> PredicateResult,
) -> i32 {
    if context.is_null() || result.is_null() {
        return 1;
    }
    let raw = &*context;
    if raw.abi_version != ABI_VERSION
        || raw.struct_size as usize != std::mem::size_of::<PredicateContextV2>()
    {
        return 2;
    }
    let context = PredicateContext {
        step: raw.step,
        grid_size: raw.grid_size,
        action_index: raw.action_index,
        cursor: raw.cursor,
        destination: raw.destination,
        goal: (raw.has_goal != 0).then_some(raw.goal),
        valid_action: raw.valid_action != 0,
        destination_in_bounds: raw.destination_in_bounds != 0,
        destination_blocked: raw.destination_blocked != 0,
        observation_filled: raw.observation_filled,
        observation_steps_remaining: raw.observation_steps_remaining,
        observation_goal_distance: (raw.has_observation_goal_distance != 0)
            .then_some(raw.observation_goal_distance),
        history: history(raw.history, raw.history_len),
        parameters: parameters(raw.parameters_json, raw.parameters_json_len),
    };
    let computed = match catch_unwind(AssertUnwindSafe(|| function(&context))) {
        Ok(value) => value,
        Err(_) => return 3,
    };
    *result = PredicateResultV2 {
        abi_version: ABI_VERSION,
        struct_size: std::mem::size_of::<PredicateResultV2>() as u32,
        feasible: u8::from(computed.feasible),
    };
    0
}

pub unsafe fn export_outcome_v2(
    context: *const OutcomeContextV2,
    result: *mut OutcomeResultV2,
    function: fn(&OutcomeContext, &mut OutcomeMutations) -> OutcomeResult,
) -> i32 {
    if context.is_null() || result.is_null() {
        return 1;
    }
    let raw = &*context;
    if raw.abi_version != ABI_VERSION
        || raw.struct_size as usize != std::mem::size_of::<OutcomeContextV2>()
    {
        return 2;
    }
    let context = OutcomeContext {
        step: raw.step,
        grid_size: raw.grid_size,
        action_index: raw.action_index,
        cursor: raw.cursor,
        destination: raw.destination,
        goal: (raw.has_goal != 0).then_some(raw.goal),
        history: history(raw.history, raw.history_len),
        parameters: parameters(raw.parameters_json, raw.parameters_json_len),
    };
    let mut mutations = OutcomeMutations::default();
    if catch_unwind(AssertUnwindSafe(|| function(&context, &mut mutations))).is_err() {
        return 3;
    }
    *result = OutcomeResultV2 {
        abi_version: ABI_VERSION,
        struct_size: std::mem::size_of::<OutcomeResultV2>() as u32,
        set_cursor: u8::from(mutations.cursor.is_some()),
        cursor: mutations.cursor.unwrap_or([0; 3]),
        place_voxel: u8::from(mutations.place.is_some()),
        place_coord: mutations.place.unwrap_or([0; 3]),
        remove_voxel: u8::from(mutations.remove.is_some()),
        remove_coord: mutations.remove.unwrap_or([0; 3]),
    };
    0
}
#[cfg(test)]
mod tests {
    use super::*;

    #[anysearch_reward]
    fn collision_penalty(context: &RewardContext) -> RewardResult {
        let penalty = if context.collision { -0.25 } else { 0.0 };
        RewardResult::add(penalty).with_component("collision_penalty", penalty)
    }

    #[anysearch_predicate(
        version = 2,
        environment_families = "voxel,surface",
        dependencies = "predicate:bounds",
        conflicts = "outcome:remove"
    )]
    fn deny_blocked(context: &PredicateContext) -> PredicateResult {
        if context.destination_blocked {
            PredicateResult::deny()
        } else {
            PredicateResult::allow()
        }
    }

    #[anysearch_outcome]
    fn place_destination(
        context: &OutcomeContext,
        mutations: &mut OutcomeMutations,
    ) -> OutcomeResult {
        mutations.set_cursor(context.destination);
        mutations.place_voxel(context.destination);
        OutcomeResult::applied()
    }

    #[anysearch_scenario]
    fn adjacent_scenario(context: &ScenarioContext) -> ScenarioResult {
        ScenarioResult::goal(
            [4, 4, 4],
            [5, 4, 4],
            format!("episode-{}", context.episode_index),
        )
    }

    unsafe extern "C" fn test_point(
        _: *mut std::ffi::c_void,
        _: u64,
        coordinate: WorldCoordV1,
        output: *mut WorldBlockV1,
    ) -> QueryStatus {
        if output.is_null() {
            return QueryStatus::InvalidArgument;
        }
        if coordinate == (WorldCoordV1 { x: 2, y: 2, z: 2 }) {
            output.write(WorldBlockV1 {
                kind: 1,
                active: 1,
                reserved: [0; 2],
                reward_weight: 0.0,
            });
            QueryStatus::BlockHit
        } else {
            QueryStatus::EmptyOrMiss
        }
    }
    #[anysearch_scenario_v2]
    fn queried_scenario(context: &ScenarioContextV2<'_>) -> ScenarioResult {
        assert!(context.world.point([2, 2, 2]).unwrap().is_some());
        ScenarioResult::goal([1, 1, 1], [2, 2, 2], "v2")
    }

    fn raw_v2_context(api: &WorldQueryApiV1) -> ScenarioContextV2Raw {
        let json = b"null";
        ScenarioContextV2Raw {
            abi_version: SCENARIO_ABI_VERSION_V2,
            struct_size: std::mem::size_of::<ScenarioContextV2Raw>() as u32,
            seed: 1,
            episode_index: 2,
            grid_size: 8,
            scope: b"training".as_ptr(),
            scope_len: 8,
            action_mode: b"discrete_26".as_ptr(),
            action_mode_len: 11,
            action_offsets_json: b"[]".as_ptr(),
            action_offsets_json_len: 2,
            previous_scenario_json: json.as_ptr(),
            previous_scenario_json_len: json.len(),
            curriculum_json: b"{}".as_ptr(),
            curriculum_json_len: 2,
            parameters_json: b"{}".as_ptr(),
            parameters_json_len: 2,
            world: api,
            extent: WorldCoordV1 { x: 8, y: 8, z: 8 },
            world_identity: std::ptr::null(),
            world_identity_len: 0,
        }
    }

    #[test]
    fn scenario_v2_macro_negotiates_exact_output_and_validates_layout() {
        let api = WorldQueryApiV1 {
            abi_version: WORLD_QUERY_ABI_VERSION,
            struct_size: std::mem::size_of::<WorldQueryApiV1>() as u32,
            coordinate_size: std::mem::size_of::<WorldCoordV1>() as u32,
            block_size: std::mem::size_of::<WorldBlockV1>() as u32,
            region_entry_size: std::mem::size_of::<WorldRegionEntryV1>() as u32,
            ray_hit_size: std::mem::size_of::<WorldRayHitV1>() as u32,
            context: 1usize as *mut _,
            call_token: 1,
            point: Some(test_point),
            region: None,
            ray: None,
            count_region: None,
            sample_candidates: None,
        };
        let context = raw_v2_context(&api);
        let mut required = 0;
        assert_eq!(
            unsafe {
                anysearch_scenario_queried_scenario_v2(
                    &context,
                    std::ptr::null_mut(),
                    0,
                    &mut required,
                )
            },
            ScenarioStatusV2::InsufficientBuffer
        );
        let mut output = vec![0; required];
        let mut written = 0;
        assert_eq!(
            unsafe {
                anysearch_scenario_queried_scenario_v2(
                    &context,
                    output.as_mut_ptr(),
                    output.len(),
                    &mut written,
                )
            },
            ScenarioStatusV2::Success
        );
        assert_eq!(written, required);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&output).unwrap()["scenario_id"],
            "v2"
        );
        let mut short_required = 0;
        let mut short = vec![0; required - 1];
        assert_eq!(
            unsafe {
                anysearch_scenario_queried_scenario_v2(
                    &context,
                    short.as_mut_ptr(),
                    short.len(),
                    &mut short_required,
                )
            },
            ScenarioStatusV2::InsufficientBuffer
        );
        assert_eq!(short_required, required);
        let mut invalid = raw_v2_context(&api);
        invalid.struct_size = 0;
        assert_eq!(
            unsafe {
                anysearch_scenario_queried_scenario_v2(
                    &invalid,
                    std::ptr::null_mut(),
                    0,
                    &mut required,
                )
            },
            ScenarioStatusV2::InvalidContext
        );
        let unsupported = WorldQueryApiV1 {
            abi_version: 99,
            ..api
        };
        let invalid_api = raw_v2_context(&unsupported);
        assert_eq!(
            unsafe {
                anysearch_scenario_queried_scenario_v2(
                    &invalid_api,
                    std::ptr::null_mut(),
                    0,
                    &mut required,
                )
            },
            ScenarioStatusV2::InvalidContext
        );
    }

    #[test]
    fn scenario_v2_panics_do_not_cross_ffi() {
        fn panic_scenario(_: &ScenarioContextV2<'_>) -> ScenarioResult {
            panic!("expected")
        }
        let api = WorldQueryApiV1 {
            abi_version: WORLD_QUERY_ABI_VERSION,
            struct_size: std::mem::size_of::<WorldQueryApiV1>() as u32,
            coordinate_size: std::mem::size_of::<WorldCoordV1>() as u32,
            block_size: std::mem::size_of::<WorldBlockV1>() as u32,
            region_entry_size: std::mem::size_of::<WorldRegionEntryV1>() as u32,
            ray_hit_size: std::mem::size_of::<WorldRayHitV1>() as u32,
            context: 1usize as *mut _,
            call_token: 1,
            point: Some(test_point),
            region: None,
            ray: None,
            count_region: None,
            sample_candidates: None,
        };
        let context = raw_v2_context(&api);
        let mut required = 0;
        assert_eq!(
            unsafe {
                export_scenario_v2(
                    &context,
                    std::ptr::null_mut(),
                    0,
                    &mut required,
                    panic_scenario,
                )
            },
            ScenarioStatusV2::Panic
        );
    }

    #[test]
    fn scenario_v2_rejects_oversized_output_and_layout_is_stable() {
        fn huge(_: &ScenarioContextV2<'_>) -> ScenarioResult {
            let mut result = ScenarioResult::goal([1, 1, 1], [2, 2, 2], "huge");
            result.metadata.insert(
                "payload".into(),
                serde_json::Value::String("x".repeat(MAX_SCENARIO_OUTPUT_BYTES)),
            );
            result
        }
        let api = WorldQueryApiV1 {
            abi_version: WORLD_QUERY_ABI_VERSION,
            struct_size: std::mem::size_of::<WorldQueryApiV1>() as u32,
            coordinate_size: std::mem::size_of::<WorldCoordV1>() as u32,
            block_size: std::mem::size_of::<WorldBlockV1>() as u32,
            region_entry_size: std::mem::size_of::<WorldRegionEntryV1>() as u32,
            ray_hit_size: std::mem::size_of::<WorldRayHitV1>() as u32,
            context: 1usize as *mut _,
            call_token: 1,
            point: Some(test_point),
            region: None,
            ray: None,
            count_region: None,
            sample_candidates: None,
        };
        let context = raw_v2_context(&api);
        let mut required = 0;
        assert_eq!(
            unsafe { export_scenario_v2(&context, std::ptr::null_mut(), 0, &mut required, huge) },
            ScenarioStatusV2::OutputTooLarge
        );
        assert_eq!(std::mem::size_of::<WorldCoordV1>(), 12);
        assert_eq!(std::mem::size_of::<WorldBlockV1>(), 8);
        assert_eq!(std::mem::size_of::<WorldRayStepV1>(), 3);
    }

    #[test]
    fn scenario_attribute_exports_json_result() {
        let input = br#"{"seed":42,"episode_index":7,"scope":"evaluation","grid_size":8,"filled_voxels":[],"action_mode":"discrete_26","action_offsets":[[1,0,0]],"previous_scenario":null,"curriculum":{},"parameters":{}}"#;
        let mut output = [0_u8; 1024];
        let mut output_len = 0_usize;
        let status = unsafe {
            anysearch_scenario_adjacent_scenario_v1(
                input.as_ptr(),
                input.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut output_len,
            )
        };
        assert_eq!(status, 0);
        let value: serde_json::Value = serde_json::from_slice(&output[..output_len]).unwrap();
        assert_eq!(value["scenario_id"], "episode-7");
        assert_eq!(value["goal"], serde_json::json!([5, 4, 4]));
    }

    #[test]
    fn predicate_and_outcome_attributes_export_v2_symbols() {
        let raw_history = [ActionHistoryEntryV2 {
            action_index: 2,
            feasible: 0,
            collision: 1,
            ..Default::default()
        }];
        let predicate_context = PredicateContextV2 {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<PredicateContextV2>() as u32,
            step: 4,
            grid_size: 8,
            action_index: 2,
            cursor: [2, 2, 2],
            destination: [3, 2, 2],
            goal: [6, 6, 6],
            has_goal: 1,
            valid_action: 1,
            destination_in_bounds: 1,
            destination_blocked: 1,
            observation_filled: 3,
            observation_steps_remaining: 6,
            observation_goal_distance: 11,
            has_observation_goal_distance: 1,
            history: raw_history.as_ptr(),
            history_len: raw_history.len(),
            parameters_json: std::ptr::null(),
            parameters_json_len: 0,
        };
        let mut predicate_result = PredicateResultV2 {
            abi_version: 0,
            struct_size: 0,
            feasible: 1,
        };
        assert_eq!(
            unsafe {
                anysearch_predicate_deny_blocked_v2(&predicate_context, &mut predicate_result)
            },
            0
        );
        assert_eq!(predicate_result.feasible, 0);

        let outcome_context = OutcomeContextV2 {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<OutcomeContextV2>() as u32,
            step: 4,
            grid_size: 8,
            action_index: 2,
            cursor: [2, 2, 2],
            destination: [3, 2, 2],
            goal: [6, 6, 6],
            has_goal: 1,
            history: raw_history.as_ptr(),
            history_len: raw_history.len(),
            parameters_json: std::ptr::null(),
            parameters_json_len: 0,
        };
        let mut outcome_result = OutcomeResultV2 {
            abi_version: 0,
            struct_size: 0,
            set_cursor: 0,
            cursor: [0; 3],
            place_voxel: 0,
            place_coord: [0; 3],
            remove_voxel: 0,
            remove_coord: [0; 3],
        };
        assert_eq!(
            unsafe {
                anysearch_outcome_place_destination_v2(&outcome_context, &mut outcome_result)
            },
            0
        );
        assert_eq!(outcome_result.cursor, [3, 2, 2]);
        assert_eq!(outcome_result.place_coord, [3, 2, 2]);
    }

    #[test]
    fn attribute_exports_typed_rule_metadata() {
        let mut output = vec![0; MAX_RULE_METADATA_BYTES];
        let mut length = 0;

        let status = unsafe {
            anysearch_rule_metadata_predicate_deny_blocked_v1(
                output.as_mut_ptr(),
                output.len(),
                &mut length,
            )
        };

        assert_eq!(status, 0);
        let metadata: EnvironmentRuleMetadata = serde_json::from_slice(&output[..length]).unwrap();
        assert_eq!(metadata.name, "deny_blocked");
        assert_eq!(metadata.kind, RuleKind::Predicate);
        assert_eq!(metadata.version, 2);
        assert_eq!(metadata.environment_families, ["voxel", "surface"]);
        assert_eq!(
            metadata.dependencies,
            [RuleReference::new(RuleKind::Predicate, "bounds")]
        );
        assert_eq!(
            metadata.conflicts,
            [RuleReference::new(RuleKind::Outcome, "remove")]
        );
    }
    #[test]
    fn attribute_generates_and_encodes_v2_export() {
        let parameters = br#"{"scale":2.5}"#;
        let context = RewardContextV2 {
            abi_version: ABI_VERSION,
            struct_size: std::mem::size_of::<RewardContextV2>() as u32,
            step: 3,
            action_index: 4,
            previous_cursor: [1, 1, 1],
            cursor: [1, 1, 1],
            goal: [5, 5, 5],
            has_goal: 1,
            invalid_action: 0,
            collision: 1,
            goal_reached: 0,
            terminated: 0,
            truncated: 0,
            consecutive_collisions: 2,
            previous_goal_distance: 7.0,
            goal_distance: 7.0,
            standard_reward: -0.01,
            segment_step: 3,
            segment_length: 7,
            parameters_json: parameters.as_ptr(),
            parameters_json_len: parameters.len(),
        };
        assert_eq!(
            RewardContext::from(&context).parameters["scale"].as_f64(),
            Some(2.5)
        );
        let mut result: RewardResultV2 = unsafe { std::mem::zeroed() };

        let status = unsafe { anysearch_reward_collision_penalty_v2(&context, &mut result) };

        assert_eq!(status, 0);
        assert_eq!(result.abi_version, ABI_VERSION);
        assert_eq!(result.mode, 0);
        assert_eq!(result.reward, -0.25);
        assert_eq!(result.component_count, 1);
        assert_eq!(result.components[0].value, -0.25);
        let name = result.components[0]
            .name
            .iter()
            .take_while(|byte| **byte != 0)
            .map(|byte| *byte as u8)
            .collect::<Vec<_>>();
        assert_eq!(name, b"collision_penalty");
    }
}
