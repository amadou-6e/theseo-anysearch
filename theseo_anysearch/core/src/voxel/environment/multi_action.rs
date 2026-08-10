//! Per-agent action pipelines used by heterogeneous voxel environments.

use std::path::Path;

use serde::Deserialize;

use crate::voxel::{
    actions::{ActionExtensionSpec, ActionHistoryEntryV2, ConfiguredOutcome, ConfiguredPredicate},
    outcomes::NativeOutcomeExtension,
    predicates::NativePredicateExtension,
    world::Coord,
};

#[derive(Debug, Deserialize)]
pub struct AgentPipelineSpec {
    pub id: String,
    pub action_predicates: Vec<ActionExtensionSpec>,
    pub action_outcomes: Vec<ActionExtensionSpec>,
    #[serde(default = "default_history_length")]
    pub action_history_length: usize,
    #[serde(default)]
    pub start: Option<[u16; 3]>,
}

fn default_history_length() -> usize {
    16
}

pub struct AgentActionPipeline {
    pub id: String,
    pub start: Option<Coord>,
    pub predicates: Vec<ConfiguredPredicate>,
    pub outcomes: Vec<ConfiguredOutcome>,
    pub history: Vec<ActionHistoryEntryV2>,
    pub history_length: usize,
}

impl AgentActionPipeline {
    pub fn standard(id: String, trail_mode: bool) -> Self {
        let mut outcomes = vec![ConfiguredOutcome::CursorMovement];
        if trail_mode {
            outcomes.push(ConfiguredOutcome::TrailPlacement);
        }
        Self {
            id,
            start: None,
            predicates: vec![
                ConfiguredPredicate::ValidAction,
                ConfiguredPredicate::Bounds,
                ConfiguredPredicate::Unoccupied,
            ],
            outcomes,
            history: Vec::new(),
            history_length: 16,
        }
    }

    pub fn load(spec: AgentPipelineSpec, native_library: Option<&Path>) -> Result<Self, String> {
        let predicates = spec
            .action_predicates
            .into_iter()
            .map(|selector| load_predicate(selector, native_library))
            .collect::<Result<Vec<_>, _>>()?;
        let outcomes = spec
            .action_outcomes
            .into_iter()
            .map(|selector| load_outcome(selector, native_library))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            id: spec.id,
            start: spec.start.map(|value| (value[0], value[1], value[2])),
            predicates,
            outcomes,
            history: Vec::new(),
            history_length: spec.action_history_length,
        })
    }
}

fn load_predicate(
    spec: ActionExtensionSpec,
    library: Option<&Path>,
) -> Result<ConfiguredPredicate, String> {
    let parameters = serde_json::to_string(&spec.parameters).expect("JSON map serializes");
    match spec.name.as_str() {
        "valid_action" => Ok(ConfiguredPredicate::ValidAction),
        "bounds" => Ok(ConfiguredPredicate::Bounds),
        "unoccupied" => Ok(ConfiguredPredicate::Unoccupied),
        _ => NativePredicateExtension::load(
            library.ok_or_else(|| format!("unknown action predicate {:?}", spec.name))?,
            &spec.name,
            parameters,
        )
        .map(ConfiguredPredicate::Native),
    }
}

fn load_outcome(
    spec: ActionExtensionSpec,
    library: Option<&Path>,
) -> Result<ConfiguredOutcome, String> {
    let parameters = serde_json::to_string(&spec.parameters).expect("JSON map serializes");
    match spec.name.as_str() {
        "cursor_movement" => Ok(ConfiguredOutcome::CursorMovement),
        "trail_placement" => Ok(ConfiguredOutcome::TrailPlacement),
        "place" => Ok(ConfiguredOutcome::Place),
        "remove" => Ok(ConfiguredOutcome::Remove),
        _ => NativeOutcomeExtension::load(
            library.ok_or_else(|| format!("unknown action outcome {:?}", spec.name))?,
            &spec.name,
            parameters,
        )
        .map(ConfiguredOutcome::Native),
    }
}
