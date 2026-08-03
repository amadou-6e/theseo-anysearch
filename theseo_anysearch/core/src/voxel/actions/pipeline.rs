use serde::Deserialize;

use crate::voxel::world::Coord;

use crate::voxel::{outcomes::NativeOutcomeExtension, predicates::NativePredicateExtension};

#[derive(Clone, Debug, Deserialize)]
pub struct ActionExtensionSpec {
    pub name: String,
    #[serde(default)]
    pub parameters: serde_json::Map<String, serde_json::Value>,
}

pub(crate) enum ConfiguredPredicate {
    ValidAction,
    Bounds,
    Unoccupied,
    Native(NativePredicateExtension),
}

pub(crate) enum ConfiguredOutcome {
    CursorMovement,
    TrailPlacement,
    Place,
    Remove,
    Native(NativeOutcomeExtension),
}

#[derive(Default)]
pub(crate) struct PendingMutations {
    pub(crate) cursor: Option<Coord>,
    pub(crate) place: Option<Coord>,
    pub(crate) remove: Option<Coord>,
}
