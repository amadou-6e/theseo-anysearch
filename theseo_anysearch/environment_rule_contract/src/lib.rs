//! Versioned metadata shared by the AnySearch core and native extension SDK.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

pub const RULE_METADATA_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Copy, Debug, Deserialize, Serialize, Eq, Hash, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RuleKind {
    Predicate,
    Outcome,
    Reward,
    TrainingMetrics,
    EvaluationMetrics,
    Scenario,
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, Hash, PartialEq)]
pub struct RuleReference {
    pub kind: RuleKind,
    pub name: String,
}

impl RuleReference {
    pub fn new(kind: RuleKind, name: impl Into<String>) -> Self {
        Self {
            kind,
            name: name.into(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, Eq, PartialEq)]
pub struct EnvironmentRuleMetadata {
    pub name: String,
    pub kind: RuleKind,
    pub version: u32,
    pub environment_families: Vec<String>,
    pub dependencies: Vec<RuleReference>,
    pub conflicts: Vec<RuleReference>,
}

impl EnvironmentRuleMetadata {
    pub fn new(name: impl Into<String>, kind: RuleKind) -> Self {
        Self {
            name: name.into(),
            kind,
            version: 1,
            environment_families: vec!["voxel".to_owned()],
            dependencies: Vec::new(),
            conflicts: Vec::new(),
        }
    }

    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }

    pub fn with_environment_families(mut self, families: &[&str]) -> Self {
        self.environment_families = families.iter().map(|value| (*value).to_owned()).collect();
        self
    }

    pub fn with_dependencies(mut self, dependencies: &[(&str, RuleKind)]) -> Self {
        self.dependencies = dependencies
            .iter()
            .map(|(name, kind)| RuleReference::new(*kind, *name))
            .collect();
        self
    }

    pub fn with_conflicts(mut self, conflicts: &[(&str, RuleKind)]) -> Self {
        self.conflicts = conflicts
            .iter()
            .map(|(name, kind)| RuleReference::new(*kind, *name))
            .collect();
        self
    }

    pub fn validate(&self) -> Result<(), String> {
        validate_name(&self.name)?;
        if self.version == 0 {
            return Err(format!(
                "{}:{} has version 0",
                kind_name(self.kind),
                self.name
            ));
        }
        if self.environment_families.is_empty() {
            return Err(format!(
                "{}:{} has no supported environment family",
                kind_name(self.kind),
                self.name
            ));
        }
        for family in &self.environment_families {
            if !matches!(family.as_str(), "voxel" | "surface") {
                return Err(format!("unknown environment family {family:?}"));
            }
        }
        for (label, references) in [
            ("dependencies", self.dependencies.as_slice()),
            ("conflicts", self.conflicts.as_slice()),
        ] {
            let mut seen = HashSet::new();
            for reference in references {
                validate_name(&reference.name)?;
                if reference.kind == self.kind && reference.name == self.name {
                    return Err(format!(
                        "{}:{} cannot reference itself",
                        kind_name(self.kind),
                        self.name
                    ));
                }
                if !seen.insert((reference.kind, reference.name.as_str())) {
                    return Err(format!(
                        "{}:{} has duplicate {label}",
                        kind_name(self.kind),
                        self.name
                    ));
                }
            }
        }
        Ok(())
    }
}

fn validate_name(name: &str) -> Result<(), String> {
    let mut characters = name.chars();
    let valid_start = characters
        .next()
        .is_some_and(|value| value == '_' || value.is_ascii_alphabetic());
    if !valid_start || !characters.all(|value| value == '_' || value.is_ascii_alphanumeric()) {
        return Err(format!("invalid environment rule name {name:?}"));
    }
    Ok(())
}

fn kind_name(kind: RuleKind) -> &'static str {
    match kind {
        RuleKind::Predicate => "predicate",
        RuleKind::Outcome => "outcome",
        RuleKind::Reward => "reward",
        RuleKind::TrainingMetrics => "training_metrics",
        RuleKind::EvaluationMetrics => "evaluation_metrics",
        RuleKind::Scenario => "scenario",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metadata() -> EnvironmentRuleMetadata {
        EnvironmentRuleMetadata {
            name: "avoid_collision".to_owned(),
            kind: RuleKind::Predicate,
            version: 2,
            environment_families: vec!["voxel".to_owned()],
            dependencies: vec![RuleReference::new(RuleKind::Predicate, "bounds")],
            conflicts: Vec::new(),
        }
    }

    #[test]
    fn metadata_round_trips_through_json() {
        let metadata = metadata();
        let encoded = serde_json::to_string(&metadata).unwrap();
        let decoded: EnvironmentRuleMetadata = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, metadata);
        decoded.validate().unwrap();
    }

    #[test]
    fn metadata_rejects_self_dependencies() {
        let mut metadata = metadata();
        metadata.dependencies = vec![RuleReference::new(metadata.kind, metadata.name.clone())];
        assert!(metadata
            .validate()
            .unwrap_err()
            .contains("cannot reference itself"));
    }
}
