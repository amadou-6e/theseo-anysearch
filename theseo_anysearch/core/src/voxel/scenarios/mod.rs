//! Native scenario extension loading and world-query ABI support.
pub mod abi;
mod loader;
mod query;
pub use loader::{NativeScenarioV2, ScenarioInvocationV2};
