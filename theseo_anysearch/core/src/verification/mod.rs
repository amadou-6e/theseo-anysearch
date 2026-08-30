//! Reusable parity, fault-injection, and benchmark adapters.
//!
//! These adapters deliberately depend on the regional world traits rather
//! than concrete storage implementations so later residency and overlay
//! backends can join the same verification matrix.

pub mod benchmark;
pub mod fault;
pub mod parity;

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PendingFaultCase {
    ChecksumMismatch,
    ShortRead,
    CandidateIndexCorruption,
    BudgetExhaustion,
    CacheEviction,
    PinnedOvercommit,
    FailedPrefetch,
}

impl PendingFaultCase {
    pub const fn dependency_issues(self) -> &'static [u32] {
        match self {
            Self::ChecksumMismatch | Self::ShortRead => &[224],
            Self::CandidateIndexCorruption | Self::BudgetExhaustion => &[223],
            Self::CacheEviction | Self::PinnedOvercommit | Self::FailedPrefetch => &[224],
        }
    }
}
