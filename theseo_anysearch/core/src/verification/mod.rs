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
