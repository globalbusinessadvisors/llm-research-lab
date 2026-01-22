//! External Service Clients
//!
//! Client implementations for external services that agents interact with.
//! Per the constitution, all persistence MUST go through ruvector-service.

pub mod ruvector;

pub use ruvector::*;
