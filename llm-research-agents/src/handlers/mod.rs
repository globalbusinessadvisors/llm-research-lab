//! Edge Function Handlers
//!
//! HTTP handlers for agent endpoints, designed for deployment as Google Cloud Edge Functions.
//!
//! # Constitution Compliance
//!
//! Per PROMPT 2 (RUNTIME & INFRASTRUCTURE IMPLEMENTATION):
//!
//! - Handlers are stateless
//! - Handlers are deterministic
//! - No orchestration logic
//! - No optimization logic
//! - No direct SQL access
//! - Async, non-blocking writes via ruvector-service only
//!
//! # Available Handlers
//!
//! - `HypothesisHandler`: Handles hypothesis evaluation requests
//! - `MetricHandler`: Handles metric computation requests

pub mod hypothesis;
pub mod metric;

pub use hypothesis::*;
pub use metric::*;
