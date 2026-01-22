//! LLM-Research-Lab Agents
//!
//! This module contains the core agent implementations for hypothesis evaluation
//! and experimental metrics computation.
//!
//! # Constitution Compliance
//!
//! All agents in this module adhere to the LLM-RESEARCH-LAB AGENT INFRASTRUCTURE
//! CONSTITUTION defined in PROMPT 0.
//!
//! All agents:
//! - Are stateless at runtime
//! - Deploy as Google Cloud Edge Functions
//! - Persist data ONLY via ruvector-service
//! - Emit exactly ONE DecisionEvent per invocation
//! - NEVER execute SQL directly
//!
//! # Agent Types
//!
//! - `HypothesisAgent`: Hypothesis evaluation and validation
//! - `ExperimentalMetricAgent`: Experimental metrics computation

pub mod hypothesis;
pub mod metric_agent;
pub mod telemetry;
pub mod traits;

pub use hypothesis::*;
pub use metric_agent::*;
pub use telemetry::*;
pub use traits::*;
