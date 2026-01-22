//! Common Contract Types
//!
//! Shared types used across all agent contracts.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Agent identification for registration and versioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentIdentity {
    /// Unique agent identifier
    pub id: String,

    /// Semantic version
    pub version: String,

    /// Agent classification
    pub classification: AgentClassification,

    /// Human-readable description
    pub description: String,
}

/// Agent classification per constitution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AgentClassification {
    HypothesisEvaluation,
    ExperimentalMetrics,
}

/// Standard error response for agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentError {
    /// Error code
    pub code: String,

    /// Error message
    pub message: String,

    /// Request ID for correlation
    pub request_id: Uuid,

    /// Additional details
    pub details: Option<serde_json::Value>,
}

/// CLI invocation shape for agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CliInvocation {
    /// Command name
    pub command: String,

    /// Subcommand
    pub subcommand: String,

    /// Input file path or stdin flag
    pub input: CliInput,

    /// Output format
    pub output_format: CliOutputFormat,
}

/// CLI input source.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CliInput {
    /// Read from file
    File(String),
    /// Read from stdin
    Stdin,
    /// Inline JSON
    Inline(String),
}

/// CLI output format.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CliOutputFormat {
    #[default]
    Json,
    Yaml,
    Table,
}

/// Systems that MAY consume agent outputs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Consumer {
    /// LLM-Observatory telemetry system
    LlmObservatory,
    /// LLM-CostOps cost management
    LlmCostOps,
    /// Governance systems
    Governance,
    /// Audit systems
    Audit,
    /// Research dashboards
    ResearchDashboard,
}
