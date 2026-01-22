//! LLM-Research-Lab Agent Infrastructure
//!
//! This crate provides the agent infrastructure for hypothesis evaluation and
//! experimental metrics computation within the LLM-Research-Lab platform.
//!
//! # Constitution Compliance
//!
//! This crate implements agents according to the LLM-RESEARCH-LAB AGENT
//! INFRASTRUCTURE CONSTITUTION (PROMPT 0). All agents:
//!
//! - Execute as Google Cloud Edge Functions
//! - Are stateless at runtime
//! - Have no local persistence
//! - Persist data only via ruvector-service
//! - Import schemas exclusively from agentics-contracts
//! - Validate all inputs and outputs against contracts
//! - Emit telemetry compatible with LLM-Observatory
//! - Emit exactly ONE DecisionEvent per invocation
//! - Expose CLI-invokable endpoints
//! - Return deterministic, machine-readable output
//!
//! # Repository Role
//!
//! LLM-Research-Lab is the PRIMARY EXPERIMENTATION, HYPOTHESIS TESTING, AND
//! RESEARCH SIGNAL GENERATION layer. It:
//!
//! - Defines and evaluates research hypotheses
//! - Computes experimental metrics and evaluation signals
//! - Supports controlled experimentation and analysis
//! - Produces structured research artifacts and findings
//! - Acts as the authoritative source of experimental insight
//!
//! It operates OUTSIDE the critical execution path. It:
//!
//! - Does NOT intercept runtime traffic
//! - Does NOT execute production workflows
//! - Does NOT route inference requests
//! - Does NOT enforce policies or decisions
//! - Does NOT optimize live configurations
//!
//! # Agent Types
//!
//! ## Hypothesis Agent
//!
//! - **Classification**: HYPOTHESIS EVALUATION
//! - **decision_type**: "hypothesis_evaluation"
//! - **Purpose**: Define, evaluate, and validate research hypotheses
//!
//! ## Experimental Metric Agent
//!
//! - **Classification**: EXPERIMENTAL METRICS
//! - **decision_type**: "experimental_metrics"
//! - **Purpose**: Compute and report experimental metrics for hypothesis evaluation
//!
//! # Usage
//!
//! ## Programmatic
//!
//! ```rust,ignore
//! use llm_research_agents::agents::{Agent, HypothesisAgent};
//! use llm_research_agents::contracts::HypothesisInput;
//! use llm_research_agents::clients::RuVectorClient;
//!
//! // Create agent and client
//! let agent = HypothesisAgent::new();
//! let ruvector = RuVectorClient::from_env()?;
//!
//! // Execute evaluation
//! let (output, event) = agent.invoke(input).await?;
//!
//! // Persist decision event
//! let persisted = ruvector.persist_decision_event(event).await?;
//! ```
//!
//! ## CLI
//!
//! ```bash
//! # Evaluate a hypothesis
//! llm-research agents hypothesis evaluate --input hypothesis.json
//!
//! # Inspect a decision event
//! llm-research agents hypothesis inspect --event-id <uuid>
//! ```
//!
//! # Modules
//!
//! - [`agents`]: Agent implementations (HypothesisAgent, etc.)
//! - [`contracts`]: Input/output schemas and DecisionEvent
//! - [`clients`]: External service clients (ruvector-service)
//! - [`handlers`]: Edge Function HTTP handlers
//! - [`telemetry`]: LLM-Observatory compatible telemetry

#![warn(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

pub mod agents;
pub mod clients;
pub mod contracts;
pub mod handlers;
pub mod telemetry;

// Re-export commonly used types
pub use agents::{
    Agent,
    HypothesisAgent, HypothesisAgentError, HYPOTHESIS_AGENT_ID, HYPOTHESIS_AGENT_VERSION,
    ExperimentalMetricAgent, MetricAgentRuntimeError, METRIC_AGENT_ID, METRIC_AGENT_VERSION,
    AgentTelemetry,
};
pub use clients::{RuVectorClient, RuVectorConfig, RuVectorError, RuVectorPersistence};
pub use contracts::{DecisionEvent, DecisionType, HypothesisInput, HypothesisOutput};
pub use contracts::metrics::{MetricsInput, MetricsOutput, MetricType, ComputedMetric};
pub use handlers::{HypothesisHandler, HypothesisEvaluateRequest, HypothesisEvaluateResponse};
pub use handlers::{MetricHandler, MetricComputeRequest, MetricComputeResponse};
pub use telemetry::TelemetryEmitter;

/// Type alias for MetricAgent (for backward compatibility with CLI)
pub type MetricAgent = ExperimentalMetricAgent;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Agent registration information for platform wiring.
#[derive(Debug, Clone)]
pub struct AgentRegistration {
    /// Agent ID
    pub id: String,
    /// Agent version
    pub version: String,
    /// Agent classification
    pub classification: String,
    /// CLI command
    pub cli_command: String,
    /// CLI subcommands
    pub cli_subcommands: Vec<String>,
    /// Endpoint path
    pub endpoint_path: String,
}

/// Get registration info for all agents.
pub fn get_agent_registrations() -> Vec<AgentRegistration> {
    vec![
        AgentRegistration {
            id: HYPOTHESIS_AGENT_ID.to_string(),
            version: HYPOTHESIS_AGENT_VERSION.to_string(),
            classification: "HYPOTHESIS_EVALUATION".to_string(),
            cli_command: "hypothesis".to_string(),
            cli_subcommands: vec!["evaluate".to_string(), "inspect".to_string(), "validate".to_string()],
            endpoint_path: "/api/v1/agents/hypothesis".to_string(),
        },
        AgentRegistration {
            id: METRIC_AGENT_ID.to_string(),
            version: METRIC_AGENT_VERSION.to_string(),
            classification: "EXPERIMENTAL_METRICS".to_string(),
            cli_command: "metric".to_string(),
            cli_subcommands: vec!["compute".to_string(), "inspect".to_string()],
            endpoint_path: "/api/v1/agents/metric".to_string(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_registrations() {
        let registrations = get_agent_registrations();
        assert!(!registrations.is_empty());

        let hypothesis_reg = &registrations[0];
        assert_eq!(hypothesis_reg.id, HYPOTHESIS_AGENT_ID);
        assert_eq!(hypothesis_reg.classification, "HYPOTHESIS_EVALUATION");
    }
}
