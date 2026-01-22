//! DecisionEvent Schema
//!
//! The DecisionEvent is the primary output structure for all LLM-Research-Lab agents.
//! Every agent invocation MUST emit exactly ONE DecisionEvent to ruvector-service.
//!
//! # Constitution Compliance
//!
//! DecisionEvent schema MUST include:
//! - agent_id: Unique identifier for the agent
//! - agent_version: Semantic version of the agent
//! - decision_type: Type of decision made
//! - inputs_hash: SHA256 hash of inputs for determinism verification
//! - outputs: Structured output data
//! - confidence: Statistical or experimental certainty (0.0 - 1.0)
//! - constraints_applied: Experimental scope and assumptions
//! - execution_ref: Reference to execution context
//! - timestamp: UTC timestamp

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use uuid::Uuid;
use validator::Validate;

/// The authoritative decision type enum for all LLM-Research-Lab agents.
///
/// Per PROMPT 1: Agents must be classified as one of these types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DecisionType {
    /// Hypothesis evaluation decisions
    HypothesisEvaluation,
    /// Experimental metrics computation
    ExperimentalMetrics,
    /// Hypothesis definition
    HypothesisDefinition,
    /// Hypothesis validation
    HypothesisValidation,
}

impl std::fmt::Display for DecisionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HypothesisEvaluation => write!(f, "hypothesis_evaluation"),
            Self::ExperimentalMetrics => write!(f, "experimental_metrics"),
            Self::HypothesisDefinition => write!(f, "hypothesis_definition"),
            Self::HypothesisValidation => write!(f, "hypothesis_validation"),
        }
    }
}

/// Confidence level with statistical or experimental certainty.
///
/// Range: 0.0 (no confidence) to 1.0 (complete confidence)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Confidence {
    /// The confidence value (0.0 - 1.0)
    /// Note: Validation is done at construction time, not via derive macro
    pub value: Decimal,

    /// Method used to calculate confidence
    pub method: ConfidenceMethod,

    /// Sample size used in calculation (if applicable)
    pub sample_size: Option<u64>,

    /// Confidence interval lower bound (if applicable)
    pub ci_lower: Option<Decimal>,

    /// Confidence interval upper bound (if applicable)
    pub ci_upper: Option<Decimal>,
}

/// Method used to calculate confidence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfidenceMethod {
    /// Bayesian inference
    Bayesian,
    /// Frequentist hypothesis testing
    Frequentist,
    /// Bootstrap resampling
    Bootstrap,
    /// Heuristic or rule-based
    Heuristic,
    /// Expert judgment
    Expert,
    /// Ensemble of methods
    Ensemble,
}

/// Constraints applied during agent execution.
///
/// Documents experimental scope, assumptions, and limitations.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConstraintsApplied {
    /// Experimental scope boundaries
    pub scope: Vec<String>,

    /// Assumptions made during evaluation
    pub assumptions: Vec<String>,

    /// Known limitations
    pub limitations: Vec<String>,

    /// Data filters applied
    pub data_filters: Vec<String>,

    /// Temporal constraints (date ranges, etc.)
    pub temporal_bounds: Option<TemporalBounds>,
}

/// Temporal bounds for experimental constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalBounds {
    pub start: Option<DateTime<Utc>>,
    pub end: Option<DateTime<Utc>>,
}

/// Execution reference for tracing and debugging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRef {
    /// Unique execution ID
    pub execution_id: Uuid,

    /// Trace ID for distributed tracing
    pub trace_id: Option<String>,

    /// Span ID for distributed tracing
    pub span_id: Option<String>,

    /// Parent execution reference (for chained executions)
    pub parent_ref: Option<String>,

    /// Cloud Run revision or function version
    pub runtime_version: Option<String>,
}

/// The DecisionEvent - primary output for all LLM-Research-Lab agents.
///
/// # Constitution Requirements
///
/// Every agent invocation MUST emit exactly ONE DecisionEvent to ruvector-service.
/// This is a NON-NEGOTIABLE requirement.
///
/// # Usage
///
/// ```rust,ignore
/// let event = DecisionEvent::builder()
///     .agent_id("hypothesis-agent-v1")
///     .agent_version("1.0.0")
///     .decision_type(DecisionType::HypothesisEvaluation)
///     .outputs(output_value)
///     .confidence(confidence)
///     .build()?;
///
/// ruvector_client.persist_decision_event(event).await?;
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct DecisionEvent {
    /// Unique identifier for this decision event
    pub id: Uuid,

    /// Agent identifier (e.g., "hypothesis-agent-v1")
    #[validate(length(min = 1, max = 128))]
    pub agent_id: String,

    /// Agent version (semantic versioning)
    #[validate(length(min = 1, max = 32))]
    pub agent_version: String,

    /// Type of decision made
    pub decision_type: DecisionType,

    /// SHA256 hash of inputs for determinism verification
    #[validate(length(equal = 64))]
    pub inputs_hash: String,

    /// Structured output data
    pub outputs: serde_json::Value,

    /// Statistical or experimental certainty
    pub confidence: Confidence,

    /// Experimental scope, assumptions, and limitations
    pub constraints_applied: ConstraintsApplied,

    /// Execution context reference
    pub execution_ref: ExecutionRef,

    /// UTC timestamp of event creation
    pub timestamp: DateTime<Utc>,

    /// Optional metadata for extensibility
    pub metadata: Option<serde_json::Value>,
}

impl DecisionEvent {
    /// Create a new DecisionEvent builder.
    pub fn builder() -> DecisionEventBuilder {
        DecisionEventBuilder::default()
    }

    /// Compute SHA256 hash of input data for determinism verification.
    pub fn compute_inputs_hash<T: Serialize>(inputs: &T) -> Result<String, serde_json::Error> {
        let json = serde_json::to_string(inputs)?;
        let mut hasher = Sha256::new();
        hasher.update(json.as_bytes());
        Ok(hex::encode(hasher.finalize()))
    }
}

/// Builder for DecisionEvent.
#[derive(Debug, Default)]
pub struct DecisionEventBuilder {
    agent_id: Option<String>,
    agent_version: Option<String>,
    decision_type: Option<DecisionType>,
    inputs_hash: Option<String>,
    outputs: Option<serde_json::Value>,
    confidence: Option<Confidence>,
    constraints_applied: Option<ConstraintsApplied>,
    execution_ref: Option<ExecutionRef>,
    metadata: Option<serde_json::Value>,
}

impl DecisionEventBuilder {
    pub fn agent_id(mut self, id: impl Into<String>) -> Self {
        self.agent_id = Some(id.into());
        self
    }

    pub fn agent_version(mut self, version: impl Into<String>) -> Self {
        self.agent_version = Some(version.into());
        self
    }

    pub fn decision_type(mut self, dt: DecisionType) -> Self {
        self.decision_type = Some(dt);
        self
    }

    pub fn inputs_hash(mut self, hash: impl Into<String>) -> Self {
        self.inputs_hash = Some(hash.into());
        self
    }

    pub fn outputs(mut self, outputs: serde_json::Value) -> Self {
        self.outputs = Some(outputs);
        self
    }

    pub fn confidence(mut self, confidence: Confidence) -> Self {
        self.confidence = Some(confidence);
        self
    }

    pub fn constraints_applied(mut self, constraints: ConstraintsApplied) -> Self {
        self.constraints_applied = Some(constraints);
        self
    }

    pub fn execution_ref(mut self, exec_ref: ExecutionRef) -> Self {
        self.execution_ref = Some(exec_ref);
        self
    }

    pub fn metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Build the DecisionEvent.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing.
    pub fn build(self) -> Result<DecisionEvent, DecisionEventBuildError> {
        let agent_id = self.agent_id.ok_or(DecisionEventBuildError::MissingField("agent_id"))?;
        let agent_version = self.agent_version.ok_or(DecisionEventBuildError::MissingField("agent_version"))?;
        let decision_type = self.decision_type.ok_or(DecisionEventBuildError::MissingField("decision_type"))?;
        let inputs_hash = self.inputs_hash.ok_or(DecisionEventBuildError::MissingField("inputs_hash"))?;
        let outputs = self.outputs.ok_or(DecisionEventBuildError::MissingField("outputs"))?;
        let confidence = self.confidence.ok_or(DecisionEventBuildError::MissingField("confidence"))?;

        let execution_ref = self.execution_ref.unwrap_or_else(|| ExecutionRef {
            execution_id: Uuid::new_v4(),
            trace_id: None,
            span_id: None,
            parent_ref: None,
            runtime_version: None,
        });

        Ok(DecisionEvent {
            id: Uuid::new_v4(),
            agent_id,
            agent_version,
            decision_type,
            inputs_hash,
            outputs,
            confidence,
            constraints_applied: self.constraints_applied.unwrap_or_default(),
            execution_ref,
            timestamp: Utc::now(),
            metadata: self.metadata,
        })
    }
}

/// Error type for DecisionEvent building.
#[derive(Debug, thiserror::Error)]
pub enum DecisionEventBuildError {
    #[error("Missing required field: {0}")]
    MissingField(&'static str),
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_decision_event_builder() {
        let confidence = Confidence {
            value: dec!(0.95),
            method: ConfidenceMethod::Bayesian,
            sample_size: Some(1000),
            ci_lower: Some(dec!(0.92)),
            ci_upper: Some(dec!(0.98)),
        };

        let event = DecisionEvent::builder()
            .agent_id("hypothesis-agent-v1")
            .agent_version("1.0.0")
            .decision_type(DecisionType::HypothesisEvaluation)
            .inputs_hash("a".repeat(64))
            .outputs(serde_json::json!({"result": "accepted"}))
            .confidence(confidence)
            .build()
            .expect("Failed to build DecisionEvent");

        assert_eq!(event.agent_id, "hypothesis-agent-v1");
        assert_eq!(event.decision_type, DecisionType::HypothesisEvaluation);
    }

    #[test]
    fn test_compute_inputs_hash() {
        let input = serde_json::json!({"hypothesis": "test", "data": [1, 2, 3]});
        let hash = DecisionEvent::compute_inputs_hash(&input).expect("Failed to compute hash");

        assert_eq!(hash.len(), 64);

        // Verify determinism - same input should produce same hash
        let hash2 = DecisionEvent::compute_inputs_hash(&input).expect("Failed to compute hash");
        assert_eq!(hash, hash2);
    }

    #[test]
    fn test_decision_type_serialization() {
        let dt = DecisionType::HypothesisEvaluation;
        let json = serde_json::to_string(&dt).expect("Failed to serialize");
        assert_eq!(json, "\"hypothesis_evaluation\"");
    }
}
