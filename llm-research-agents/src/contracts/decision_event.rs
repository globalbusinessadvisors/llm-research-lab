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
//! - phase7_identity: Phase 7 identity metadata for traceability (REQUIRED)
//! - evidence_refs: References to supporting evidence (run IDs, telemetry, datasets)

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sha2::{Sha256, Digest};
use uuid::Uuid;
use validator::Validate;

/// The authoritative decision type enum for all LLM-Research-Lab agents.
///
/// Per PROMPT 1: Agents must be classified as one of these types.
/// Phase 7 adds signal-based types for intelligence INPUTS (not outcomes).
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

    // Phase 7 signal types (intelligence INPUTS, not outcomes)
    /// Signal indicating a hypothesis has been generated or updated
    HypothesisSignal,
    /// Signal from simulation execution outcome
    SimulationOutcomeSignal,
    /// Signal comparing multiple scenarios
    ScenarioComparisonSignal,
    /// Signal indicating confidence level change
    ConfidenceDeltaSignal,
    /// Signal representing uncertainty quantification
    UncertaintySignal,
    /// Signal capturing research insights
    ResearchInsightSignal,
}

impl std::fmt::Display for DecisionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HypothesisEvaluation => write!(f, "hypothesis_evaluation"),
            Self::ExperimentalMetrics => write!(f, "experimental_metrics"),
            Self::HypothesisDefinition => write!(f, "hypothesis_definition"),
            Self::HypothesisValidation => write!(f, "hypothesis_validation"),
            // Phase 7 signal types
            Self::HypothesisSignal => write!(f, "hypothesis_signal"),
            Self::SimulationOutcomeSignal => write!(f, "simulation_outcome_signal"),
            Self::ScenarioComparisonSignal => write!(f, "scenario_comparison_signal"),
            Self::ConfidenceDeltaSignal => write!(f, "confidence_delta_signal"),
            Self::UncertaintySignal => write!(f, "uncertainty_signal"),
            Self::ResearchInsightSignal => write!(f, "research_insight_signal"),
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

/// Phase 7 agent identity for traceability.
///
/// Required metadata for all Phase 7 Layer 2 agents to enable
/// full traceability and audit compliance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phase7Identity {
    /// Source agent name (e.g., "hypothesis-evaluator")
    pub source_agent: String,

    /// Agent domain (e.g., "research", "simulation", "evaluation")
    pub domain: String,

    /// Phase identifier (always "phase7" for Phase 7 agents)
    pub phase: String,

    /// Layer identifier (always "layer2" for Layer 2 hardening)
    pub layer: String,

    /// Agent version following semantic versioning
    pub agent_version: String,
}

impl Default for Phase7Identity {
    fn default() -> Self {
        Self {
            source_agent: std::env::var("AGENT_NAME").unwrap_or_else(|_| "unknown".to_string()),
            domain: std::env::var("AGENT_DOMAIN").unwrap_or_else(|_| "research".to_string()),
            phase: "phase7".to_string(),
            layer: "layer2".to_string(),
            agent_version: std::env::var("AGENT_VERSION").unwrap_or_else(|_| "1.0.0".to_string()),
        }
    }
}

impl Phase7Identity {
    /// Create a new Phase7Identity with explicit values.
    pub fn new(
        source_agent: impl Into<String>,
        domain: impl Into<String>,
        agent_version: impl Into<String>,
    ) -> Self {
        Self {
            source_agent: source_agent.into(),
            domain: domain.into(),
            phase: "phase7".to_string(),
            layer: "layer2".to_string(),
            agent_version: agent_version.into(),
        }
    }
}

/// Evidence references for DecisionEvent.
///
/// Links the decision to supporting evidence for audit and reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvidenceRefs {
    /// Run IDs referenced (e.g., experiment run identifiers)
    pub run_ids: Vec<String>,

    /// Telemetry IDs referenced (e.g., trace/span identifiers)
    pub telemetry_ids: Vec<String>,

    /// Dataset references (e.g., dataset URIs or identifiers)
    pub dataset_refs: Vec<String>,
}

impl EvidenceRefs {
    /// Create a new EvidenceRefs builder.
    pub fn builder() -> EvidenceRefsBuilder {
        EvidenceRefsBuilder::default()
    }

    /// Check if any evidence references are present.
    pub fn is_empty(&self) -> bool {
        self.run_ids.is_empty() && self.telemetry_ids.is_empty() && self.dataset_refs.is_empty()
    }
}

/// Builder for EvidenceRefs.
#[derive(Debug, Default)]
pub struct EvidenceRefsBuilder {
    run_ids: Vec<String>,
    telemetry_ids: Vec<String>,
    dataset_refs: Vec<String>,
}

impl EvidenceRefsBuilder {
    pub fn run_id(mut self, id: impl Into<String>) -> Self {
        self.run_ids.push(id.into());
        self
    }

    pub fn run_ids(mut self, ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.run_ids.extend(ids.into_iter().map(Into::into));
        self
    }

    pub fn telemetry_id(mut self, id: impl Into<String>) -> Self {
        self.telemetry_ids.push(id.into());
        self
    }

    pub fn telemetry_ids(mut self, ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.telemetry_ids.extend(ids.into_iter().map(Into::into));
        self
    }

    pub fn dataset_ref(mut self, ref_: impl Into<String>) -> Self {
        self.dataset_refs.push(ref_.into());
        self
    }

    pub fn dataset_refs(mut self, refs: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.dataset_refs.extend(refs.into_iter().map(Into::into));
        self
    }

    pub fn build(self) -> EvidenceRefs {
        EvidenceRefs {
            run_ids: self.run_ids,
            telemetry_ids: self.telemetry_ids,
            dataset_refs: self.dataset_refs,
        }
    }
}

/// The DecisionEvent - primary output for all LLM-Research-Lab agents.
///
/// # Constitution Requirements
///
/// Every agent invocation MUST emit exactly ONE DecisionEvent to ruvector-service.
/// This is a NON-NEGOTIABLE requirement.
///
/// # Phase 7 Layer 2 Requirements
///
/// All DecisionEvents MUST include:
/// - `phase7_identity`: Agent identity metadata for traceability
/// - `evidence_refs`: References to supporting evidence
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
///     .phase7_identity(Phase7Identity::new("hypothesis-evaluator", "research", "1.0.0"))
///     .evidence_refs(EvidenceRefs::builder().run_id("run-123").build())
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

    /// Phase 7 identity metadata (REQUIRED for Layer 2 compliance)
    pub phase7_identity: Phase7Identity,

    /// Evidence references for audit and reproducibility
    pub evidence_refs: EvidenceRefs,
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
    phase7_identity: Option<Phase7Identity>,
    evidence_refs: Option<EvidenceRefs>,
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

    /// Set Phase 7 identity metadata.
    ///
    /// If not provided, defaults will be used based on environment variables.
    pub fn phase7_identity(mut self, identity: Phase7Identity) -> Self {
        self.phase7_identity = Some(identity);
        self
    }

    /// Set evidence references.
    ///
    /// If not provided, defaults to empty references.
    pub fn evidence_refs(mut self, refs: EvidenceRefs) -> Self {
        self.evidence_refs = Some(refs);
        self
    }

    /// Build the DecisionEvent.
    ///
    /// # Errors
    ///
    /// Returns an error if required fields are missing.
    ///
    /// # Phase 7 Layer 2 Defaults
    ///
    /// - `phase7_identity`: Defaults from environment variables if not provided
    /// - `evidence_refs`: Defaults to empty references if not provided
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

        // Phase 7 identity defaults if not provided
        let phase7_identity = self.phase7_identity.unwrap_or_default();

        // Evidence refs defaults to empty if not provided
        let evidence_refs = self.evidence_refs.unwrap_or_default();

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
            phase7_identity,
            evidence_refs,
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
        // Phase 7 defaults should be applied
        assert_eq!(event.phase7_identity.phase, "phase7");
        assert_eq!(event.phase7_identity.layer, "layer2");
        assert!(event.evidence_refs.is_empty());
    }

    #[test]
    fn test_decision_event_with_phase7_identity() {
        let confidence = Confidence {
            value: dec!(0.85),
            method: ConfidenceMethod::Frequentist,
            sample_size: Some(500),
            ci_lower: None,
            ci_upper: None,
        };

        let phase7_identity = Phase7Identity::new(
            "simulation-runner",
            "simulation",
            "2.0.0",
        );

        let evidence_refs = EvidenceRefs::builder()
            .run_id("run-abc-123")
            .run_id("run-def-456")
            .telemetry_id("trace-xyz")
            .dataset_ref("gs://bucket/dataset.parquet")
            .build();

        let event = DecisionEvent::builder()
            .agent_id("simulation-agent-v2")
            .agent_version("2.0.0")
            .decision_type(DecisionType::SimulationOutcomeSignal)
            .inputs_hash("b".repeat(64))
            .outputs(serde_json::json!({"simulation_result": "converged"}))
            .confidence(confidence)
            .phase7_identity(phase7_identity)
            .evidence_refs(evidence_refs)
            .build()
            .expect("Failed to build DecisionEvent");

        assert_eq!(event.phase7_identity.source_agent, "simulation-runner");
        assert_eq!(event.phase7_identity.domain, "simulation");
        assert_eq!(event.phase7_identity.phase, "phase7");
        assert_eq!(event.phase7_identity.layer, "layer2");
        assert_eq!(event.phase7_identity.agent_version, "2.0.0");
        assert_eq!(event.evidence_refs.run_ids.len(), 2);
        assert_eq!(event.evidence_refs.telemetry_ids.len(), 1);
        assert_eq!(event.evidence_refs.dataset_refs.len(), 1);
        assert!(!event.evidence_refs.is_empty());
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

    #[test]
    fn test_phase7_signal_types_serialization() {
        let signal_types = vec![
            (DecisionType::HypothesisSignal, "\"hypothesis_signal\""),
            (DecisionType::SimulationOutcomeSignal, "\"simulation_outcome_signal\""),
            (DecisionType::ScenarioComparisonSignal, "\"scenario_comparison_signal\""),
            (DecisionType::ConfidenceDeltaSignal, "\"confidence_delta_signal\""),
            (DecisionType::UncertaintySignal, "\"uncertainty_signal\""),
            (DecisionType::ResearchInsightSignal, "\"research_insight_signal\""),
        ];

        for (dt, expected) in signal_types {
            let json = serde_json::to_string(&dt).expect("Failed to serialize");
            assert_eq!(json, expected, "Failed for {:?}", dt);
        }
    }

    #[test]
    fn test_phase7_identity_default() {
        let identity = Phase7Identity::default();
        assert_eq!(identity.phase, "phase7");
        assert_eq!(identity.layer, "layer2");
        // source_agent and domain depend on env vars, but should not panic
    }

    #[test]
    fn test_evidence_refs_builder() {
        let refs = EvidenceRefs::builder()
            .run_ids(vec!["run-1", "run-2", "run-3"])
            .telemetry_ids(vec!["trace-1"])
            .dataset_refs(vec!["dataset-a", "dataset-b"])
            .build();

        assert_eq!(refs.run_ids.len(), 3);
        assert_eq!(refs.telemetry_ids.len(), 1);
        assert_eq!(refs.dataset_refs.len(), 2);
        assert!(!refs.is_empty());
    }

    #[test]
    fn test_evidence_refs_empty() {
        let refs = EvidenceRefs::default();
        assert!(refs.is_empty());
    }
}
