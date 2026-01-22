//! Experimental Metrics Agent Contracts
//!
//! This module defines the input and output schemas for experimental metrics computation.
//!
//! # Agent Contract Definition (PROMPT 1 Compliance)
//!
//! ## Agent Identity
//!
//! - **Agent Name**: Experimental Metric Agent
//! - **Agent ID**: `experimental-metric-agent`
//! - **Version**: `1.0.0`
//! - **Purpose**: Compute and report experimental metrics used to evaluate hypotheses and research outcomes.
//! - **Classification**: EXPERIMENTAL METRICS
//! - **decision_type**: `experimental_metrics`
//!
//! ## Scope
//!
//! This agent MAY:
//! - Compute experimental metrics (central tendency, dispersion, correlation, etc.)
//! - Normalize and validate metric outputs
//! - Emit structured metric artifacts
//! - Compute confidence intervals for metrics
//! - Aggregate metrics across groups
//!
//! ## Explicit Non-Responsibilities (MUST NOT)
//!
//! This agent MUST NOT:
//! - Execute inference
//! - Modify prompts or responses
//! - Route inference requests
//! - Trigger orchestration or retries
//! - Apply optimizations automatically
//! - Enforce policies or governance decisions
//! - Connect directly to Google SQL
//! - Execute SQL queries
//! - Own or manage persistence (only via ruvector-service)
//!
//! ## Consumers
//!
//! Systems that MAY consume this agent's output:
//! - LLM-Observatory (telemetry)
//! - LLM-CostOps (cost signals for research analysis)
//! - Governance systems (research artifacts)
//! - Audit systems (decision events)
//! - Research dashboards
//!
//! ## CLI Invocation
//!
//! ```bash
//! llm-research agents metric compute --input <file|stdin> [--output-format json|yaml|table]
//! llm-research agents metric inspect --event-id <uuid>
//! ```
//!
//! ## Failure Modes
//!
//! | Error Code | Description | Recovery |
//! |------------|-------------|----------|
//! | METRIC_INPUT_INVALID | Input validation failed | Fix input schema |
//! | METRIC_DATA_EMPTY | No data records provided | Provide data |
//! | METRIC_COMPUTATION_FAILED | Metric calculation error | Check data types |
//! | METRIC_MISSING_REQUIRED | Missing required field | Provide field |
//! | METRIC_PERSISTENCE_FAILED | RuVector persistence error | Retry or escalate |
//! | METRIC_CONFIDENCE_ERROR | Confidence calculation failed | Check sample size |
//!
//! ## Versioning
//!
//! - Semantic versioning (MAJOR.MINOR.PATCH)
//! - Breaking changes increment MAJOR
//! - New features increment MINOR
//! - Bug fixes increment PATCH
//! - inputs_hash ensures determinism verification

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

/// Input schema for metrics computation.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MetricsInput {
    /// Unique request identifier
    pub request_id: Uuid,

    /// Experiment or research context
    #[validate(length(min = 1, max = 255))]
    pub context_id: String,

    /// Metrics to compute
    #[validate(length(min = 1))]
    pub metrics_requested: Vec<MetricRequest>,

    /// Raw data for computation
    pub data: MetricsData,

    /// Computation configuration
    pub config: MetricsConfig,
}

/// Single metric computation request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricRequest {
    /// Metric name
    pub name: String,

    /// Metric type
    pub metric_type: MetricType,

    /// Variable to compute metric on
    pub variable: String,

    /// Optional group-by field
    pub group_by: Option<String>,

    /// Additional parameters
    pub params: Option<serde_json::Value>,
}

/// Type of metric to compute.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MetricType {
    /// Central tendency (mean, median, mode)
    CentralTendency,
    /// Dispersion (variance, std dev, IQR)
    Dispersion,
    /// Distribution shape (skewness, kurtosis)
    DistributionShape,
    /// Percentile or quantile
    Percentile,
    /// Correlation coefficient
    Correlation,
    /// Regression coefficient
    Regression,
    /// Custom aggregation
    CustomAggregation,
}

/// Raw data for metrics computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsData {
    /// Data source reference
    pub source: String,

    /// Data records
    pub records: Vec<serde_json::Value>,

    /// Column/field definitions
    pub schema: Option<DataSchema>,
}

/// Schema definition for data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSchema {
    pub fields: Vec<FieldDefinition>,
}

/// Field definition in schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
}

/// Configuration for metrics computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Handle missing values
    pub handle_missing: MissingValueStrategy,

    /// Decimal precision
    pub precision: u8,

    /// Include confidence intervals
    pub include_ci: bool,

    /// CI level (e.g., 0.95)
    pub ci_level: Option<Decimal>,
}

/// Strategy for handling missing values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MissingValueStrategy {
    Skip,
    ZeroFill,
    MeanImpute,
    MedianImpute,
    Fail,
}

/// Output schema for metrics computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsOutput {
    /// Computed metrics
    pub metrics: Vec<ComputedMetric>,

    /// Computation metadata
    pub metadata: MetricsMetadata,

    /// Any warnings generated
    pub warnings: Vec<String>,
}

/// Single computed metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputedMetric {
    /// Metric name
    pub name: String,

    /// Metric type
    pub metric_type: MetricType,

    /// Computed value
    pub value: Decimal,

    /// Confidence interval (if computed)
    pub confidence_interval: Option<MetricConfidenceInterval>,

    /// Group (if grouped computation)
    pub group: Option<String>,

    /// Sample size used
    pub sample_size: u64,

    /// Missing values encountered
    pub missing_count: u64,
}

/// Confidence interval for a metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricConfidenceInterval {
    pub lower: Decimal,
    pub upper: Decimal,
    pub level: Decimal,
}

/// Metadata about metrics computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsMetadata {
    /// Computation timestamp
    pub computed_at: DateTime<Utc>,

    /// Total records processed
    pub records_processed: u64,

    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

// =============================================================================
// Extended Contract Types for Full Agent Compliance
// =============================================================================

/// Agent identity for the Experimental Metric Agent.
pub const AGENT_ID: &str = "experimental-metric-agent";
pub const AGENT_VERSION: &str = "1.0.0";

/// Error codes for the Experimental Metric Agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MetricErrorCode {
    /// Input validation failed
    MetricInputInvalid,
    /// No data records provided
    MetricDataEmpty,
    /// Metric calculation error
    MetricComputationFailed,
    /// Missing required field
    MetricMissingRequired,
    /// RuVector persistence error
    MetricPersistenceFailed,
    /// Confidence calculation failed
    MetricConfidenceError,
    /// Unknown or internal error
    MetricInternalError,
}

impl std::fmt::Display for MetricErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MetricInputInvalid => write!(f, "METRIC_INPUT_INVALID"),
            Self::MetricDataEmpty => write!(f, "METRIC_DATA_EMPTY"),
            Self::MetricComputationFailed => write!(f, "METRIC_COMPUTATION_FAILED"),
            Self::MetricMissingRequired => write!(f, "METRIC_MISSING_REQUIRED"),
            Self::MetricPersistenceFailed => write!(f, "METRIC_PERSISTENCE_FAILED"),
            Self::MetricConfidenceError => write!(f, "METRIC_CONFIDENCE_ERROR"),
            Self::MetricInternalError => write!(f, "METRIC_INTERNAL_ERROR"),
        }
    }
}

/// Error type for metric agent operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricAgentError {
    /// Error code
    pub code: MetricErrorCode,
    /// Human-readable message
    pub message: String,
    /// Request ID for correlation
    pub request_id: Uuid,
    /// Field that caused the error (if applicable)
    pub field: Option<String>,
    /// Additional details
    pub details: Option<serde_json::Value>,
}

impl std::fmt::Display for MetricAgentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for MetricAgentError {}

impl MetricAgentError {
    /// Create a new MetricAgentError.
    pub fn new(code: MetricErrorCode, message: impl Into<String>, request_id: Uuid) -> Self {
        Self {
            code,
            message: message.into(),
            request_id,
            field: None,
            details: None,
        }
    }

    /// Add field context.
    pub fn with_field(mut self, field: impl Into<String>) -> Self {
        self.field = Some(field.into());
        self
    }

    /// Add details.
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }
}

/// Validated metrics input (after input validation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedMetricsInput {
    /// Original input
    pub input: MetricsInput,
    /// Validation timestamp
    pub validated_at: DateTime<Utc>,
    /// Input hash for determinism
    pub input_hash: String,
}

impl ValidatedMetricsInput {
    /// Create validated input from raw input.
    pub fn from_input(input: MetricsInput, input_hash: String) -> Self {
        Self {
            input,
            validated_at: Utc::now(),
            input_hash,
        }
    }
}

/// Full metric computation result including DecisionEvent mapping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComputationResult {
    /// Computed metrics output
    pub output: MetricsOutput,
    /// Confidence for the overall computation
    pub confidence: MetricComputationConfidence,
    /// Constraints that were applied
    pub constraints: MetricConstraints,
}

/// Confidence assessment for the overall metric computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComputationConfidence {
    /// Overall confidence value (0.0 - 1.0)
    pub value: Decimal,
    /// Method used to compute confidence
    pub method: String,
    /// Total sample size across all metrics
    pub total_sample_size: u64,
    /// Confidence interval (if applicable)
    pub ci_lower: Option<Decimal>,
    /// Confidence interval upper bound
    pub ci_upper: Option<Decimal>,
}

/// Constraints applied during metric computation.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricConstraints {
    /// Scope boundaries (e.g., "experiment-123", "model-comparison")
    pub scope: Vec<String>,
    /// Assumptions made (e.g., "normal distribution", "independent samples")
    pub assumptions: Vec<String>,
    /// Known limitations (e.g., "small sample size", "missing data")
    pub limitations: Vec<String>,
    /// Data filters applied (e.g., "status=completed", "created_after=2024-01-01")
    pub data_filters: Vec<String>,
}

/// What gets persisted to ruvector-service.
///
/// Per constitution: DecisionEvent is persisted with:
/// - agent_id, agent_version, decision_type, inputs_hash, outputs, confidence, constraints_applied, execution_ref, timestamp
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricEventPersistence {
    /// Fields that ARE persisted
    pub persisted_fields: Vec<String>,
    /// Fields that MUST NOT be persisted (raw data, PII, etc.)
    pub excluded_fields: Vec<String>,
}

impl Default for MetricEventPersistence {
    fn default() -> Self {
        Self {
            persisted_fields: vec![
                "agent_id".to_string(),
                "agent_version".to_string(),
                "decision_type".to_string(),
                "inputs_hash".to_string(),
                "outputs".to_string(),
                "confidence".to_string(),
                "constraints_applied".to_string(),
                "execution_ref".to_string(),
                "timestamp".to_string(),
                "metadata".to_string(),
            ],
            excluded_fields: vec![
                "raw_input_data".to_string(),
                "pii_fields".to_string(),
                "api_keys".to_string(),
                "credentials".to_string(),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_metrics_output_serialization() {
        let output = MetricsOutput {
            metrics: vec![ComputedMetric {
                name: "mean_accuracy".to_string(),
                metric_type: MetricType::CentralTendency,
                value: dec!(0.89),
                confidence_interval: Some(MetricConfidenceInterval {
                    lower: dec!(0.85),
                    upper: dec!(0.93),
                    level: dec!(0.95),
                }),
                group: None,
                sample_size: 1000,
                missing_count: 5,
            }],
            metadata: MetricsMetadata {
                computed_at: Utc::now(),
                records_processed: 1000,
                processing_time_ms: 45,
            },
            warnings: vec![],
        };

        let json = serde_json::to_string(&output).expect("Serialization failed");
        assert!(json.contains("mean_accuracy"));
    }

    #[test]
    fn test_metric_error_creation() {
        let request_id = Uuid::new_v4();
        let error = MetricAgentError::new(
            MetricErrorCode::MetricInputInvalid,
            "Missing required field: metrics_requested",
            request_id,
        )
        .with_field("metrics_requested");

        assert_eq!(error.code, MetricErrorCode::MetricInputInvalid);
        assert_eq!(error.field, Some("metrics_requested".to_string()));
    }

    #[test]
    fn test_metric_constraints_default() {
        let constraints = MetricConstraints::default();
        assert!(constraints.scope.is_empty());
        assert!(constraints.assumptions.is_empty());
    }

    #[test]
    fn test_persistence_fields() {
        let persistence = MetricEventPersistence::default();
        assert!(persistence.persisted_fields.contains(&"agent_id".to_string()));
        assert!(persistence.persisted_fields.contains(&"decision_type".to_string()));
        assert!(persistence.excluded_fields.contains(&"pii_fields".to_string()));
    }
}
