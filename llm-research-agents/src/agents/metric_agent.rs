//! Experimental Metric Agent Implementation
//!
//! # Agent Contract (PROMPT 1 Compliance)
//!
//! ## Agent Identity
//!
//! - **Agent Name**: Experimental Metric Agent
//! - **Agent ID**: `experimental-metric-agent`
//! - **Version**: `1.0.0`
//! - **Purpose**: Compute and report experimental metrics used to evaluate hypotheses and research outcomes.
//! - **Classification**: EXPERIMENTAL METRICS
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
//! ## decision_type
//!
//! `experimental_metrics`
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
//! ## Failure Modes
//!
//! - Invalid input schema: Returns validation error (METRIC_INPUT_INVALID)
//! - Empty data: Returns error (METRIC_DATA_EMPTY)
//! - Computation failure: Returns error (METRIC_COMPUTATION_FAILED)
//! - Missing required field: Returns error (METRIC_MISSING_REQUIRED)
//! - Persistence failure: Returns error (METRIC_PERSISTENCE_FAILED)
//!
//! ## CLI Invocation
//!
//! ```bash
//! llm-research agents metric compute --input <file|stdin> [--output-format json|yaml|table]
//! llm-research agents metric inspect --event-id <uuid>
//! ```

use async_trait::async_trait;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde_json::json;
use std::time::Instant;
use thiserror::Error;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;
use validator::Validate;
use chrono::Utc;

use crate::contracts::{
    metrics::*,
    decision_event::*,
    common::*,
};
use super::traits::{Agent, ConfidenceEstimator, PerformanceBounded, PerformanceBudget, BudgetViolation};
use super::telemetry::{AgentTelemetry, TelemetryEvent, TelemetryEventType};

// =============================================================================
// Agent Constants
// =============================================================================

/// Agent identifier per contract.
pub const METRIC_AGENT_ID: &str = "experimental-metric-agent";

/// Agent version (semantic versioning).
pub const METRIC_AGENT_VERSION: &str = "1.0.0";

// =============================================================================
// Error Types
// =============================================================================

/// Errors from Experimental Metric Agent operations.
#[derive(Debug, Error)]
pub enum MetricAgentRuntimeError {
    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Empty data: no records provided for computation")]
    EmptyData,

    #[error("Computation error for metric '{metric}': {message}")]
    Computation { metric: String, message: String },

    #[error("Missing required field: {0}")]
    MissingRequired(String),

    #[error("Type mismatch for field '{field}': expected {expected}, got {actual}")]
    TypeMismatch {
        field: String,
        expected: String,
        actual: String,
    },

    #[error("Confidence calculation error: {0}")]
    ConfidenceError(String),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Performance budget exceeded: {0}")]
    BudgetExceeded(String),
}

impl From<validator::ValidationErrors> for MetricAgentRuntimeError {
    fn from(err: validator::ValidationErrors) -> Self {
        MetricAgentRuntimeError::Validation(err.to_string())
    }
}

impl From<serde_json::Error> for MetricAgentRuntimeError {
    fn from(err: serde_json::Error) -> Self {
        MetricAgentRuntimeError::Serialization(err.to_string())
    }
}

impl From<BudgetViolation> for MetricAgentRuntimeError {
    fn from(err: BudgetViolation) -> Self {
        MetricAgentRuntimeError::BudgetExceeded(err.message)
    }
}

// =============================================================================
// Agent Configuration
// =============================================================================

/// Configuration for Experimental Metric Agent.
#[derive(Debug, Clone)]
pub struct MetricAgentConfig {
    /// Minimum sample size for meaningful computation
    pub min_sample_size: u64,

    /// Default decimal precision for computed values
    pub default_precision: u8,

    /// Enable telemetry emission
    pub emit_telemetry: bool,

    /// Enable confidence interval computation
    pub compute_confidence_intervals: bool,

    /// Default confidence level for intervals
    pub default_ci_level: Decimal,

    /// Maximum records to process (memory protection)
    pub max_records: usize,
}

impl Default for MetricAgentConfig {
    fn default() -> Self {
        Self {
            min_sample_size: 3,
            default_precision: 4,
            emit_telemetry: true,
            compute_confidence_intervals: true,
            default_ci_level: dec!(0.95),
            max_records: 1_000_000,
        }
    }
}

// =============================================================================
// Agent Implementation
// =============================================================================

/// Experimental Metric Agent for computing research metrics.
///
/// This agent implements the core metrics computation logic for LLM-Research-Lab.
/// It is stateless, deterministic, and produces structured DecisionEvents.
///
/// # Constitution Compliance
///
/// - Stateless runtime execution
/// - Deterministic for same inputs (verified via inputs_hash)
/// - Emits exactly ONE DecisionEvent per invocation
/// - No direct database access (persistence via ruvector-service only)
/// - Telemetry compatible with LLM-Observatory
///
/// # Phase 7 Performance Budgets
///
/// This agent enforces strict performance budgets:
/// - Maximum latency: 5000ms (default)
/// - Maximum tokens: 2500 (default)
/// - Maximum API calls: 5 (default)
///
/// Execution will ABORT if any budget is exceeded.
#[derive(Clone)]
pub struct ExperimentalMetricAgent {
    identity: AgentIdentity,
    config: MetricAgentConfig,
    telemetry: AgentTelemetry,
    /// Performance budget for this agent (Phase 7 MANDATORY)
    budget: PerformanceBudget,
}

impl ExperimentalMetricAgent {
    /// Create a new Experimental Metric Agent with default configuration.
    pub fn new() -> Self {
        Self::with_config(MetricAgentConfig::default())
    }

    /// Create a new Experimental Metric Agent with custom configuration.
    pub fn with_config(config: MetricAgentConfig) -> Self {
        Self::with_config_and_budget(config, PerformanceBudget::default())
    }

    /// Create a new Experimental Metric Agent with custom configuration and budget.
    pub fn with_config_and_budget(config: MetricAgentConfig, budget: PerformanceBudget) -> Self {
        Self {
            identity: AgentIdentity {
                id: METRIC_AGENT_ID.to_string(),
                version: METRIC_AGENT_VERSION.to_string(),
                classification: AgentClassification::ExperimentalMetrics,
                description: "Computes experimental metrics for hypothesis evaluation and research outcomes".to_string(),
            },
            config,
            telemetry: AgentTelemetry::new(METRIC_AGENT_ID.to_string()),
            budget,
        }
    }

    /// Compute a single metric from data records.
    #[instrument(skip(self, records), fields(metric_name = %request.name, metric_type = ?request.metric_type))]
    fn compute_metric(
        &self,
        request: &MetricRequest,
        records: &[serde_json::Value],
        config: &MetricsConfig,
    ) -> Result<ComputedMetric, MetricAgentRuntimeError> {
        debug!("Computing metric: {}", request.name);

        // Extract numeric values for the requested variable
        let values = self.extract_numeric_values(records, &request.variable)?;

        if values.is_empty() {
            return Err(MetricAgentRuntimeError::Computation {
                metric: request.name.clone(),
                message: format!("No valid numeric values for variable '{}'", request.variable),
            });
        }

        let sample_size = values.len() as u64;
        let missing_count = (records.len() - values.len()) as u64;

        // Compute the metric value based on type
        let (value, ci) = match request.metric_type {
            MetricType::CentralTendency => self.compute_central_tendency(&values, config)?,
            MetricType::Dispersion => self.compute_dispersion(&values, config)?,
            MetricType::DistributionShape => self.compute_distribution_shape(&values, config)?,
            MetricType::Percentile => self.compute_percentile(&values, request.params.as_ref(), config)?,
            MetricType::Correlation => self.compute_correlation(records, &request.variable, request.params.as_ref(), config)?,
            MetricType::Regression => self.compute_regression(records, &request.variable, request.params.as_ref(), config)?,
            MetricType::CustomAggregation => self.compute_custom_aggregation(&values, request.params.as_ref(), config)?,
        };

        // Round to configured precision
        let value = self.round_to_precision(value, config.precision);

        Ok(ComputedMetric {
            name: request.name.clone(),
            metric_type: request.metric_type.clone(),
            value,
            confidence_interval: ci,
            group: request.group_by.clone(),
            sample_size,
            missing_count,
        })
    }

    /// Extract numeric values from records for a given variable.
    fn extract_numeric_values(
        &self,
        records: &[serde_json::Value],
        variable: &str,
    ) -> Result<Vec<f64>, MetricAgentRuntimeError> {
        let mut values = Vec::with_capacity(records.len());

        for record in records {
            if let Some(value) = self.get_nested_value(record, variable) {
                if let Some(num) = value.as_f64() {
                    if num.is_finite() {
                        values.push(num);
                    }
                } else if let Some(s) = value.as_str() {
                    if let Ok(num) = s.parse::<f64>() {
                        if num.is_finite() {
                            values.push(num);
                        }
                    }
                }
            }
        }

        Ok(values)
    }

    /// Get nested value from JSON using dot notation (e.g., "metrics.accuracy").
    fn get_nested_value<'a>(&self, record: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
        let parts: Vec<&str> = path.split('.').collect();
        let mut current = record;

        for part in parts {
            current = current.get(part)?;
        }

        Some(current)
    }

    /// Compute central tendency (mean).
    fn compute_central_tendency(
        &self,
        values: &[f64],
        config: &MetricsConfig,
    ) -> Result<(Decimal, Option<MetricConfidenceInterval>), MetricAgentRuntimeError> {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;

        let ci = if config.include_ci && values.len() >= 3 {
            let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
            let std_error = (variance / n).sqrt();
            
            // t-critical for 95% CI with n-1 df (approximation for large n)
            let t_critical = if n > 30.0 { 1.96 } else { 2.0 + 2.0 / n };
            let margin = t_critical * std_error;

            let ci_level = config.ci_level.unwrap_or(dec!(0.95));

            Some(MetricConfidenceInterval {
                lower: Decimal::try_from(mean - margin).unwrap_or(dec!(0)),
                upper: Decimal::try_from(mean + margin).unwrap_or(dec!(0)),
                level: ci_level,
            })
        } else {
            None
        };

        let decimal_mean = Decimal::try_from(mean)
            .map_err(|e| MetricAgentRuntimeError::Computation {
                metric: "central_tendency".to_string(),
                message: e.to_string(),
            })?;

        Ok((decimal_mean, ci))
    }

    /// Compute dispersion (standard deviation).
    fn compute_dispersion(
        &self,
        values: &[f64],
        _config: &MetricsConfig,
    ) -> Result<(Decimal, Option<MetricConfidenceInterval>), MetricAgentRuntimeError> {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        let decimal_std = Decimal::try_from(std_dev)
            .map_err(|e| MetricAgentRuntimeError::Computation {
                metric: "dispersion".to_string(),
                message: e.to_string(),
            })?;

        Ok((decimal_std, None))
    }

    /// Compute distribution shape (skewness).
    fn compute_distribution_shape(
        &self,
        values: &[f64],
        _config: &MetricsConfig,
    ) -> Result<(Decimal, Option<MetricConfidenceInterval>), MetricAgentRuntimeError> {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return Ok((dec!(0), None));
        }

        // Compute skewness
        let m3 = values.iter().map(|x| ((x - mean) / std_dev).powi(3)).sum::<f64>() / n;
        let skewness = m3 * (n * (n - 1.0)).sqrt() / (n - 2.0);

        let decimal_skew = Decimal::try_from(skewness)
            .map_err(|e| MetricAgentRuntimeError::Computation {
                metric: "distribution_shape".to_string(),
                message: e.to_string(),
            })?;

        Ok((decimal_skew, None))
    }

    /// Compute percentile.
    fn compute_percentile(
        &self,
        values: &[f64],
        params: Option<&serde_json::Value>,
        _config: &MetricsConfig,
    ) -> Result<(Decimal, Option<MetricConfidenceInterval>), MetricAgentRuntimeError> {
        let percentile = params
            .and_then(|p| p.get("percentile"))
            .and_then(|v| v.as_f64())
            .unwrap_or(50.0); // Default to median

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = ((percentile / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        let value = sorted.get(idx).copied().unwrap_or(0.0);

        let decimal_value = Decimal::try_from(value)
            .map_err(|e| MetricAgentRuntimeError::Computation {
                metric: "percentile".to_string(),
                message: e.to_string(),
            })?;

        Ok((decimal_value, None))
    }

    /// Compute correlation (Pearson's r).
    fn compute_correlation(
        &self,
        records: &[serde_json::Value],
        variable_x: &str,
        params: Option<&serde_json::Value>,
        _config: &MetricsConfig,
    ) -> Result<(Decimal, Option<MetricConfidenceInterval>), MetricAgentRuntimeError> {
        let variable_y = params
            .and_then(|p| p.get("variable_y"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| MetricAgentRuntimeError::MissingRequired(
                "params.variable_y required for correlation".to_string()
            ))?;

        // Extract paired values
        let mut pairs: Vec<(f64, f64)> = Vec::new();
        for record in records {
            if let (Some(x), Some(y)) = (
                self.get_nested_value(record, variable_x).and_then(|v| v.as_f64()),
                self.get_nested_value(record, variable_y).and_then(|v| v.as_f64()),
            ) {
                if x.is_finite() && y.is_finite() {
                    pairs.push((x, y));
                }
            }
        }

        if pairs.len() < 3 {
            return Err(MetricAgentRuntimeError::Computation {
                metric: "correlation".to_string(),
                message: "Insufficient paired observations (need at least 3)".to_string(),
            });
        }

        let n = pairs.len() as f64;
        let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
        let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;

        let cov = pairs.iter().map(|(x, y)| (x - mean_x) * (y - mean_y)).sum::<f64>() / (n - 1.0);
        let std_x = (pairs.iter().map(|(x, _)| (x - mean_x).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let std_y = (pairs.iter().map(|(_, y)| (y - mean_y).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

        let r = if std_x > 0.0 && std_y > 0.0 {
            cov / (std_x * std_y)
        } else {
            0.0
        };

        let decimal_r = Decimal::try_from(r.clamp(-1.0, 1.0))
            .map_err(|e| MetricAgentRuntimeError::Computation {
                metric: "correlation".to_string(),
                message: e.to_string(),
            })?;

        Ok((decimal_r, None))
    }

    /// Compute regression coefficient (simple linear regression slope).
    fn compute_regression(
        &self,
        records: &[serde_json::Value],
        variable_x: &str,
        params: Option<&serde_json::Value>,
        _config: &MetricsConfig,
    ) -> Result<(Decimal, Option<MetricConfidenceInterval>), MetricAgentRuntimeError> {
        let variable_y = params
            .and_then(|p| p.get("variable_y"))
            .and_then(|v| v.as_str())
            .ok_or_else(|| MetricAgentRuntimeError::MissingRequired(
                "params.variable_y required for regression".to_string()
            ))?;

        // Extract paired values
        let mut pairs: Vec<(f64, f64)> = Vec::new();
        for record in records {
            if let (Some(x), Some(y)) = (
                self.get_nested_value(record, variable_x).and_then(|v| v.as_f64()),
                self.get_nested_value(record, variable_y).and_then(|v| v.as_f64()),
            ) {
                if x.is_finite() && y.is_finite() {
                    pairs.push((x, y));
                }
            }
        }

        if pairs.len() < 3 {
            return Err(MetricAgentRuntimeError::Computation {
                metric: "regression".to_string(),
                message: "Insufficient paired observations (need at least 3)".to_string(),
            });
        }

        let n = pairs.len() as f64;
        let mean_x = pairs.iter().map(|(x, _)| x).sum::<f64>() / n;
        let mean_y = pairs.iter().map(|(_, y)| y).sum::<f64>() / n;

        let numerator = pairs.iter().map(|(x, y)| (x - mean_x) * (y - mean_y)).sum::<f64>();
        let denominator = pairs.iter().map(|(x, _)| (x - mean_x).powi(2)).sum::<f64>();

        let slope = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        let decimal_slope = Decimal::try_from(slope)
            .map_err(|e| MetricAgentRuntimeError::Computation {
                metric: "regression".to_string(),
                message: e.to_string(),
            })?;

        Ok((decimal_slope, None))
    }

    /// Compute custom aggregation (sum by default).
    fn compute_custom_aggregation(
        &self,
        values: &[f64],
        params: Option<&serde_json::Value>,
        _config: &MetricsConfig,
    ) -> Result<(Decimal, Option<MetricConfidenceInterval>), MetricAgentRuntimeError> {
        let aggregation = params
            .and_then(|p| p.get("aggregation"))
            .and_then(|v| v.as_str())
            .unwrap_or("sum");

        let value = match aggregation {
            "sum" => values.iter().sum::<f64>(),
            "count" => values.len() as f64,
            "min" => values.iter().cloned().fold(f64::INFINITY, f64::min),
            "max" => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            "range" => {
                let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
                let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                max - min
            }
            _ => values.iter().sum::<f64>(),
        };

        let decimal_value = Decimal::try_from(value)
            .map_err(|e| MetricAgentRuntimeError::Computation {
                metric: "custom_aggregation".to_string(),
                message: e.to_string(),
            })?;

        Ok((decimal_value, None))
    }

    /// Round decimal to specified precision.
    fn round_to_precision(&self, value: Decimal, precision: u8) -> Decimal {
        value.round_dp(precision as u32)
    }

    /// Compute overall confidence for the metric computation.
    fn compute_overall_confidence(
        &self,
        metrics: &[ComputedMetric],
        total_sample_size: u64,
    ) -> MetricComputationConfidence {
        // Base confidence from sample size
        let sample_confidence = (total_sample_size as f64 / 1000.0).min(0.9);

        // Adjust based on number of metrics and missing data
        let total_missing: u64 = metrics.iter().map(|m| m.missing_count).sum();
        let missing_penalty = (total_missing as f64 / total_sample_size as f64).min(0.3);

        let value = (sample_confidence - missing_penalty).max(0.1);

        MetricComputationConfidence {
            value: Decimal::try_from(value).unwrap_or(dec!(0.5)),
            method: "sample_size_heuristic".to_string(),
            total_sample_size,
            ci_lower: None,
            ci_upper: None,
        }
    }

    /// Build constraints applied during computation.
    fn build_constraints(&self, input: &MetricsInput, warnings: &[String]) -> MetricConstraints {
        MetricConstraints {
            scope: vec![
                format!("context_id: {}", input.context_id),
                format!("metrics_count: {}", input.metrics_requested.len()),
            ],
            assumptions: vec![
                "Numeric data assumed for all computations".to_string(),
                "Independent observations assumed".to_string(),
            ],
            limitations: warnings.to_vec(),
            data_filters: vec![
                format!("handle_missing: {:?}", input.config.handle_missing),
            ],
        }
    }
}

impl Default for ExperimentalMetricAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceEstimator for ExperimentalMetricAgent {
    fn estimate_confidence(&self, sample_size: u64, _effect_size: Option<f64>) -> f64 {
        // Base confidence from sample size
        let size_confidence = (sample_size as f64 / 1000.0).min(0.9);
        size_confidence.max(0.1)
    }
}

impl PerformanceBounded for ExperimentalMetricAgent {
    fn budget(&self) -> &PerformanceBudget {
        &self.budget
    }
}

#[async_trait]
impl Agent for ExperimentalMetricAgent {
    type Input = MetricsInput;
    type Output = MetricsOutput;
    type Error = MetricAgentRuntimeError;

    fn identity(&self) -> &AgentIdentity {
        &self.identity
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), Self::Error> {
        // Validate against contracts
        input.validate()?;

        // Check for empty data
        if input.data.records.is_empty() {
            return Err(MetricAgentRuntimeError::EmptyData);
        }

        // Check for requested metrics
        if input.metrics_requested.is_empty() {
            return Err(MetricAgentRuntimeError::MissingRequired(
                "metrics_requested".to_string()
            ));
        }

        // Check max records limit
        if input.data.records.len() > self.config.max_records {
            return Err(MetricAgentRuntimeError::Validation(
                format!("Record count {} exceeds maximum {}", 
                    input.data.records.len(), self.config.max_records)
            ));
        }

        Ok(())
    }

    #[instrument(skip(self, input), fields(
        request_id = %input.request_id,
        context_id = %input.context_id,
        metrics_count = input.metrics_requested.len(),
        records_count = input.data.records.len()
    ))]
    async fn execute(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        info!("Executing metric computation");
        let start_time = Instant::now();

        // Emit telemetry: execution started
        if self.config.emit_telemetry {
            self.telemetry.emit(TelemetryEvent {
                event_type: TelemetryEventType::ExecutionStarted,
                agent_id: METRIC_AGENT_ID.to_string(),
                timestamp: Utc::now(),
                metadata: json!({
                    "request_id": input.request_id,
                    "metrics_count": input.metrics_requested.len(),
                    "records_count": input.data.records.len(),
                }),
            });
        }

        let mut metrics = Vec::with_capacity(input.metrics_requested.len());
        let mut warnings = Vec::new();

        // Compute each requested metric
        for request in &input.metrics_requested {
            match self.compute_metric(request, &input.data.records, &input.config) {
                Ok(metric) => {
                    debug!(
                        metric_name = %metric.name,
                        value = %metric.value,
                        sample_size = metric.sample_size,
                        "Metric computed successfully"
                    );
                    metrics.push(metric);
                }
                Err(e) => {
                    warn!(
                        metric_name = %request.name,
                        error = %e,
                        "Failed to compute metric"
                    );
                    warnings.push(format!("Failed to compute '{}': {}", request.name, e));
                }
            }
        }

        // Check if all metrics failed
        if metrics.is_empty() && !input.metrics_requested.is_empty() {
            return Err(MetricAgentRuntimeError::Computation {
                metric: "all".to_string(),
                message: "All metric computations failed".to_string(),
            });
        }

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        // Phase 7: Check latency budget BEFORE returning result
        if let Err(violation) = self.check_latency(processing_time_ms) {
            tracing::error!(
                elapsed_ms = processing_time_ms,
                budget_ms = self.budget.max_latency_ms,
                budget_type = %violation.budget_type,
                "Performance budget exceeded - ABORTING"
            );
            return Err(MetricAgentRuntimeError::BudgetExceeded(format!(
                "Latency budget exceeded: {}ms > {}ms limit",
                processing_time_ms, self.budget.max_latency_ms
            )));
        }

        let output = MetricsOutput {
            metrics,
            metadata: MetricsMetadata {
                computed_at: Utc::now(),
                records_processed: input.data.records.len() as u64,
                processing_time_ms,
            },
            warnings,
        };

        // Emit telemetry: execution completed
        if self.config.emit_telemetry {
            self.telemetry.emit(TelemetryEvent {
                event_type: TelemetryEventType::ExecutionCompleted,
                agent_id: METRIC_AGENT_ID.to_string(),
                timestamp: Utc::now(),
                metadata: json!({
                    "request_id": input.request_id,
                    "metrics_computed": output.metrics.len(),
                    "processing_time_ms": processing_time_ms,
                    "warnings_count": output.warnings.len(),
                }),
            });
        }

        info!(
            metrics_computed = output.metrics.len(),
            processing_time_ms = processing_time_ms,
            budget_ms = self.budget.max_latency_ms,
            warnings = output.warnings.len(),
            "Metric computation complete"
        );

        Ok(output)
    }

    fn build_decision_event(
        &self,
        input: &Self::Input,
        output: &Self::Output,
        execution_id: Uuid,
    ) -> Result<DecisionEvent, Self::Error> {
        // Compute inputs hash for determinism verification
        let inputs_hash = DecisionEvent::compute_inputs_hash(input)?;

        // Compute overall confidence
        let total_sample_size: u64 = output.metrics.iter().map(|m| m.sample_size).sum();
        let computation_confidence = self.compute_overall_confidence(&output.metrics, total_sample_size);

        let confidence = Confidence {
            value: computation_confidence.value,
            method: ConfidenceMethod::Heuristic,
            sample_size: Some(computation_confidence.total_sample_size),
            ci_lower: computation_confidence.ci_lower,
            ci_upper: computation_confidence.ci_upper,
        };

        // Build constraints
        let metric_constraints = self.build_constraints(input, &output.warnings);
        let constraints = ConstraintsApplied {
            scope: metric_constraints.scope,
            assumptions: metric_constraints.assumptions,
            limitations: metric_constraints.limitations,
            data_filters: metric_constraints.data_filters,
            temporal_bounds: None,
        };

        // Build execution ref
        let execution_ref = ExecutionRef {
            execution_id,
            trace_id: None, // Would be populated from tracing context
            span_id: None,
            parent_ref: None,
            runtime_version: Some(METRIC_AGENT_VERSION.to_string()),
        };

        // Serialize output
        let outputs = serde_json::to_value(output)?;

        DecisionEvent::builder()
            .agent_id(METRIC_AGENT_ID)
            .agent_version(METRIC_AGENT_VERSION)
            .decision_type(DecisionType::ExperimentalMetrics)
            .inputs_hash(inputs_hash)
            .outputs(outputs)
            .confidence(confidence)
            .constraints_applied(constraints)
            .execution_ref(execution_ref)
            .metadata(json!({
                "request_id": input.request_id,
                "context_id": input.context_id,
                "metrics_count": output.metrics.len(),
                "processing_time_ms": output.metadata.processing_time_ms,
            }))
            .build()
            .map_err(|e| MetricAgentRuntimeError::Internal(e.to_string()))
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_input(num_records: usize) -> MetricsInput {
        let records: Vec<serde_json::Value> = (0..num_records)
            .map(|i| {
                json!({
                    "value": (i as f64) * 0.1 + 1.0,
                    "accuracy": 0.5 + (i as f64) * 0.01,
                    "latency": 100.0 + (i as f64) * 10.0,
                })
            })
            .collect();

        MetricsInput {
            request_id: Uuid::new_v4(),
            context_id: "test-experiment-123".to_string(),
            metrics_requested: vec![
                MetricRequest {
                    name: "mean_accuracy".to_string(),
                    metric_type: MetricType::CentralTendency,
                    variable: "accuracy".to_string(),
                    group_by: None,
                    params: None,
                },
                MetricRequest {
                    name: "std_latency".to_string(),
                    metric_type: MetricType::Dispersion,
                    variable: "latency".to_string(),
                    group_by: None,
                    params: None,
                },
            ],
            data: MetricsData {
                source: "test-source".to_string(),
                records,
                schema: None,
            },
            config: MetricsConfig {
                handle_missing: MissingValueStrategy::Skip,
                precision: 4,
                include_ci: true,
                ci_level: Some(dec!(0.95)),
            },
        }
    }

    #[tokio::test]
    async fn test_metric_agent_execution() {
        let agent = ExperimentalMetricAgent::new();
        let input = create_test_input(100);

        let result = agent.execute(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert_eq!(output.metrics.len(), 2);
        assert!(output.warnings.is_empty());
    }

    #[tokio::test]
    async fn test_metric_agent_empty_data() {
        let agent = ExperimentalMetricAgent::new();
        let mut input = create_test_input(10);
        input.data.records.clear();

        let result = agent.validate_input(&input);
        assert!(matches!(result, Err(MetricAgentRuntimeError::EmptyData)));
    }

    #[tokio::test]
    async fn test_metric_agent_decision_event() {
        let agent = ExperimentalMetricAgent::new();
        let input = create_test_input(50);

        let (output, event) = agent.invoke(input).await.unwrap();

        assert_eq!(event.agent_id, METRIC_AGENT_ID);
        assert_eq!(event.agent_version, METRIC_AGENT_VERSION);
        assert_eq!(event.decision_type, DecisionType::ExperimentalMetrics);
        assert_eq!(event.inputs_hash.len(), 64);
        assert!(!output.metrics.is_empty());
    }

    #[test]
    fn test_confidence_estimator() {
        let agent = ExperimentalMetricAgent::new();

        let conf_small = agent.estimate_confidence(10, None);
        let conf_medium = agent.estimate_confidence(100, None);
        let conf_large = agent.estimate_confidence(1000, None);

        assert!(conf_small < conf_medium);
        assert!(conf_medium < conf_large);
    }

    #[tokio::test]
    async fn test_percentile_computation() {
        let agent = ExperimentalMetricAgent::new();
        let mut input = create_test_input(100);
        input.metrics_requested = vec![MetricRequest {
            name: "p90_latency".to_string(),
            metric_type: MetricType::Percentile,
            variable: "latency".to_string(),
            group_by: None,
            params: Some(json!({"percentile": 90.0})),
        }];

        let output = agent.execute(input).await.unwrap();
        assert_eq!(output.metrics.len(), 1);
        assert_eq!(output.metrics[0].name, "p90_latency");
    }

    #[tokio::test]
    async fn test_correlation_computation() {
        let agent = ExperimentalMetricAgent::new();
        let mut input = create_test_input(100);
        input.metrics_requested = vec![MetricRequest {
            name: "accuracy_latency_corr".to_string(),
            metric_type: MetricType::Correlation,
            variable: "accuracy".to_string(),
            group_by: None,
            params: Some(json!({"variable_y": "latency"})),
        }];

        let output = agent.execute(input).await.unwrap();
        assert_eq!(output.metrics.len(), 1);
        
        // Since our test data has a positive relationship, correlation should be positive
        let corr: f64 = output.metrics[0].value.try_into().unwrap();
        assert!(corr > 0.0);
    }

    #[test]
    fn test_extract_nested_values() {
        let agent = ExperimentalMetricAgent::new();
        let records = vec![
            json!({"metrics": {"accuracy": 0.9}}),
            json!({"metrics": {"accuracy": 0.85}}),
            json!({"metrics": {"accuracy": 0.95}}),
        ];

        let values = agent.extract_numeric_values(&records, "metrics.accuracy").unwrap();
        assert_eq!(values.len(), 3);
        assert!((values[0] - 0.9).abs() < 0.001);
    }
}
