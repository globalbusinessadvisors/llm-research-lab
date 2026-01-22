//! Metric Agent Handler
//!
//! HTTP handler for the Experimental Metric Agent endpoint.
//!
//! # Constitution Compliance
//!
//! Per PROMPT 2 (RUNTIME & INFRASTRUCTURE IMPLEMENTATION):
//!
//! - Handler is stateless
//! - Handler is deterministic
//! - No orchestration logic
//! - No optimization logic
//! - No direct SQL access
//! - Async, non-blocking writes via ruvector-service only
//!
//! # Endpoint
//!
//! `POST /api/v1/agents/metric`
//!
//! # Request Format
//!
//! ```json
//! {
//!   "request_id": "uuid",
//!   "context_id": "experiment-123",
//!   "metrics_requested": [...],
//!   "data": {...},
//!   "config": {...}
//! }
//! ```
//!
//! # Response Format
//!
//! ```json
//! {
//!   "success": true,
//!   "request_id": "uuid",
//!   "output": {...},
//!   "decision_event": {...}
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{error, info, instrument, warn};
use uuid::Uuid;
use validator::Validate;

use crate::agents::{Agent, ExperimentalMetricAgent, METRIC_AGENT_ID, METRIC_AGENT_VERSION};
use crate::clients::{RuVectorClient, RuVectorPersistence};
use crate::contracts::metrics::{MetricsInput, MetricsOutput};
use crate::contracts::DecisionEvent;
use crate::telemetry::TelemetryEmitter;

/// Trace context for distributed tracing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricTraceContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
}

/// Request for metric computation.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct MetricComputeRequest {
    /// Metrics input
    #[validate(nested)]
    pub input: MetricsInput,

    /// Optional trace context for distributed tracing
    pub trace_context: Option<MetricTraceContext>,
}

/// Response from metric computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComputeResponse {
    /// Request succeeded
    pub success: bool,

    /// Original request ID
    pub request_id: Uuid,

    /// Computation output (if successful)
    pub output: Option<MetricsOutput>,

    /// Decision event (if successful)
    pub decision_event: Option<DecisionEventSummary>,

    /// Error message (if failed)
    pub error: Option<String>,

    /// Error code (if failed)
    pub error_code: Option<String>,

    /// Processing time in milliseconds
    pub processing_time_ms: u64,
}

/// Summary of decision event (without full outputs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionEventSummary {
    pub id: Uuid,
    pub agent_id: String,
    pub agent_version: String,
    pub decision_type: String,
    pub confidence: String,
    pub storage_ref: Option<String>,
}

/// Handler for metric computation requests.
///
/// This handler is designed to be deployed as a Google Cloud Edge Function.
/// It is stateless and all persistence is done via ruvector-service.
pub struct MetricHandler {
    agent: ExperimentalMetricAgent,
    ruvector: Option<RuVectorClient>,
    telemetry: TelemetryEmitter,
}

impl MetricHandler {
    /// Create a new metric handler.
    pub fn new() -> Self {
        let ruvector = RuVectorClient::from_env().ok();

        Self {
            agent: ExperimentalMetricAgent::new(),
            ruvector,
            telemetry: TelemetryEmitter::new(),
        }
    }

    /// Create handler with custom configuration.
    pub fn with_config(ruvector: Option<RuVectorClient>, telemetry: TelemetryEmitter) -> Self {
        Self {
            agent: ExperimentalMetricAgent::new(),
            ruvector,
            telemetry,
        }
    }

    /// Handle a metric computation request.
    ///
    /// This is the primary entry point for the Edge Function.
    #[instrument(skip(self, request), fields(
        request_id = %request.input.request_id,
        context_id = %request.input.context_id,
        metrics_count = request.input.metrics_requested.len()
    ))]
    pub async fn handle(&self, request: MetricComputeRequest) -> MetricComputeResponse {
        let start_time = Instant::now();
        let request_id = request.input.request_id;

        info!("Handling metric computation request");

        // Validate request
        if let Err(e) = request.validate() {
            error!(error = %e, "Request validation failed");
            return MetricComputeResponse {
                success: false,
                request_id,
                output: None,
                decision_event: None,
                error: Some(format!("Validation error: {}", e)),
                error_code: Some("METRIC_INPUT_INVALID".to_string()),
                processing_time_ms: start_time.elapsed().as_millis() as u64,
            };
        }

        // Execute agent
        let (output, event) = match self.agent.invoke(request.input).await {
            Ok(result) => result,
            Err(e) => {
                error!(error = %e, "Metric computation failed");
                return MetricComputeResponse {
                    success: false,
                    request_id,
                    output: None,
                    decision_event: None,
                    error: Some(format!("Computation error: {}", e)),
                    error_code: Some("METRIC_COMPUTATION_FAILED".to_string()),
                    processing_time_ms: start_time.elapsed().as_millis() as u64,
                };
            }
        };

        // Persist decision event
        let storage_ref = self.persist_decision_event(&event).await;

        // Build response
        let decision_summary = DecisionEventSummary {
            id: event.id,
            agent_id: event.agent_id.clone(),
            agent_version: event.agent_version.clone(),
            decision_type: event.decision_type.to_string(),
            confidence: event.confidence.value.to_string(),
            storage_ref,
        };

        let processing_time_ms = start_time.elapsed().as_millis() as u64;

        info!(
            metrics_computed = output.metrics.len(),
            processing_time_ms = processing_time_ms,
            "Metric computation completed"
        );

        MetricComputeResponse {
            success: true,
            request_id,
            output: Some(output),
            decision_event: Some(decision_summary),
            error: None,
            error_code: None,
            processing_time_ms,
        }
    }

    /// Persist decision event to ruvector-service.
    async fn persist_decision_event(&self, event: &DecisionEvent) -> Option<String> {
        let Some(ref client) = self.ruvector else {
            warn!("RuVector client not configured, skipping persistence");
            return None;
        };

        match client.persist_decision_event(event.clone()).await {
            Ok(persisted) => {
                info!(
                    storage_ref = %persisted.storage_ref,
                    "Decision event persisted"
                );
                Some(persisted.storage_ref)
            }
            Err(e) => {
                error!(error = %e, "Failed to persist decision event");
                None
            }
        }
    }

    /// Get agent identity information.
    pub fn agent_info(&self) -> AgentInfo {
        AgentInfo {
            id: METRIC_AGENT_ID.to_string(),
            version: METRIC_AGENT_VERSION.to_string(),
            classification: "EXPERIMENTAL_METRICS".to_string(),
            endpoint: "/api/v1/agents/metric".to_string(),
        }
    }
}

impl Default for MetricHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Agent information response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: String,
    pub version: String,
    pub classification: String,
    pub endpoint: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::metrics::*;
    use rust_decimal_macros::dec;

    fn create_test_request() -> MetricComputeRequest {
        let records: Vec<serde_json::Value> = (0..50)
            .map(|i| {
                serde_json::json!({
                    "accuracy": 0.5 + (i as f64) * 0.01,
                    "latency": 100.0 + (i as f64) * 5.0,
                })
            })
            .collect();

        MetricComputeRequest {
            input: MetricsInput {
                request_id: Uuid::new_v4(),
                context_id: "test-experiment".to_string(),
                metrics_requested: vec![
                    MetricRequest {
                        name: "mean_accuracy".to_string(),
                        metric_type: MetricType::CentralTendency,
                        variable: "accuracy".to_string(),
                        group_by: None,
                        params: None,
                    },
                ],
                data: MetricsData {
                    source: "test".to_string(),
                    records,
                    schema: None,
                },
                config: MetricsConfig {
                    handle_missing: MissingValueStrategy::Skip,
                    precision: 4,
                    include_ci: true,
                    ci_level: Some(dec!(0.95)),
                },
            },
            trace_context: None,
        }
    }

    #[tokio::test]
    async fn test_metric_handler_success() {
        let handler = MetricHandler::new();
        let request = create_test_request();

        let response = handler.handle(request).await;

        assert!(response.success);
        assert!(response.output.is_some());
        assert!(response.decision_event.is_some());
        assert!(response.error.is_none());

        let output = response.output.unwrap();
        assert_eq!(output.metrics.len(), 1);
        assert_eq!(output.metrics[0].name, "mean_accuracy");
    }

    #[tokio::test]
    async fn test_metric_handler_empty_data() {
        let handler = MetricHandler::new();
        let mut request = create_test_request();
        request.input.data.records.clear();

        let response = handler.handle(request).await;

        assert!(!response.success);
        assert!(response.error.is_some());
        assert_eq!(response.error_code, Some("METRIC_COMPUTATION_FAILED".to_string()));
    }

    #[test]
    fn test_agent_info() {
        let handler = MetricHandler::new();
        let info = handler.agent_info();

        assert_eq!(info.id, METRIC_AGENT_ID);
        assert_eq!(info.classification, "EXPERIMENTAL_METRICS");
    }
}
