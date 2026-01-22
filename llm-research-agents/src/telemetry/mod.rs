//! Telemetry Module
//!
//! Telemetry emission compatible with LLM-Observatory.
//!
//! # Constitution Compliance
//!
//! Per PROMPT 0: All agents must emit telemetry compatible with LLM-Observatory.
//!
//! # Integration Points
//!
//! - LLM-Observatory: Telemetry, experiment traces, research metrics
//! - Distributed tracing: OpenTelemetry-compatible
//! - Metrics: Prometheus-compatible

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::contracts::DecisionEvent;
use crate::handlers::TraceContext;

/// Telemetry event types.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryEventType {
    /// Agent invocation started
    InvocationStarted,
    /// Agent invocation completed successfully
    InvocationCompleted,
    /// Agent invocation failed
    InvocationFailed,
    /// Decision event emitted
    DecisionEmitted,
    /// Metric computed
    MetricComputed,
}

/// Telemetry event for LLM-Observatory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    /// Event ID
    pub event_id: Uuid,

    /// Event type
    pub event_type: TelemetryEventType,

    /// Timestamp
    pub timestamp: DateTime<Utc>,

    /// Agent ID
    pub agent_id: String,

    /// Agent version
    pub agent_version: String,

    /// Request ID (for correlation)
    pub request_id: Option<Uuid>,

    /// Trace context (for distributed tracing)
    pub trace_context: Option<SerializableTraceContext>,

    /// Event payload
    pub payload: serde_json::Value,

    /// Duration in milliseconds (for completed/failed events)
    pub duration_ms: Option<u64>,
}

/// Serializable trace context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableTraceContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
}

impl From<&TraceContext> for SerializableTraceContext {
    fn from(ctx: &TraceContext) -> Self {
        Self {
            trace_id: ctx.trace_id.clone(),
            span_id: ctx.span_id.clone(),
            parent_span_id: ctx.parent_span_id.clone(),
        }
    }
}

/// Telemetry emitter for LLM-Observatory compatibility.
#[derive(Clone)]
pub struct TelemetryEmitter {
    /// Observatory endpoint (if configured)
    observatory_endpoint: Option<String>,

    /// Whether to emit to stdout (for development)
    emit_to_stdout: bool,
}

impl Default for TelemetryEmitter {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetryEmitter {
    /// Create a new telemetry emitter.
    pub fn new() -> Self {
        let observatory_endpoint = std::env::var("LLM_OBSERVATORY_ENDPOINT").ok();
        let emit_to_stdout = std::env::var("TELEMETRY_STDOUT")
            .map(|v| v == "true" || v == "1")
            .unwrap_or(true);

        Self {
            observatory_endpoint,
            emit_to_stdout,
        }
    }

    /// Create emitter with custom configuration.
    pub fn with_config(observatory_endpoint: Option<String>, emit_to_stdout: bool) -> Self {
        Self {
            observatory_endpoint,
            emit_to_stdout,
        }
    }

    /// Emit a success telemetry event.
    pub async fn emit_success(
        &self,
        decision_event: &DecisionEvent,
        trace_context: Option<&TraceContext>,
    ) {
        let event = TelemetryEvent {
            event_id: Uuid::new_v4(),
            event_type: TelemetryEventType::InvocationCompleted,
            timestamp: Utc::now(),
            agent_id: decision_event.agent_id.clone(),
            agent_version: decision_event.agent_version.clone(),
            request_id: decision_event.metadata
                .as_ref()
                .and_then(|m| m.get("request_id"))
                .and_then(|v| v.as_str())
                .and_then(|s| Uuid::parse_str(s).ok()),
            trace_context: trace_context.map(Into::into),
            payload: serde_json::json!({
                "decision_type": decision_event.decision_type.to_string(),
                "confidence": decision_event.confidence.value,
                "event_id": decision_event.id,
            }),
            duration_ms: None,
        };

        self.emit(event).await;

        // Also emit decision event telemetry
        let decision_telemetry = TelemetryEvent {
            event_id: Uuid::new_v4(),
            event_type: TelemetryEventType::DecisionEmitted,
            timestamp: Utc::now(),
            agent_id: decision_event.agent_id.clone(),
            agent_version: decision_event.agent_version.clone(),
            request_id: None,
            trace_context: trace_context.map(Into::into),
            payload: serde_json::json!({
                "decision_event_id": decision_event.id,
                "decision_type": decision_event.decision_type.to_string(),
                "inputs_hash": decision_event.inputs_hash,
            }),
            duration_ms: None,
        };

        self.emit(decision_telemetry).await;
    }

    /// Emit a failure telemetry event.
    pub async fn emit_failure(
        &self,
        error_message: &str,
        request_id: Uuid,
        trace_context: Option<&TraceContext>,
    ) {
        let event = TelemetryEvent {
            event_id: Uuid::new_v4(),
            event_type: TelemetryEventType::InvocationFailed,
            timestamp: Utc::now(),
            agent_id: crate::agents::HYPOTHESIS_AGENT_ID.to_string(),
            agent_version: crate::agents::HYPOTHESIS_AGENT_VERSION.to_string(),
            request_id: Some(request_id),
            trace_context: trace_context.map(Into::into),
            payload: serde_json::json!({
                "error": error_message,
            }),
            duration_ms: None,
        };

        self.emit(event).await;
    }

    /// Emit a telemetry event.
    async fn emit(&self, event: TelemetryEvent) {
        // Log to tracing
        match event.event_type {
            TelemetryEventType::InvocationFailed => {
                warn!(
                    event_type = ?event.event_type,
                    agent_id = %event.agent_id,
                    "Telemetry: invocation failed"
                );
            }
            _ => {
                info!(
                    event_type = ?event.event_type,
                    agent_id = %event.agent_id,
                    "Telemetry event"
                );
            }
        }

        // Emit to stdout if configured
        if self.emit_to_stdout {
            if let Ok(json) = serde_json::to_string(&event) {
                debug!(telemetry = %json);
            }
        }

        // Send to Observatory if configured
        if let Some(ref endpoint) = self.observatory_endpoint {
            self.send_to_observatory(endpoint, &event).await;
        }
    }

    /// Send telemetry to LLM-Observatory.
    async fn send_to_observatory(&self, endpoint: &str, event: &TelemetryEvent) {
        // In production, this would use an HTTP client
        // For now, just log that we would send it
        debug!(
            endpoint = %endpoint,
            event_id = %event.event_id,
            "Would send telemetry to Observatory"
        );

        // TODO: Implement actual HTTP send when Observatory is available
        // let client = reqwest::Client::new();
        // let _ = client
        //     .post(format!("{}/api/v1/telemetry", endpoint))
        //     .json(event)
        //     .send()
        //     .await;
    }
}

/// Metrics for Prometheus export.
#[derive(Debug, Clone)]
pub struct AgentMetrics {
    /// Total invocations
    pub invocations_total: u64,

    /// Successful invocations
    pub invocations_success: u64,

    /// Failed invocations
    pub invocations_failed: u64,

    /// Average latency in milliseconds
    pub latency_avg_ms: f64,

    /// P95 latency in milliseconds
    pub latency_p95_ms: f64,

    /// P99 latency in milliseconds
    pub latency_p99_ms: f64,
}

impl Default for AgentMetrics {
    fn default() -> Self {
        Self {
            invocations_total: 0,
            invocations_success: 0,
            invocations_failed: 0,
            latency_avg_ms: 0.0,
            latency_p95_ms: 0.0,
            latency_p99_ms: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_telemetry_emitter_creation() {
        let emitter = TelemetryEmitter::new();
        assert!(emitter.emit_to_stdout);
    }

    #[test]
    fn test_telemetry_event_serialization() {
        let event = TelemetryEvent {
            event_id: Uuid::new_v4(),
            event_type: TelemetryEventType::InvocationCompleted,
            timestamp: Utc::now(),
            agent_id: "test-agent".to_string(),
            agent_version: "1.0.0".to_string(),
            request_id: Some(Uuid::new_v4()),
            trace_context: None,
            payload: serde_json::json!({"test": "data"}),
            duration_ms: Some(100),
        };

        let json = serde_json::to_string(&event).expect("Serialization should succeed");
        assert!(json.contains("invocation_completed"));
    }
}
