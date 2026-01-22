//! Agent Telemetry Module
//!
//! Provides telemetry emission capabilities compatible with LLM-Observatory.
//!
//! # Constitution Compliance
//!
//! Per PROMPT 0: All agents must emit telemetry compatible with LLM-Observatory.
//! This module provides the standardized telemetry interface for all research agents.
//!
//! # Integration Points
//!
//! Telemetry emitted by this module can be consumed by:
//! - LLM-Observatory (primary consumer)
//! - LLM-CostOps (cost signal analysis)
//! - Governance systems
//! - Audit systems
//! - Research dashboards

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, span, Level};
use uuid::Uuid;

/// Telemetry event types for agent operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TelemetryEventType {
    /// Agent execution started
    ExecutionStarted,
    /// Agent execution completed successfully
    ExecutionCompleted,
    /// Agent execution failed
    ExecutionFailed,
    /// Validation started
    ValidationStarted,
    /// Validation completed
    ValidationCompleted,
    /// Validation failed
    ValidationFailed,
    /// Decision event built
    DecisionEventBuilt,
    /// Decision event persisted
    DecisionEventPersisted,
    /// Metric computed
    MetricComputed,
    /// Hypothesis evaluated
    HypothesisEvaluated,
    /// Warning generated
    WarningGenerated,
    /// Custom event type
    Custom(String),
}

impl std::fmt::Display for TelemetryEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ExecutionStarted => write!(f, "execution_started"),
            Self::ExecutionCompleted => write!(f, "execution_completed"),
            Self::ExecutionFailed => write!(f, "execution_failed"),
            Self::ValidationStarted => write!(f, "validation_started"),
            Self::ValidationCompleted => write!(f, "validation_completed"),
            Self::ValidationFailed => write!(f, "validation_failed"),
            Self::DecisionEventBuilt => write!(f, "decision_event_built"),
            Self::DecisionEventPersisted => write!(f, "decision_event_persisted"),
            Self::MetricComputed => write!(f, "metric_computed"),
            Self::HypothesisEvaluated => write!(f, "hypothesis_evaluated"),
            Self::WarningGenerated => write!(f, "warning_generated"),
            Self::Custom(name) => write!(f, "custom_{}", name),
        }
    }
}

/// Telemetry event structure compatible with LLM-Observatory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    /// Event type
    pub event_type: TelemetryEventType,
    /// Agent ID that emitted the event
    pub agent_id: String,
    /// Timestamp of the event
    pub timestamp: DateTime<Utc>,
    /// Additional metadata
    pub metadata: serde_json::Value,
}

impl TelemetryEvent {
    /// Create a new telemetry event.
    pub fn new(
        event_type: TelemetryEventType,
        agent_id: impl Into<String>,
        metadata: serde_json::Value,
    ) -> Self {
        Self {
            event_type,
            agent_id: agent_id.into(),
            timestamp: Utc::now(),
            metadata,
        }
    }

    /// Create an execution started event.
    pub fn execution_started(agent_id: impl Into<String>, request_id: Uuid) -> Self {
        Self::new(
            TelemetryEventType::ExecutionStarted,
            agent_id,
            serde_json::json!({
                "request_id": request_id,
            }),
        )
    }

    /// Create an execution completed event.
    pub fn execution_completed(
        agent_id: impl Into<String>,
        request_id: Uuid,
        duration_ms: u64,
    ) -> Self {
        Self::new(
            TelemetryEventType::ExecutionCompleted,
            agent_id,
            serde_json::json!({
                "request_id": request_id,
                "duration_ms": duration_ms,
            }),
        )
    }

    /// Create an execution failed event.
    pub fn execution_failed(
        agent_id: impl Into<String>,
        request_id: Uuid,
        error: impl Into<String>,
    ) -> Self {
        Self::new(
            TelemetryEventType::ExecutionFailed,
            agent_id,
            serde_json::json!({
                "request_id": request_id,
                "error": error.into(),
            }),
        )
    }
}

/// Telemetry severity levels for compatibility with LLM-Observatory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TelemetrySeverity {
    Debug,
    Info,
    Warn,
    Error,
    Critical,
}

/// Agent telemetry emitter.
///
/// Provides standardized telemetry emission for LLM-Research-Lab agents.
/// Telemetry is emitted via structured logging (tracing) which can be
/// collected by LLM-Observatory.
#[derive(Debug, Clone)]
pub struct AgentTelemetry {
    /// Agent ID for correlation
    agent_id: String,
    /// Enable telemetry emission
    enabled: bool,
}

impl AgentTelemetry {
    /// Create a new telemetry emitter for an agent.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            enabled: true,
        }
    }

    /// Create a disabled telemetry emitter (for testing).
    pub fn disabled(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            enabled: false,
        }
    }

    /// Check if telemetry is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Enable telemetry.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable telemetry.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Emit a telemetry event.
    ///
    /// Events are emitted via structured logging (tracing) for collection
    /// by LLM-Observatory and other telemetry consumers.
    pub fn emit(&self, event: TelemetryEvent) {
        if !self.enabled {
            return;
        }

        // Create a span for the telemetry event
        let span = span!(
            Level::INFO,
            "agent_telemetry",
            agent_id = %self.agent_id,
            event_type = %event.event_type,
        );
        let _guard = span.enter();

        // Log the event based on type
        match event.event_type {
            TelemetryEventType::ExecutionStarted => {
                info!(
                    event_type = %event.event_type,
                    metadata = %event.metadata,
                    "Agent execution started"
                );
            }
            TelemetryEventType::ExecutionCompleted => {
                info!(
                    event_type = %event.event_type,
                    metadata = %event.metadata,
                    "Agent execution completed"
                );
            }
            TelemetryEventType::ExecutionFailed => {
                tracing::error!(
                    event_type = %event.event_type,
                    metadata = %event.metadata,
                    "Agent execution failed"
                );
            }
            TelemetryEventType::ValidationFailed => {
                tracing::warn!(
                    event_type = %event.event_type,
                    metadata = %event.metadata,
                    "Validation failed"
                );
            }
            TelemetryEventType::WarningGenerated => {
                tracing::warn!(
                    event_type = %event.event_type,
                    metadata = %event.metadata,
                    "Warning generated"
                );
            }
            _ => {
                debug!(
                    event_type = %event.event_type,
                    metadata = %event.metadata,
                    "Telemetry event"
                );
            }
        }

        // Also emit as a metrics counter for LLM-Observatory
        self.emit_metric_counter(&event);
    }

    /// Emit a metric counter for the event.
    fn emit_metric_counter(&self, event: &TelemetryEvent) {
        // Use the metrics crate to emit counters that LLM-Observatory can collect
        // Note: metrics::counter! requires owned strings for labels
        let agent_id = self.agent_id.clone();
        let event_type = event.event_type.to_string();

        metrics::counter!(
            "agent_telemetry_events_total",
            "agent_id" => agent_id.clone(),
            "event_type" => event_type
        ).increment(1);

        // Emit duration metric if available
        if let Some(duration_ms) = event.metadata.get("duration_ms").and_then(|v| v.as_u64()) {
            metrics::histogram!(
                "agent_execution_duration_ms",
                "agent_id" => agent_id
            ).record(duration_ms as f64);
        }
    }

    /// Emit execution started telemetry.
    pub fn execution_started(&self, request_id: Uuid, metadata: serde_json::Value) {
        let mut merged = serde_json::json!({
            "request_id": request_id,
        });
        if let (Some(base), Some(extra)) = (merged.as_object_mut(), metadata.as_object()) {
            base.extend(extra.clone());
        }
        self.emit(TelemetryEvent {
            event_type: TelemetryEventType::ExecutionStarted,
            agent_id: self.agent_id.clone(),
            timestamp: Utc::now(),
            metadata: merged,
        });
    }

    /// Emit execution completed telemetry.
    pub fn execution_completed(&self, request_id: Uuid, duration_ms: u64, result_metadata: serde_json::Value) {
        self.emit(TelemetryEvent {
            event_type: TelemetryEventType::ExecutionCompleted,
            agent_id: self.agent_id.clone(),
            timestamp: Utc::now(),
            metadata: serde_json::json!({
                "request_id": request_id,
                "duration_ms": duration_ms,
                "result": result_metadata,
            }),
        });
    }

    /// Emit execution failed telemetry.
    pub fn execution_failed(&self, request_id: Uuid, error: &str) {
        self.emit(TelemetryEvent {
            event_type: TelemetryEventType::ExecutionFailed,
            agent_id: self.agent_id.clone(),
            timestamp: Utc::now(),
            metadata: serde_json::json!({
                "request_id": request_id,
                "error": error,
            }),
        });
    }

    /// Emit decision event persisted telemetry.
    pub fn decision_event_persisted(&self, event_id: Uuid, storage_ref: &str) {
        self.emit(TelemetryEvent {
            event_type: TelemetryEventType::DecisionEventPersisted,
            agent_id: self.agent_id.clone(),
            timestamp: Utc::now(),
            metadata: serde_json::json!({
                "event_id": event_id,
                "storage_ref": storage_ref,
            }),
        });
    }

    /// Emit metric computed telemetry.
    pub fn metric_computed(&self, metric_name: &str, value: &str, sample_size: u64) {
        self.emit(TelemetryEvent {
            event_type: TelemetryEventType::MetricComputed,
            agent_id: self.agent_id.clone(),
            timestamp: Utc::now(),
            metadata: serde_json::json!({
                "metric_name": metric_name,
                "value": value,
                "sample_size": sample_size,
            }),
        });
    }

    /// Emit warning telemetry.
    pub fn warning(&self, message: &str, context: serde_json::Value) {
        self.emit(TelemetryEvent {
            event_type: TelemetryEventType::WarningGenerated,
            agent_id: self.agent_id.clone(),
            timestamp: Utc::now(),
            metadata: serde_json::json!({
                "message": message,
                "context": context,
            }),
        });
    }
}

impl Default for AgentTelemetry {
    fn default() -> Self {
        Self::new("unknown-agent")
    }
}

// =============================================================================
// Telemetry Span Helpers
// =============================================================================

/// Create a telemetry span for agent execution.
#[macro_export]
macro_rules! agent_span {
    ($agent_id:expr, $operation:expr) => {
        tracing::span!(
            tracing::Level::INFO,
            "agent_operation",
            agent_id = $agent_id,
            operation = $operation,
            otel.kind = "internal"
        )
    };
}

/// Create a telemetry span for metric computation.
#[macro_export]
macro_rules! metric_span {
    ($agent_id:expr, $metric_name:expr) => {
        tracing::span!(
            tracing::Level::DEBUG,
            "metric_computation",
            agent_id = $agent_id,
            metric_name = $metric_name,
            otel.kind = "internal"
        )
    };
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_telemetry_event_creation() {
        let event = TelemetryEvent::execution_started("test-agent", Uuid::new_v4());
        assert_eq!(event.event_type, TelemetryEventType::ExecutionStarted);
        assert_eq!(event.agent_id, "test-agent");
    }

    #[test]
    fn test_telemetry_event_type_display() {
        assert_eq!(TelemetryEventType::ExecutionStarted.to_string(), "execution_started");
        assert_eq!(TelemetryEventType::MetricComputed.to_string(), "metric_computed");
        assert_eq!(TelemetryEventType::Custom("test".to_string()).to_string(), "custom_test");
    }

    #[test]
    fn test_agent_telemetry_enable_disable() {
        let mut telemetry = AgentTelemetry::new("test-agent");
        assert!(telemetry.is_enabled());

        telemetry.disable();
        assert!(!telemetry.is_enabled());

        telemetry.enable();
        assert!(telemetry.is_enabled());
    }

    #[test]
    fn test_disabled_telemetry() {
        let telemetry = AgentTelemetry::disabled("test-agent");
        assert!(!telemetry.is_enabled());

        // This should not panic even when disabled
        telemetry.emit(TelemetryEvent::new(
            TelemetryEventType::ExecutionStarted,
            "test-agent",
            serde_json::json!({}),
        ));
    }

    #[test]
    fn test_telemetry_serialization() {
        let event = TelemetryEvent {
            event_type: TelemetryEventType::MetricComputed,
            agent_id: "test-agent".to_string(),
            timestamp: Utc::now(),
            metadata: serde_json::json!({
                "metric_name": "accuracy",
                "value": 0.95,
            }),
        };

        let json = serde_json::to_string(&event).expect("Serialization failed");
        assert!(json.contains("metric_computed"));
        assert!(json.contains("test-agent"));
    }
}
