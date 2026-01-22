//! Hypothesis Agent HTTP Handler
//!
//! Edge Function handler for hypothesis evaluation requests.
//!
//! # Constitution Compliance (PROMPT 2)
//!
//! Constraints (NON-NEGOTIABLE):
//! - Google Cloud Edge Function compatible
//! - Stateless execution
//! - Deterministic behavior
//! - No orchestration logic
//! - No optimization logic
//! - No direct SQL access
//! - Async, non-blocking writes via ruvector-service only
//!
//! This handler:
//! - Validates input
//! - Executes core hypothesis/metric logic
//! - Computes confidence
//! - Emits DecisionEvent to ruvector-service
//! - Emits telemetry
//! - Returns structured response

use serde::{Deserialize, Serialize};
use tracing::{error, info, instrument, span, Level};
use uuid::Uuid;

use crate::agents::{Agent, HypothesisAgent, HypothesisAgentError};
use crate::clients::{RuVectorClient, RuVectorError, RuVectorPersistence};
use crate::contracts::{HypothesisInput, HypothesisOutput, AgentError, DecisionEvent};
use crate::telemetry::TelemetryEmitter;

/// Request for hypothesis evaluation.
#[derive(Debug, Deserialize)]
pub struct HypothesisEvaluateRequest {
    /// The hypothesis evaluation input
    pub input: HypothesisInput,

    /// Optional trace context
    pub trace_context: Option<TraceContext>,
}

/// Trace context for distributed tracing.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TraceContext {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
}

/// Response from hypothesis evaluation.
#[derive(Debug, Serialize)]
pub struct HypothesisEvaluateResponse {
    /// Success indicator
    pub success: bool,

    /// Request ID for correlation
    pub request_id: Uuid,

    /// Evaluation output (if successful)
    pub output: Option<HypothesisOutput>,

    /// Decision event reference (for audit trail)
    pub decision_event_ref: Option<DecisionEventRef>,

    /// Error details (if failed)
    pub error: Option<AgentError>,
}

/// Reference to persisted decision event.
#[derive(Debug, Serialize)]
pub struct DecisionEventRef {
    pub event_id: Uuid,
    pub storage_ref: String,
}

/// Handler errors.
#[derive(Debug, thiserror::Error)]
pub enum HandlerError {
    #[error("Agent error: {0}")]
    Agent(#[from] HypothesisAgentError),

    #[error("Persistence error: {0}")]
    Persistence(#[from] RuVectorError),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    Configuration(String),
}

/// Hypothesis evaluation handler.
///
/// This is the main entry point for Edge Function deployment.
/// It orchestrates the full evaluation lifecycle:
///
/// 1. Validate input
/// 2. Execute agent logic
/// 3. Build DecisionEvent
/// 4. Persist to ruvector-service
/// 5. Emit telemetry
/// 6. Return response
pub struct HypothesisHandler {
    agent: HypothesisAgent,
    ruvector_client: RuVectorClient,
    telemetry: TelemetryEmitter,
}

impl HypothesisHandler {
    /// Create a new handler with default configuration.
    pub fn new() -> Result<Self, HandlerError> {
        let agent = HypothesisAgent::new();
        let ruvector_client = RuVectorClient::from_env()
            .map_err(|e| HandlerError::Configuration(e.to_string()))?;
        let telemetry = TelemetryEmitter::new();

        Ok(Self {
            agent,
            ruvector_client,
            telemetry,
        })
    }

    /// Create a handler with custom components (for testing).
    pub fn with_components(
        agent: HypothesisAgent,
        ruvector_client: RuVectorClient,
        telemetry: TelemetryEmitter,
    ) -> Self {
        Self {
            agent,
            ruvector_client,
            telemetry,
        }
    }

    /// Handle a hypothesis evaluation request.
    ///
    /// This is the primary handler method for Edge Function invocation.
    #[instrument(skip(self, request), fields(
        request_id = %request.input.request_id,
        hypothesis_id = %request.input.hypothesis.id
    ))]
    pub async fn handle(&self, request: HypothesisEvaluateRequest) -> HypothesisEvaluateResponse {
        let request_id = request.input.request_id;

        info!("Handling hypothesis evaluation request");

        // Create trace span if context provided
        let _span = if let Some(ref ctx) = request.trace_context {
            Some(span!(
                Level::INFO,
                "hypothesis_evaluation",
                trace_id = %ctx.trace_id,
                span_id = %ctx.span_id
            ))
        } else {
            None
        };

        // Execute agent with full lifecycle
        match self.execute_evaluation(request.input).await {
            Ok((output, event, storage_ref)) => {
                info!(
                    hypothesis_status = ?output.status,
                    event_id = %event.id,
                    "Hypothesis evaluation completed successfully"
                );

                // Emit success telemetry
                self.telemetry.emit_success(
                    &event,
                    request.trace_context.as_ref(),
                ).await;

                HypothesisEvaluateResponse {
                    success: true,
                    request_id,
                    output: Some(output),
                    decision_event_ref: Some(DecisionEventRef {
                        event_id: event.id,
                        storage_ref,
                    }),
                    error: None,
                }
            }
            Err(e) => {
                error!(error = %e, "Hypothesis evaluation failed");

                // Emit failure telemetry
                self.telemetry.emit_failure(
                    &e.to_string(),
                    request_id,
                    request.trace_context.as_ref(),
                ).await;

                HypothesisEvaluateResponse {
                    success: false,
                    request_id,
                    output: None,
                    decision_event_ref: None,
                    error: Some(AgentError {
                        code: error_code(&e),
                        message: e.to_string(),
                        request_id,
                        details: None,
                    }),
                }
            }
        }
    }

    /// Execute the full evaluation lifecycle.
    async fn execute_evaluation(
        &self,
        input: HypothesisInput,
    ) -> Result<(HypothesisOutput, DecisionEvent, String), HandlerError> {
        // 1. Validate and execute agent
        let (output, event) = self.agent.invoke(input).await?;

        // 2. Persist DecisionEvent to ruvector-service
        let persisted = self.ruvector_client.persist_decision_event(event.clone()).await?;

        Ok((output, event, persisted.storage_ref))
    }
}

/// Map errors to error codes.
fn error_code(error: &HandlerError) -> String {
    match error {
        HandlerError::Agent(HypothesisAgentError::Validation(_)) => "HYPOTHESIS_INPUT_INVALID".to_string(),
        HandlerError::Agent(HypothesisAgentError::InsufficientSampleSize { .. }) => "HYPOTHESIS_SAMPLE_SIZE".to_string(),
        HandlerError::Agent(HypothesisAgentError::StatisticalComputation(_)) => "HYPOTHESIS_COMPUTATION".to_string(),
        HandlerError::Agent(HypothesisAgentError::Configuration(_)) => "HYPOTHESIS_CONFIG".to_string(),
        HandlerError::Agent(HypothesisAgentError::Internal(_)) => "HYPOTHESIS_INTERNAL".to_string(),
        HandlerError::Persistence(_) => "HYPOTHESIS_PERSISTENCE".to_string(),
        HandlerError::Serialization(_) => "HYPOTHESIS_SERIALIZATION".to_string(),
        HandlerError::Configuration(_) => "HYPOTHESIS_CONFIG".to_string(),
    }
}

/// Health check handler.
pub async fn health_check(ruvector_client: &RuVectorClient) -> bool {
    match ruvector_client.health_check().await {
        Ok(healthy) => healthy,
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::hypothesis::*;
    use chrono::Utc;
    use rust_decimal_macros::dec;
    use serde_json::json;

    fn create_test_request() -> HypothesisEvaluateRequest {
        let observations: Vec<Observation> = (0..100)
            .map(|i| Observation {
                id: Uuid::new_v4(),
                values: json!({"value": (i as f64) * 0.1 + 0.5}),
                group: None,
                weight: None,
                timestamp: None,
            })
            .collect();

        HypothesisEvaluateRequest {
            input: HypothesisInput {
                request_id: Uuid::new_v4(),
                hypothesis: HypothesisDefinition {
                    id: Uuid::new_v4(),
                    name: "Test Hypothesis".to_string(),
                    statement: "Mean is greater than zero".to_string(),
                    hypothesis_type: HypothesisType::Threshold,
                    null_hypothesis: "Mean equals zero".to_string(),
                    alternative_hypothesis: "Mean is greater than zero".to_string(),
                    variables: vec![HypothesisVariable {
                        name: "value".to_string(),
                        role: VariableRole::Dependent,
                        data_type: VariableDataType::Continuous,
                        unit: None,
                    }],
                    expected_effect_size: Some(dec!(0.5)),
                    significance_level: dec!(0.05),
                    required_power: Some(dec!(0.8)),
                },
                experimental_data: ExperimentalData {
                    source_id: "test-source".to_string(),
                    collected_at: Utc::now(),
                    observations,
                    sample_size: 100,
                    quality_metrics: DataQualityMetrics {
                        completeness: dec!(1.0),
                        validity: dec!(1.0),
                        outlier_count: 0,
                        duplicate_count: 0,
                    },
                },
                config: EvaluationConfig {
                    test_method: StatisticalTest::TTest,
                    apply_correction: false,
                    correction_method: None,
                    bootstrap_iterations: None,
                    random_seed: Some(42),
                    compute_effect_size: true,
                    generate_diagnostics: true,
                },
                context: None,
            },
            trace_context: None,
        }
    }

    #[test]
    fn test_error_code_mapping() {
        let validation_error = HandlerError::Agent(HypothesisAgentError::Validation("test".to_string()));
        assert_eq!(error_code(&validation_error), "HYPOTHESIS_INPUT_INVALID");

        let sample_error = HandlerError::Agent(HypothesisAgentError::InsufficientSampleSize {
            required: 30,
            actual: 10,
        });
        assert_eq!(error_code(&sample_error), "HYPOTHESIS_SAMPLE_SIZE");
    }
}
