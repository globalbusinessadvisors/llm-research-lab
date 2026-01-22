//! Agent Traits
//!
//! Common traits that all LLM-Research-Lab agents must implement.

use async_trait::async_trait;
use uuid::Uuid;

use crate::contracts::{AgentIdentity, DecisionEvent};

/// Trait for all LLM-Research-Lab agents.
///
/// # Constitution Requirements
///
/// Every agent MUST:
/// - Import schemas exclusively from agentics-contracts
/// - Validate all inputs and outputs against contracts
/// - Emit telemetry compatible with LLM-Observatory
/// - Emit exactly ONE DecisionEvent per invocation
/// - Be deployable as a Google Edge Function
/// - Return deterministic, machine-readable output
#[async_trait]
pub trait Agent: Send + Sync {
    /// Input type for this agent
    type Input: Clone + Send + Sync;

    /// Output type for this agent
    type Output: Clone + Send + Sync;

    /// Error type for this agent
    type Error: std::error::Error + Send + Sync;

    /// Get the agent's identity.
    fn identity(&self) -> &AgentIdentity;

    /// Get the agent's version.
    fn version(&self) -> &str {
        &self.identity().version
    }

    /// Get the agent's ID.
    fn agent_id(&self) -> &str {
        &self.identity().id
    }

    /// Validate input according to contracts.
    fn validate_input(&self, input: &Self::Input) -> Result<(), Self::Error>;

    /// Execute the agent's core logic.
    ///
    /// This method performs the actual hypothesis evaluation or metrics computation.
    /// It MUST be deterministic for the same input.
    async fn execute(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;

    /// Build a DecisionEvent from the execution output.
    ///
    /// This is called after successful execution to create the event
    /// that will be persisted to ruvector-service.
    fn build_decision_event(
        &self,
        input: &Self::Input,
        output: &Self::Output,
        execution_id: Uuid,
    ) -> Result<DecisionEvent, Self::Error>;

    /// Full invocation cycle: validate, execute, build event.
    ///
    /// This is the primary entry point for agent invocation.
    async fn invoke(&self, input: Self::Input) -> Result<(Self::Output, DecisionEvent), Self::Error> {
        // Validate input
        self.validate_input(&input)?;

        // Execute core logic
        let output = self.execute(input.clone()).await?;

        // Build decision event
        let execution_id = Uuid::new_v4();
        let event = self.build_decision_event(&input, &output, execution_id)?;

        Ok((output, event))
    }
}

/// Trait for agents that support batch operations.
#[async_trait]
pub trait BatchAgent: Agent {
    /// Execute multiple inputs in batch.
    async fn execute_batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>, Self::Error>;
}

/// Trait for agents that can estimate their confidence.
pub trait ConfidenceEstimator {
    /// Estimate confidence based on input characteristics.
    fn estimate_confidence(&self, sample_size: u64, effect_size: Option<f64>) -> f64;
}
