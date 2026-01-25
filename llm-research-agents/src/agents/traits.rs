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

// =============================================================================
// Phase 7 Performance Budgets (MANDATORY)
// =============================================================================

/// Performance budget configuration for agents.
///
/// All agents MUST enforce these budgets and abort execution if any limit is exceeded.
/// This ensures predictable resource usage and prevents runaway operations.
#[derive(Debug, Clone)]
pub struct PerformanceBudget {
    /// Maximum tokens (input + output) per execution
    pub max_tokens: usize,
    /// Maximum latency in milliseconds
    pub max_latency_ms: u64,
    /// Maximum LLM API calls per execution
    pub max_calls_per_run: u32,
}

impl Default for PerformanceBudget {
    fn default() -> Self {
        Self {
            max_tokens: 2500,
            max_latency_ms: 5000,
            max_calls_per_run: 5,
        }
    }
}

impl PerformanceBudget {
    /// Create a strict budget for latency-sensitive operations.
    pub fn strict() -> Self {
        Self {
            max_tokens: 1000,
            max_latency_ms: 2000,
            max_calls_per_run: 2,
        }
    }

    /// Create a relaxed budget for complex operations.
    pub fn relaxed() -> Self {
        Self {
            max_tokens: 5000,
            max_latency_ms: 10000,
            max_calls_per_run: 10,
        }
    }
}

/// Performance budget violation error.
///
/// This error is returned when an agent exceeds its allocated budget.
/// Agents MUST NOT retry automatically after a budget violation.
#[derive(Debug, Clone)]
pub struct BudgetViolation {
    /// Type of budget that was violated (latency, tokens, calls)
    pub budget_type: String,
    /// The configured limit
    pub limit: u64,
    /// The actual observed value
    pub actual: u64,
    /// Human-readable error message
    pub message: String,
}

impl std::fmt::Display for BudgetViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "BudgetViolation[{}]: {} (limit={}, actual={})",
            self.budget_type, self.message, self.limit, self.actual)
    }
}

impl std::error::Error for BudgetViolation {}

/// Trait for performance-bounded agents.
///
/// All agents MUST implement this trait to enforce resource limits.
/// Budget checks MUST be performed before returning results.
pub trait PerformanceBounded {
    /// Get the agent's performance budget.
    fn budget(&self) -> &PerformanceBudget;

    /// Check if latency is within budget.
    ///
    /// Returns `Ok(())` if within budget, or `Err(BudgetViolation)` if exceeded.
    fn check_latency(&self, elapsed_ms: u64) -> Result<(), BudgetViolation> {
        let budget = self.budget();
        if elapsed_ms > budget.max_latency_ms {
            return Err(BudgetViolation {
                budget_type: "latency".to_string(),
                limit: budget.max_latency_ms,
                actual: elapsed_ms,
                message: format!(
                    "Latency {}ms exceeds budget {}ms",
                    elapsed_ms, budget.max_latency_ms
                ),
            });
        }
        Ok(())
    }

    /// Check if token count is within budget.
    ///
    /// Returns `Ok(())` if within budget, or `Err(BudgetViolation)` if exceeded.
    fn check_tokens(&self, token_count: usize) -> Result<(), BudgetViolation> {
        let budget = self.budget();
        if token_count > budget.max_tokens {
            return Err(BudgetViolation {
                budget_type: "tokens".to_string(),
                limit: budget.max_tokens as u64,
                actual: token_count as u64,
                message: format!(
                    "Token count {} exceeds budget {}",
                    token_count, budget.max_tokens
                ),
            });
        }
        Ok(())
    }

    /// Check if API call count is within budget.
    ///
    /// Returns `Ok(())` if within budget, or `Err(BudgetViolation)` if exceeded.
    fn check_calls(&self, call_count: u32) -> Result<(), BudgetViolation> {
        let budget = self.budget();
        if call_count > budget.max_calls_per_run {
            return Err(BudgetViolation {
                budget_type: "api_calls".to_string(),
                limit: budget.max_calls_per_run as u64,
                actual: call_count as u64,
                message: format!(
                    "API call count {} exceeds budget {}",
                    call_count, budget.max_calls_per_run
                ),
            });
        }
        Ok(())
    }
}
