//! Hypothesis Agent Contracts
//!
//! This module defines the input and output schemas for the Hypothesis Agent.
//!
//! # Agent Classification
//!
//! Agent Name: Hypothesis Agent
//! Purpose: Define, evaluate, and validate research hypotheses using structured
//!          experimental inputs and observed signals.
//! Classification: HYPOTHESIS EVALUATION
//!
//! # Scope
//!
//! - Define testable hypotheses
//! - Evaluate hypotheses against experimental data
//! - Emit structured hypothesis outcomes
//!
//! # decision_type
//!
//! "hypothesis_evaluation"
//!
//! # Non-Responsibilities (MUST NEVER)
//!
//! - Execute inference
//! - Modify prompts or responses
//! - Route inference requests
//! - Trigger orchestration or retries
//! - Apply optimizations automatically
//! - Enforce policies or governance decisions

use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use validator::Validate;

/// Hypothesis status in the evaluation lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisStatus {
    /// Hypothesis has been defined but not yet evaluated
    Defined,
    /// Hypothesis evaluation is in progress
    Evaluating,
    /// Hypothesis has been accepted based on evidence
    Accepted,
    /// Hypothesis has been rejected based on evidence
    Rejected,
    /// Evaluation was inconclusive
    Inconclusive,
    /// Hypothesis evaluation failed due to error
    Failed,
}

/// Type of hypothesis being evaluated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HypothesisType {
    /// Comparative hypothesis (A vs B)
    Comparative,
    /// Causal hypothesis (A causes B)
    Causal,
    /// Correlational hypothesis (A correlates with B)
    Correlational,
    /// Existence hypothesis (phenomenon X exists)
    Existence,
    /// Threshold hypothesis (metric exceeds threshold)
    Threshold,
    /// Trend hypothesis (metric follows trend)
    Trend,
}

/// Statistical test method for hypothesis evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StatisticalTest {
    /// Student's t-test
    TTest,
    /// Welch's t-test (unequal variances)
    WelchTTest,
    /// Chi-squared test
    ChiSquared,
    /// Mann-Whitney U test
    MannWhitneyU,
    /// Wilcoxon signed-rank test
    WilcoxonSignedRank,
    /// ANOVA (Analysis of Variance)
    Anova,
    /// Kruskal-Wallis H test
    KruskalWallis,
    /// Bayesian hypothesis testing
    BayesianTest,
    /// Bootstrap confidence interval
    BootstrapCI,
}

/// Input schema for hypothesis evaluation.
///
/// This is the primary input structure for the Hypothesis Agent.
/// All fields are validated according to agentics-contracts standards.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct HypothesisInput {
    /// Unique identifier for this hypothesis evaluation request
    pub request_id: Uuid,

    /// The hypothesis to evaluate
    #[validate(nested)]
    pub hypothesis: HypothesisDefinition,

    /// Experimental data for evaluation
    #[validate(nested)]
    pub experimental_data: ExperimentalData,

    /// Evaluation configuration
    #[validate(nested)]
    pub config: EvaluationConfig,

    /// Optional context from upstream systems
    pub context: Option<EvaluationContext>,
}

/// Definition of a testable hypothesis.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct HypothesisDefinition {
    /// Unique hypothesis ID
    pub id: Uuid,

    /// Human-readable hypothesis name
    #[validate(length(min = 1, max = 255))]
    pub name: String,

    /// Detailed hypothesis statement
    #[validate(length(min = 1, max = 2000))]
    pub statement: String,

    /// Type of hypothesis
    pub hypothesis_type: HypothesisType,

    /// Null hypothesis (H0) statement
    #[validate(length(min = 1, max = 1000))]
    pub null_hypothesis: String,

    /// Alternative hypothesis (H1) statement
    #[validate(length(min = 1, max = 1000))]
    pub alternative_hypothesis: String,

    /// Variables involved in the hypothesis
    #[validate(length(min = 1))]
    pub variables: Vec<HypothesisVariable>,

    /// Expected effect size (if known)
    pub expected_effect_size: Option<Decimal>,

    /// Significance level (alpha)
    pub significance_level: Decimal,

    /// Minimum required statistical power
    pub required_power: Option<Decimal>,
}

/// Variable definition for hypothesis testing.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct HypothesisVariable {
    /// Variable name
    #[validate(length(min = 1, max = 128))]
    pub name: String,

    /// Variable role (independent, dependent, confounding)
    pub role: VariableRole,

    /// Data type of the variable
    pub data_type: VariableDataType,

    /// Unit of measurement
    pub unit: Option<String>,
}

/// Role of a variable in hypothesis testing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VariableRole {
    Independent,
    Dependent,
    Confounding,
    Control,
}

/// Data type of a variable.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VariableDataType {
    Continuous,
    Discrete,
    Categorical,
    Ordinal,
    Binary,
}

/// Experimental data for hypothesis evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct ExperimentalData {
    /// Data source identifier
    #[validate(length(min = 1, max = 255))]
    pub source_id: String,

    /// Timestamp of data collection
    pub collected_at: DateTime<Utc>,

    /// Sample observations
    #[validate(length(min = 1))]
    pub observations: Vec<Observation>,

    /// Total sample size
    pub sample_size: u64,

    /// Data quality indicators
    pub quality_metrics: DataQualityMetrics,
}

/// Single observation in experimental data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Observation ID
    pub id: Uuid,

    /// Values for each variable
    pub values: serde_json::Value,

    /// Optional group/treatment assignment
    pub group: Option<String>,

    /// Observation weight (for weighted analysis)
    pub weight: Option<Decimal>,

    /// Timestamp of observation
    pub timestamp: Option<DateTime<Utc>>,
}

/// Data quality metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataQualityMetrics {
    /// Percentage of complete records (0.0 - 1.0)
    pub completeness: Decimal,

    /// Percentage of valid values (0.0 - 1.0)
    pub validity: Decimal,

    /// Number of outliers detected
    pub outlier_count: u64,

    /// Number of duplicate records
    pub duplicate_count: u64,
}

/// Configuration for hypothesis evaluation.
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct EvaluationConfig {
    /// Statistical test to use
    pub test_method: StatisticalTest,

    /// Whether to apply multiple testing correction
    pub apply_correction: bool,

    /// Correction method (if applicable)
    pub correction_method: Option<CorrectionMethod>,

    /// Number of bootstrap iterations (if using bootstrap)
    pub bootstrap_iterations: Option<u64>,

    /// Random seed for reproducibility
    pub random_seed: Option<u64>,

    /// Whether to compute effect size
    pub compute_effect_size: bool,

    /// Whether to generate diagnostic plots (metadata only)
    pub generate_diagnostics: bool,
}

/// Multiple testing correction method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionMethod {
    Bonferroni,
    Holm,
    BenjaminiHochberg,
    BenjaminiYekutieli,
}

/// Optional context from upstream systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationContext {
    /// Experiment ID from LLM-Research-Lab
    pub experiment_id: Option<Uuid>,

    /// Related prior hypothesis evaluations
    pub prior_evaluations: Vec<Uuid>,

    /// Telemetry reference from LLM-Observatory
    pub telemetry_ref: Option<String>,

    /// Cost signals from LLM-CostOps
    pub cost_signals: Option<serde_json::Value>,

    /// Performance signals from LLM-Latency-Lens
    pub performance_signals: Option<serde_json::Value>,
}

/// Output schema for hypothesis evaluation.
///
/// This is the structured output that will be included in the DecisionEvent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisOutput {
    /// Evaluated hypothesis ID
    pub hypothesis_id: Uuid,

    /// Final status of the hypothesis
    pub status: HypothesisStatus,

    /// Statistical test results
    pub test_results: TestResults,

    /// Effect size metrics
    pub effect_size: Option<EffectSize>,

    /// Diagnostic information
    pub diagnostics: DiagnosticInfo,

    /// Recommendations based on results
    pub recommendations: Vec<String>,
}

/// Statistical test results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResults {
    /// Test statistic value
    pub test_statistic: Decimal,

    /// P-value
    pub p_value: Decimal,

    /// Corrected p-value (if correction applied)
    pub corrected_p_value: Option<Decimal>,

    /// Degrees of freedom
    pub degrees_of_freedom: Option<Decimal>,

    /// Confidence interval for the estimate
    pub confidence_interval: Option<ConfidenceInterval>,

    /// Whether the null hypothesis is rejected
    pub null_rejected: bool,

    /// Decision based on configured significance level
    pub decision: String,
}

/// Confidence interval.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub lower: Decimal,
    pub upper: Decimal,
    pub level: Decimal,
}

/// Effect size metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSize {
    /// Effect size measure used
    pub measure: EffectSizeMeasure,

    /// Effect size value
    pub value: Decimal,

    /// Effect size interpretation
    pub interpretation: String,
}

/// Effect size measure type.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EffectSizeMeasure {
    CohensD,
    HedgesG,
    GlassDelta,
    EtaSquared,
    OmegaSquared,
    CramersV,
    PhiCoefficient,
    OddsRatio,
    RiskRatio,
}

/// Diagnostic information for the evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticInfo {
    /// Power of the test (post-hoc)
    pub achieved_power: Option<Decimal>,

    /// Sample size adequacy
    pub sample_adequacy: SampleAdequacy,

    /// Assumption violations detected
    pub assumption_violations: Vec<AssumptionViolation>,

    /// Warnings generated during evaluation
    pub warnings: Vec<String>,
}

/// Sample size adequacy assessment.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SampleAdequacy {
    Adequate,
    Marginal,
    Inadequate,
    Unknown,
}

/// Assumption violation in statistical testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssumptionViolation {
    /// Name of the assumption
    pub assumption: String,

    /// Test used to check assumption
    pub test_used: String,

    /// Severity of violation
    pub severity: ViolationSeverity,

    /// Recommendation for handling
    pub recommendation: String,
}

/// Severity of assumption violation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ViolationSeverity {
    Minor,
    Moderate,
    Severe,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_hypothesis_input_serialization() {
        let input = HypothesisInput {
            request_id: Uuid::new_v4(),
            hypothesis: HypothesisDefinition {
                id: Uuid::new_v4(),
                name: "Test Hypothesis".to_string(),
                statement: "Model A outperforms Model B".to_string(),
                hypothesis_type: HypothesisType::Comparative,
                null_hypothesis: "No difference between A and B".to_string(),
                alternative_hypothesis: "A performs better than B".to_string(),
                variables: vec![HypothesisVariable {
                    name: "accuracy".to_string(),
                    role: VariableRole::Dependent,
                    data_type: VariableDataType::Continuous,
                    unit: Some("percentage".to_string()),
                }],
                expected_effect_size: Some(dec!(0.5)),
                significance_level: dec!(0.05),
                required_power: Some(dec!(0.8)),
            },
            experimental_data: ExperimentalData {
                source_id: "experiment-001".to_string(),
                collected_at: Utc::now(),
                observations: vec![],
                sample_size: 100,
                quality_metrics: DataQualityMetrics {
                    completeness: dec!(0.98),
                    validity: dec!(0.99),
                    outlier_count: 2,
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
        };

        let json = serde_json::to_string(&input).expect("Serialization failed");
        let deserialized: HypothesisInput = serde_json::from_str(&json).expect("Deserialization failed");

        assert_eq!(input.hypothesis.name, deserialized.hypothesis.name);
    }

    #[test]
    fn test_hypothesis_status() {
        let status = HypothesisStatus::Accepted;
        let json = serde_json::to_string(&status).expect("Serialization failed");
        assert_eq!(json, "\"accepted\"");
    }
}
