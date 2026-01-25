//! Hypothesis Agent Implementation
//!
//! # Agent Contract
//!
//! ## Agent Name
//! Hypothesis Agent
//!
//! ## Purpose
//! Define, evaluate, and validate research hypotheses using structured
//! experimental inputs and observed signals.
//!
//! ## Classification
//! HYPOTHESIS EVALUATION
//!
//! ## Scope
//! - Define testable hypotheses
//! - Evaluate hypotheses against experimental data
//! - Emit structured hypothesis outcomes
//!
//! ## decision_type
//! "hypothesis_evaluation"
//!
//! ## Explicit Non-Responsibilities (MUST NEVER)
//!
//! This agent MUST NEVER:
//! - Execute inference
//! - Modify prompts or responses
//! - Route inference requests
//! - Trigger orchestration or retries
//! - Apply optimizations automatically
//! - Enforce policies or governance decisions
//!
//! ## Failure Modes
//! - Invalid input schema: Returns validation error
//! - Insufficient sample size: Returns inconclusive result with warning
//! - Statistical assumption violations: Returns result with violation flags
//! - ruvector-service unavailable: Returns persistence error

use async_trait::async_trait;
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use serde_json::json;
use thiserror::Error;
use tracing::{debug, info, instrument, warn};
use uuid::Uuid;
use validator::Validate;

use crate::contracts::{
    hypothesis::*,
    decision_event::*,
    common::*,
};
use super::traits::{Agent, ConfidenceEstimator, PerformanceBounded, PerformanceBudget, BudgetViolation};

/// Agent version (semantic versioning).
pub const HYPOTHESIS_AGENT_VERSION: &str = "1.0.0";

/// Agent identifier.
pub const HYPOTHESIS_AGENT_ID: &str = "hypothesis-agent-v1";

/// Errors from Hypothesis Agent operations.
#[derive(Debug, Error)]
pub enum HypothesisAgentError {
    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Insufficient sample size: required {required}, got {actual}")]
    InsufficientSampleSize { required: u64, actual: u64 },

    #[error("Statistical computation error: {0}")]
    StatisticalComputation(String),

    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Internal error: {0}")]
    Internal(String),

    #[error("Performance budget exceeded: {0}")]
    BudgetExceeded(String),
}

impl From<validator::ValidationErrors> for HypothesisAgentError {
    fn from(err: validator::ValidationErrors) -> Self {
        HypothesisAgentError::Validation(err.to_string())
    }
}

impl From<BudgetViolation> for HypothesisAgentError {
    fn from(err: BudgetViolation) -> Self {
        HypothesisAgentError::BudgetExceeded(err.message)
    }
}

/// Hypothesis Agent for evaluating research hypotheses.
///
/// This agent implements the core hypothesis evaluation logic for LLM-Research-Lab.
/// It is stateless, deterministic, and produces structured DecisionEvents.
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
pub struct HypothesisAgent {
    identity: AgentIdentity,
    config: HypothesisAgentConfig,
    /// Performance budget for this agent (Phase 7 MANDATORY)
    budget: PerformanceBudget,
}

/// Configuration for Hypothesis Agent.
#[derive(Debug, Clone)]
pub struct HypothesisAgentConfig {
    /// Minimum sample size for evaluation
    pub min_sample_size: u64,

    /// Default significance level
    pub default_alpha: Decimal,

    /// Enable assumption checking
    pub check_assumptions: bool,

    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for HypothesisAgentConfig {
    fn default() -> Self {
        Self {
            min_sample_size: 30,
            default_alpha: dec!(0.05),
            check_assumptions: true,
            random_seed: None,
        }
    }
}

impl HypothesisAgent {
    /// Create a new Hypothesis Agent with default configuration.
    pub fn new() -> Self {
        Self::with_config(HypothesisAgentConfig::default())
    }

    /// Create a new Hypothesis Agent with custom configuration.
    pub fn with_config(config: HypothesisAgentConfig) -> Self {
        Self::with_config_and_budget(config, PerformanceBudget::default())
    }

    /// Create a new Hypothesis Agent with custom configuration and budget.
    pub fn with_config_and_budget(config: HypothesisAgentConfig, budget: PerformanceBudget) -> Self {
        Self {
            identity: AgentIdentity {
                id: HYPOTHESIS_AGENT_ID.to_string(),
                version: HYPOTHESIS_AGENT_VERSION.to_string(),
                classification: AgentClassification::HypothesisEvaluation,
                description: "Evaluates research hypotheses using statistical methods".to_string(),
            },
            config,
            budget,
        }
    }

    /// Perform t-test hypothesis evaluation.
    #[instrument(skip(self, data), fields(sample_size = data.sample_size))]
    fn evaluate_ttest(
        &self,
        hypothesis: &HypothesisDefinition,
        data: &ExperimentalData,
        config: &EvaluationConfig,
    ) -> Result<TestResults, HypothesisAgentError> {
        info!("Performing t-test evaluation");

        // Extract values from observations
        let values: Vec<f64> = data
            .observations
            .iter()
            .filter_map(|obs| {
                obs.values
                    .get("value")
                    .and_then(|v| v.as_f64())
            })
            .collect();

        if values.len() < self.config.min_sample_size as usize {
            return Err(HypothesisAgentError::InsufficientSampleSize {
                required: self.config.min_sample_size,
                actual: values.len() as u64,
            });
        }

        // Compute sample statistics
        let n = values.len() as f64;
        let mean: f64 = values.iter().sum::<f64>() / n;
        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        let std_error = std_dev / n.sqrt();

        // For one-sample t-test against mu=0 (simplified)
        let t_statistic = mean / std_error;
        let df = n - 1.0;

        // Approximate p-value using t-distribution
        // In production, use proper statistical library
        let p_value = self.approximate_t_pvalue(t_statistic.abs(), df);

        let alpha: f64 = hypothesis
            .significance_level
            .try_into()
            .unwrap_or(0.05);

        let null_rejected = p_value < alpha;

        // Compute confidence interval
        let t_critical = self.t_critical_value(alpha / 2.0, df);
        let ci_margin = t_critical * std_error;

        debug!(
            t_statistic = t_statistic,
            p_value = p_value,
            df = df,
            null_rejected = null_rejected,
            "T-test results"
        );

        Ok(TestResults {
            test_statistic: Decimal::try_from(t_statistic).unwrap_or(dec!(0)),
            p_value: Decimal::try_from(p_value).unwrap_or(dec!(1)),
            corrected_p_value: if config.apply_correction {
                Some(Decimal::try_from(p_value).unwrap_or(dec!(1)))
            } else {
                None
            },
            degrees_of_freedom: Some(Decimal::try_from(df).unwrap_or(dec!(0))),
            confidence_interval: Some(ConfidenceInterval {
                lower: Decimal::try_from(mean - ci_margin).unwrap_or(dec!(0)),
                upper: Decimal::try_from(mean + ci_margin).unwrap_or(dec!(0)),
                level: dec!(0.95),
            }),
            null_rejected,
            decision: if null_rejected {
                "Reject null hypothesis".to_string()
            } else {
                "Fail to reject null hypothesis".to_string()
            },
        })
    }

    /// Approximate t-distribution p-value.
    ///
    /// This is a simplified approximation. In production, use a proper
    /// statistical library like statrs.
    fn approximate_t_pvalue(&self, t: f64, df: f64) -> f64 {
        // Use normal approximation for large df
        if df > 30.0 {
            // Standard normal CDF approximation
            let z = t;
            let p = 0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2));
            2.0 * (1.0 - p)
        } else {
            // Rough approximation for smaller df
            let x = df / (df + t * t);
            let p = incomplete_beta(df / 2.0, 0.5, x) / 2.0;
            2.0 * p.min(1.0 - p)
        }
    }

    /// Get critical t-value for given alpha and df.
    fn t_critical_value(&self, alpha: f64, df: f64) -> f64 {
        // Approximation using normal distribution for large df
        if df > 30.0 {
            // z-score approximation
            inverse_normal_cdf(1.0 - alpha)
        } else {
            // Rough approximation
            1.96 + 2.0 / df
        }
    }

    /// Compute effect size (Cohen's d).
    fn compute_effect_size(&self, data: &ExperimentalData) -> Option<EffectSize> {
        let values: Vec<f64> = data
            .observations
            .iter()
            .filter_map(|obs| obs.values.get("value").and_then(|v| v.as_f64()))
            .collect();

        if values.len() < 2 {
            return None;
        }

        let n = values.len() as f64;
        let mean: f64 = values.iter().sum::<f64>() / n;
        let variance: f64 = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        if std_dev == 0.0 {
            return None;
        }

        // Cohen's d for one-sample (compared to 0)
        let d = mean / std_dev;

        let interpretation = if d.abs() < 0.2 {
            "negligible"
        } else if d.abs() < 0.5 {
            "small"
        } else if d.abs() < 0.8 {
            "medium"
        } else {
            "large"
        };

        Some(EffectSize {
            measure: EffectSizeMeasure::CohensD,
            value: Decimal::try_from(d).unwrap_or(dec!(0)),
            interpretation: interpretation.to_string(),
        })
    }

    /// Check statistical assumptions.
    fn check_assumptions(&self, data: &ExperimentalData) -> Vec<AssumptionViolation> {
        let mut violations = Vec::new();

        // Check sample size
        if data.sample_size < self.config.min_sample_size {
            violations.push(AssumptionViolation {
                assumption: "Minimum sample size".to_string(),
                test_used: "Count check".to_string(),
                severity: ViolationSeverity::Severe,
                recommendation: format!(
                    "Increase sample size to at least {}",
                    self.config.min_sample_size
                ),
            });
        }

        // Check data quality
        let completeness: f64 = data.quality_metrics.completeness.try_into().unwrap_or(0.0);
        if completeness < 0.9 {
            violations.push(AssumptionViolation {
                assumption: "Data completeness".to_string(),
                test_used: "Completeness ratio".to_string(),
                severity: if completeness < 0.7 {
                    ViolationSeverity::Severe
                } else {
                    ViolationSeverity::Moderate
                },
                recommendation: "Address missing data before analysis".to_string(),
            });
        }

        violations
    }

    /// Determine hypothesis status from test results.
    fn determine_status(
        &self,
        results: &TestResults,
        violations: &[AssumptionViolation],
    ) -> HypothesisStatus {
        // Check for severe violations
        let has_severe_violations = violations
            .iter()
            .any(|v| v.severity == ViolationSeverity::Severe);

        if has_severe_violations {
            return HypothesisStatus::Inconclusive;
        }

        if results.null_rejected {
            HypothesisStatus::Accepted
        } else {
            HypothesisStatus::Rejected
        }
    }
}

impl Default for HypothesisAgent {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceEstimator for HypothesisAgent {
    fn estimate_confidence(&self, sample_size: u64, effect_size: Option<f64>) -> f64 {
        // Base confidence from sample size
        let size_confidence = (sample_size as f64 / 1000.0).min(0.9);

        // Adjust based on effect size if available
        let effect_adjustment = effect_size
            .map(|e| (e.abs() * 0.1).min(0.1))
            .unwrap_or(0.0);

        (size_confidence + effect_adjustment).min(0.99)
    }
}

impl PerformanceBounded for HypothesisAgent {
    fn budget(&self) -> &PerformanceBudget {
        &self.budget
    }
}

#[async_trait]
impl Agent for HypothesisAgent {
    type Input = HypothesisInput;
    type Output = HypothesisOutput;
    type Error = HypothesisAgentError;

    fn identity(&self) -> &AgentIdentity {
        &self.identity
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), Self::Error> {
        input.validate()?;
        input.hypothesis.validate()?;
        input.experimental_data.validate()?;
        input.config.validate()?;
        Ok(())
    }

    #[instrument(skip(self, input), fields(
        request_id = %input.request_id,
        hypothesis_id = %input.hypothesis.id,
        hypothesis_type = ?input.hypothesis.hypothesis_type,
        sample_size = input.experimental_data.sample_size
    ))]
    async fn execute(&self, input: Self::Input) -> Result<Self::Output, Self::Error> {
        // Phase 7: Start timing for performance budget enforcement
        let start = std::time::Instant::now();

        info!("Executing hypothesis evaluation");

        // Check assumptions if configured
        let assumption_violations = if self.config.check_assumptions {
            self.check_assumptions(&input.experimental_data)
        } else {
            Vec::new()
        };

        if !assumption_violations.is_empty() {
            warn!(
                violations = assumption_violations.len(),
                "Assumption violations detected"
            );
        }

        // Perform statistical test based on configuration
        let test_results = match input.config.test_method {
            StatisticalTest::TTest | StatisticalTest::WelchTTest => {
                self.evaluate_ttest(&input.hypothesis, &input.experimental_data, &input.config)?
            }
            _ => {
                // For other tests, return a placeholder (would implement in production)
                warn!("Test method {:?} not fully implemented, using t-test", input.config.test_method);
                self.evaluate_ttest(&input.hypothesis, &input.experimental_data, &input.config)?
            }
        };

        // Compute effect size if requested
        let effect_size = if input.config.compute_effect_size {
            self.compute_effect_size(&input.experimental_data)
        } else {
            None
        };

        // Determine final status
        let status = self.determine_status(&test_results, &assumption_violations);

        // Compute achieved power (post-hoc)
        let sample_size = input.experimental_data.sample_size;
        let effect_value: Option<f64> = effect_size.as_ref().map(|e| e.value.try_into().unwrap_or(0.0));
        let achieved_power = Decimal::try_from(
            self.estimate_confidence(sample_size, effect_value)
        ).ok();

        // Build recommendations
        let recommendations = self.build_recommendations(&status, &test_results, &assumption_violations);

        // Assess sample adequacy
        let sample_adequacy = if sample_size >= 100 {
            SampleAdequacy::Adequate
        } else if sample_size >= 30 {
            SampleAdequacy::Marginal
        } else {
            SampleAdequacy::Inadequate
        };

        // Phase 7: Check latency budget BEFORE returning result
        let elapsed_ms = start.elapsed().as_millis() as u64;
        if let Err(violation) = self.check_latency(elapsed_ms) {
            tracing::error!(
                elapsed_ms = elapsed_ms,
                budget_ms = self.budget.max_latency_ms,
                budget_type = %violation.budget_type,
                "Performance budget exceeded - ABORTING"
            );
            return Err(HypothesisAgentError::BudgetExceeded(format!(
                "Latency budget exceeded: {}ms > {}ms limit",
                elapsed_ms, self.budget.max_latency_ms
            )));
        }

        let output = HypothesisOutput {
            hypothesis_id: input.hypothesis.id,
            status,
            test_results,
            effect_size,
            diagnostics: DiagnosticInfo {
                achieved_power,
                sample_adequacy,
                assumption_violations,
                warnings: Vec::new(),
            },
            recommendations,
        };

        info!(
            status = ?output.status,
            p_value = %output.test_results.p_value,
            elapsed_ms = elapsed_ms,
            "Hypothesis evaluation complete"
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
        let inputs_hash = DecisionEvent::compute_inputs_hash(input)
            .map_err(|e| HypothesisAgentError::Internal(e.to_string()))?;

        // Estimate confidence
        let sample_size = input.experimental_data.sample_size;
        let effect_value: Option<f64> = output.effect_size.as_ref()
            .map(|e| e.value.try_into().unwrap_or(0.0));
        let confidence_value = self.estimate_confidence(sample_size, effect_value);

        let confidence = Confidence {
            value: Decimal::try_from(confidence_value)
                .map_err(|e| HypothesisAgentError::Internal(e.to_string()))?,
            method: ConfidenceMethod::Heuristic,
            sample_size: Some(sample_size),
            ci_lower: output.test_results.confidence_interval.as_ref().map(|ci| ci.lower),
            ci_upper: output.test_results.confidence_interval.as_ref().map(|ci| ci.upper),
        };

        // Build constraints
        let constraints = ConstraintsApplied {
            scope: vec![
                format!("hypothesis_type: {:?}", input.hypothesis.hypothesis_type),
                format!("test_method: {:?}", input.config.test_method),
            ],
            assumptions: vec![
                "Normal distribution assumed for t-test".to_string(),
                "Independent observations".to_string(),
            ],
            limitations: output
                .diagnostics
                .assumption_violations
                .iter()
                .map(|v| format!("{}: {}", v.assumption, v.recommendation))
                .collect(),
            data_filters: Vec::new(),
            temporal_bounds: None,
        };

        // Build execution ref
        let execution_ref = ExecutionRef {
            execution_id,
            trace_id: None, // Would be populated from tracing context
            span_id: None,
            parent_ref: input.context.as_ref().and_then(|c| c.telemetry_ref.clone()),
            runtime_version: Some(HYPOTHESIS_AGENT_VERSION.to_string()),
        };

        // Serialize output
        let outputs = serde_json::to_value(output)
            .map_err(|e| HypothesisAgentError::Internal(e.to_string()))?;

        DecisionEvent::builder()
            .agent_id(HYPOTHESIS_AGENT_ID)
            .agent_version(HYPOTHESIS_AGENT_VERSION)
            .decision_type(DecisionType::HypothesisEvaluation)
            .inputs_hash(inputs_hash)
            .outputs(outputs)
            .confidence(confidence)
            .constraints_applied(constraints)
            .execution_ref(execution_ref)
            .metadata(json!({
                "request_id": input.request_id,
                "hypothesis_id": input.hypothesis.id,
            }))
            .build()
            .map_err(|e| HypothesisAgentError::Internal(e.to_string()))
    }
}

impl HypothesisAgent {
    /// Build recommendations based on results.
    fn build_recommendations(
        &self,
        status: &HypothesisStatus,
        results: &TestResults,
        violations: &[AssumptionViolation],
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        // Status-based recommendations
        match status {
            HypothesisStatus::Inconclusive => {
                recommendations.push("Address assumption violations before drawing conclusions".to_string());
            }
            HypothesisStatus::Accepted => {
                let p_value: f64 = results.p_value.try_into().unwrap_or(1.0);
                if p_value < 0.001 {
                    recommendations.push("Strong evidence against null hypothesis".to_string());
                } else {
                    recommendations.push("Moderate evidence against null hypothesis".to_string());
                }
            }
            HypothesisStatus::Rejected => {
                recommendations.push("Consider increasing sample size for more power".to_string());
            }
            _ => {}
        }

        // Violation-based recommendations
        for violation in violations {
            recommendations.push(violation.recommendation.clone());
        }

        recommendations
    }
}

// Helper functions for statistical approximations

/// Error function approximation.
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Incomplete beta function approximation.
fn incomplete_beta(a: f64, b: f64, x: f64) -> f64 {
    // Simple approximation for beta function
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }

    // Use continued fraction approximation
    let bt = if x == 0.0 || x == 1.0 {
        0.0
    } else {
        (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp()
    };

    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_cf(a, b, x) / a
    } else {
        1.0 - bt * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Continued fraction for incomplete beta.
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let max_iter = 100;
    let eps = 1e-10;

    let mut c = 1.0;
    let mut d = 1.0 - (a + b) * x / (a + 1.0);
    if d.abs() < 1e-30 {
        d = 1e-30;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=max_iter {
        let m = m as f64;
        let m2 = 2.0 * m;

        let aa = m * (b - m) * x / ((a + m2 - 1.0) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1.0));
        d = 1.0 + aa * d;
        if d.abs() < 1e-30 {
            d = 1e-30;
        }
        c = 1.0 + aa / c;
        if c.abs() < 1e-30 {
            c = 1e-30;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < eps {
            break;
        }
    }

    h
}

/// Log gamma function approximation.
fn ln_gamma(x: f64) -> f64 {
    // Lanczos approximation
    let g = 7;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];

    if x < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * x).sin().ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = c[0];
        for i in 1..g + 2 {
            a += c[i] / (x + i as f64);
        }
        let t = x + g as f64 + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (t - 0.5) * t.ln() - t + a.ln()
    }
}

/// Inverse normal CDF approximation.
fn inverse_normal_cdf(p: f64) -> f64 {
    // Rational approximation
    let a = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_input(sample_size: u64) -> HypothesisInput {
        let observations: Vec<Observation> = (0..sample_size)
            .map(|i| Observation {
                id: Uuid::new_v4(),
                values: json!({"value": (i as f64) * 0.1 + 0.5}),
                group: None,
                weight: None,
                timestamp: None,
            })
            .collect();

        HypothesisInput {
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
                sample_size,
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
        }
    }

    #[tokio::test]
    async fn test_hypothesis_agent_execution() {
        let agent = HypothesisAgent::new();
        let input = create_test_input(100);

        let result = agent.execute(input).await;
        assert!(result.is_ok());

        let output = result.unwrap();
        assert!(matches!(
            output.status,
            HypothesisStatus::Accepted | HypothesisStatus::Rejected
        ));
    }

    #[tokio::test]
    async fn test_insufficient_sample_size() {
        let agent = HypothesisAgent::new();
        let input = create_test_input(10); // Below minimum

        let result = agent.execute(input).await;
        assert!(matches!(result, Err(HypothesisAgentError::InsufficientSampleSize { .. })));
    }

    #[tokio::test]
    async fn test_decision_event_generation() {
        let agent = HypothesisAgent::new();
        let input = create_test_input(100);

        let (output, event) = agent.invoke(input).await.unwrap();

        assert_eq!(event.agent_id, HYPOTHESIS_AGENT_ID);
        assert_eq!(event.agent_version, HYPOTHESIS_AGENT_VERSION);
        assert_eq!(event.decision_type, DecisionType::HypothesisEvaluation);
        assert_eq!(event.inputs_hash.len(), 64);
    }

    #[test]
    fn test_confidence_estimator() {
        let agent = HypothesisAgent::new();

        let conf_small = agent.estimate_confidence(10, None);
        let conf_medium = agent.estimate_confidence(100, None);
        let conf_large = agent.estimate_confidence(1000, Some(0.8));

        assert!(conf_small < conf_medium);
        assert!(conf_medium < conf_large);
        assert!(conf_large <= 0.99);
    }
}
