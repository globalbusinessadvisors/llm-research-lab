//! Adapters Module for LLM Research Lab Benchmark Targets
//!
//! This module implements the canonical BenchTarget trait and provides
//! adapters that expose Research Lab operations as benchmark targets.
//!
//! # Architecture
//!
//! The adapters module follows the adapter pattern to wrap existing
//! Research Lab functionality (metrics, evaluators, workflows) and
//! expose them through a standardized benchmark interface.
//!
//! # Adding New Targets
//!
//! To add a new benchmark target:
//! 1. Create a struct implementing `BenchTarget`
//! 2. Implement the `id()` and `run()` methods
//! 3. Add the target to `all_targets()`

pub mod metrics;
pub mod evaluators;
pub mod workflows;
pub mod scoring;

pub use metrics::*;
pub use evaluators::*;
pub use workflows::*;
pub use scoring::*;

use serde_json::Value;
use std::error::Error;

/// Canonical BenchTarget trait for benchmark adapters.
///
/// All benchmark targets must implement this trait to be included
/// in the benchmark system. The trait provides:
/// - `id()`: Unique identifier for the target
/// - `run()`: Execute the benchmark and return metrics
///
/// # Example
///
/// ```rust,ignore
/// struct MyBenchmark;
///
/// impl BenchTarget for MyBenchmark {
///     fn id(&self) -> String {
///         "my-benchmark".to_string()
///     }
///
///     fn run(&self) -> Result<Value, Box<dyn Error>> {
///         // Perform benchmark operations
///         Ok(json!({"operations": 1000, "throughput": 500.0}))
///     }
/// }
/// ```
pub trait BenchTarget: Send + Sync {
    /// Returns the unique identifier for this benchmark target.
    ///
    /// The ID should be:
    /// - Lowercase with hyphens (kebab-case)
    /// - Descriptive of what is being benchmarked
    /// - Unique across all targets
    fn id(&self) -> String;

    /// Execute the benchmark and return metrics as JSON.
    ///
    /// The returned JSON should contain relevant metrics such as:
    /// - `iterations`: Number of iterations performed
    /// - `throughput`: Operations per second
    /// - `memory_bytes`: Memory usage (if applicable)
    /// - Custom metrics specific to the target
    ///
    /// Note: `duration_ms` and `success` are automatically added
    /// by the benchmark runner.
    fn run(&self) -> Result<Value, Box<dyn Error>>;

    /// Optional: Returns a description of what this target benchmarks
    fn description(&self) -> Option<String> {
        None
    }

    /// Optional: Returns the category of this benchmark
    fn category(&self) -> Option<String> {
        None
    }
}

/// Registry of all benchmark targets.
///
/// This function returns a vector of all available benchmark targets
/// that will be executed when `run_all_benchmarks()` is called.
///
/// # Returns
///
/// A vector of boxed trait objects implementing `BenchTarget`.
pub fn all_targets() -> Vec<Box<dyn BenchTarget>> {
    vec![
        // Metric computation benchmarks
        Box::new(metrics::AccuracyMetricBenchmark::new()),
        Box::new(metrics::BleuMetricBenchmark::new()),
        Box::new(metrics::RougeMetricBenchmark::new()),
        Box::new(metrics::LatencyMetricBenchmark::new()),
        Box::new(metrics::PerplexityMetricBenchmark::new()),
        Box::new(metrics::AggregatorBenchmark::new()),
        Box::new(metrics::StatisticalAnalysisBenchmark::new()),

        // Evaluator alpha-testing benchmarks
        Box::new(evaluators::BatchEvaluationBenchmark::new()),
        Box::new(evaluators::ComparativeEvaluationBenchmark::new()),
        Box::new(evaluators::LLMJudgeBenchmark::new()),

        // Workflow/pipeline benchmarks
        Box::new(workflows::PipelineOrchestrationBenchmark::new()),
        Box::new(workflows::TaskExecutionBenchmark::new()),
        Box::new(workflows::DataLoadingBenchmark::new()),

        // Scoring/ranking benchmarks
        Box::new(scoring::HeuristicScoringBenchmark::new()),
        Box::new(scoring::ModelRankingBenchmark::new()),
        Box::new(scoring::ChainOfThoughtBenchmark::new()),
    ]
}

/// Get a specific benchmark target by ID
pub fn get_target(id: &str) -> Option<Box<dyn BenchTarget>> {
    all_targets().into_iter().find(|t| t.id() == id)
}

/// Get targets by category
pub fn get_targets_by_category(category: &str) -> Vec<Box<dyn BenchTarget>> {
    all_targets()
        .into_iter()
        .filter(|t| t.category().as_deref() == Some(category))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_targets_not_empty() {
        let targets = all_targets();
        assert!(!targets.is_empty(), "Should have benchmark targets registered");
    }

    #[test]
    fn test_targets_have_unique_ids() {
        let targets = all_targets();
        let mut ids: Vec<_> = targets.iter().map(|t| t.id()).collect();
        let original_len = ids.len();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), original_len, "All target IDs should be unique");
    }

    #[test]
    fn test_get_target() {
        let target = get_target("accuracy-metric");
        assert!(target.is_some(), "Should find accuracy-metric target");
    }
}
