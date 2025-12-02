//! Canonical Benchmark Module for LLM Research Lab
//!
//! This module provides the standardized benchmark interface used across
//! all 25 benchmark-target repositories for consistent performance measurement
//! and result reporting.
//!
//! # Architecture
//!
//! The benchmark system consists of:
//! - `BenchmarkResult`: Standardized result struct with target_id, metrics, and timestamp
//! - `BenchmarkIO`: File I/O operations for persisting results
//! - `MarkdownGenerator`: Human-readable report generation
//! - `run_all_benchmarks()`: Main entrypoint that executes all registered targets
//!
//! # Usage
//!
//! ```rust,ignore
//! use benchmarks::run_all_benchmarks;
//!
//! let results = run_all_benchmarks();
//! for result in &results {
//!     println!("{}: {:?}", result.target_id, result.metrics);
//! }
//! ```

pub mod io;
pub mod markdown;
pub mod result;

pub use io::BenchmarkIO;
pub use markdown::{create_summary, MarkdownGenerator};
pub use result::BenchmarkResult;

use chrono::Utc;
use serde_json::json;
use std::time::Instant;

// Import adapters for benchmark targets
#[path = "../adapters/mod.rs"]
pub mod adapters;

/// Run all registered benchmark targets and return their results.
///
/// This is the canonical entrypoint for the benchmark system. It:
/// 1. Retrieves all registered benchmark targets from the adapters module
/// 2. Executes each target's `run()` method
/// 3. Collects results into a Vec<BenchmarkResult>
/// 4. Writes results to the canonical output directories
/// 5. Generates a summary.md report
///
/// # Returns
///
/// A vector of `BenchmarkResult` structs containing the results from all targets.
///
/// # Example
///
/// ```rust,ignore
/// let results = run_all_benchmarks();
/// println!("Completed {} benchmarks", results.len());
/// ```
pub fn run_all_benchmarks() -> Vec<BenchmarkResult> {
    let targets = adapters::all_targets();
    let mut results = Vec::with_capacity(targets.len());

    println!("Running {} benchmark targets...", targets.len());

    for target in targets {
        let target_id = target.id();
        println!("  Running: {}", target_id);

        let start = Instant::now();
        let run_result = target.run();
        let duration = start.elapsed();

        let result = match run_result {
            Ok(metrics) => {
                // Merge duration into metrics
                let mut merged_metrics = metrics;
                if let Some(obj) = merged_metrics.as_object_mut() {
                    obj.insert("duration_ms".to_string(), json!(duration.as_secs_f64() * 1000.0));
                    obj.insert("success".to_string(), json!(true));
                }
                BenchmarkResult::new(target_id, merged_metrics)
            }
            Err(e) => BenchmarkResult::new(
                target_id,
                json!({
                    "duration_ms": duration.as_secs_f64() * 1000.0,
                    "success": false,
                    "error": e.to_string()
                }),
            ),
        };

        results.push(result);
    }

    // Write results to disk
    let io = BenchmarkIO::new();
    if let Err(e) = io.write_results(&results) {
        eprintln!("Warning: Failed to write benchmark results: {}", e);
    }

    // Generate summary
    if let Err(e) = create_summary(&results, io.output_dir()) {
        eprintln!("Warning: Failed to create summary: {}", e);
    }

    println!("Completed {} benchmarks.", results.len());
    results
}

/// Run benchmarks for specific targets only
pub fn run_benchmarks(target_ids: &[&str]) -> Vec<BenchmarkResult> {
    let all_targets = adapters::all_targets();
    let filtered: Vec<_> = all_targets
        .into_iter()
        .filter(|t| target_ids.contains(&t.id().as_str()))
        .collect();

    let mut results = Vec::with_capacity(filtered.len());

    for target in filtered {
        let target_id = target.id();
        let start = Instant::now();
        let run_result = target.run();
        let duration = start.elapsed();

        let result = match run_result {
            Ok(metrics) => {
                let mut merged_metrics = metrics;
                if let Some(obj) = merged_metrics.as_object_mut() {
                    obj.insert("duration_ms".to_string(), json!(duration.as_secs_f64() * 1000.0));
                    obj.insert("success".to_string(), json!(true));
                }
                BenchmarkResult::new(target_id, merged_metrics)
            }
            Err(e) => BenchmarkResult::new(
                target_id,
                json!({
                    "duration_ms": duration.as_secs_f64() * 1000.0,
                    "success": false,
                    "error": e.to_string()
                }),
            ),
        };

        results.push(result);
    }

    results
}

/// Get list of all available benchmark target IDs
pub fn list_targets() -> Vec<String> {
    adapters::all_targets().iter().map(|t| t.id()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_targets() {
        let targets = list_targets();
        assert!(!targets.is_empty(), "Should have at least one benchmark target");
    }

    #[test]
    fn test_benchmark_result_creation() {
        let result = BenchmarkResult::new("test", json!({"value": 42}));
        assert_eq!(result.target_id, "test");
        assert!(result.timestamp <= Utc::now());
    }
}
