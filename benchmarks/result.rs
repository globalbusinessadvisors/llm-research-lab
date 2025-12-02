//! Canonical BenchmarkResult struct for LLM Research Lab
//!
//! This module defines the standardized result type used across all benchmark targets.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Canonical benchmark result containing standardized fields
/// for cross-repository benchmark compatibility.
///
/// # Fields
/// - `target_id`: Unique identifier for the benchmark target
/// - `metrics`: JSON value containing all metric measurements
/// - `timestamp`: UTC timestamp when the benchmark was executed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Unique identifier for the benchmark target that produced this result
    pub target_id: String,

    /// Metrics captured during benchmark execution.
    /// Structure varies by target but typically includes:
    /// - duration_ms: Execution time in milliseconds
    /// - memory_bytes: Memory usage (if applicable)
    /// - iterations: Number of iterations performed
    /// - throughput: Operations per second (if applicable)
    /// - custom metrics specific to the target
    pub metrics: serde_json::Value,

    /// UTC timestamp when the benchmark was executed
    pub timestamp: DateTime<Utc>,
}

impl BenchmarkResult {
    /// Create a new BenchmarkResult with the current timestamp
    pub fn new(target_id: impl Into<String>, metrics: serde_json::Value) -> Self {
        Self {
            target_id: target_id.into(),
            metrics,
            timestamp: Utc::now(),
        }
    }

    /// Create a BenchmarkResult with a specific timestamp
    pub fn with_timestamp(
        target_id: impl Into<String>,
        metrics: serde_json::Value,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            target_id: target_id.into(),
            metrics,
            timestamp,
        }
    }

    /// Get the duration in milliseconds if present in metrics
    pub fn duration_ms(&self) -> Option<f64> {
        self.metrics.get("duration_ms").and_then(|v| v.as_f64())
    }

    /// Get the iteration count if present in metrics
    pub fn iterations(&self) -> Option<u64> {
        self.metrics.get("iterations").and_then(|v| v.as_u64())
    }

    /// Check if the benchmark succeeded
    pub fn is_success(&self) -> bool {
        self.metrics
            .get("success")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Get error message if benchmark failed
    pub fn error(&self) -> Option<&str> {
        self.metrics.get("error").and_then(|v| v.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_benchmark_result_new() {
        let result = BenchmarkResult::new(
            "test-target",
            json!({
                "duration_ms": 100.5,
                "iterations": 1000,
                "success": true
            }),
        );

        assert_eq!(result.target_id, "test-target");
        assert_eq!(result.duration_ms(), Some(100.5));
        assert_eq!(result.iterations(), Some(1000));
        assert!(result.is_success());
    }

    #[test]
    fn test_benchmark_result_serialization() {
        let result = BenchmarkResult::new(
            "serialize-test",
            json!({"value": 42}),
        );

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: BenchmarkResult = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.target_id, "serialize-test");
    }
}
