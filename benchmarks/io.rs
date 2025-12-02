//! Benchmark I/O operations for reading and writing benchmark results
//!
//! This module provides utilities for persisting benchmark results to disk
//! and loading them for analysis.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde_json;

use crate::result::BenchmarkResult;

/// Default output directory for benchmark results
pub const DEFAULT_OUTPUT_DIR: &str = "benchmarks/output";

/// Default raw output directory for individual benchmark results
pub const DEFAULT_RAW_DIR: &str = "benchmarks/output/raw";

/// I/O handler for benchmark results
pub struct BenchmarkIO {
    output_dir: PathBuf,
    raw_dir: PathBuf,
}

impl Default for BenchmarkIO {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchmarkIO {
    /// Create a new BenchmarkIO with default paths
    pub fn new() -> Self {
        Self {
            output_dir: PathBuf::from(DEFAULT_OUTPUT_DIR),
            raw_dir: PathBuf::from(DEFAULT_RAW_DIR),
        }
    }

    /// Create a BenchmarkIO with custom paths
    pub fn with_paths(output_dir: impl Into<PathBuf>, raw_dir: impl Into<PathBuf>) -> Self {
        Self {
            output_dir: output_dir.into(),
            raw_dir: raw_dir.into(),
        }
    }

    /// Ensure output directories exist
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        fs::create_dir_all(&self.output_dir)?;
        fs::create_dir_all(&self.raw_dir)?;
        Ok(())
    }

    /// Write a single benchmark result to the raw directory
    pub fn write_result(&self, result: &BenchmarkResult) -> std::io::Result<PathBuf> {
        self.ensure_dirs()?;

        let filename = format!(
            "{}_{}.json",
            result.target_id.replace(['/', '\\', ':'], "_"),
            result.timestamp.format("%Y%m%d_%H%M%S")
        );
        let path = self.raw_dir.join(&filename);

        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, result)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writer.flush()?;

        Ok(path)
    }

    /// Write multiple benchmark results to a combined output file
    pub fn write_results(&self, results: &[BenchmarkResult]) -> std::io::Result<PathBuf> {
        self.ensure_dirs()?;

        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("benchmark_results_{}.json", timestamp);
        let path = self.output_dir.join(&filename);

        let file = File::create(&path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, results)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        writer.flush()?;

        // Also write individual results to raw directory
        for result in results {
            self.write_result(result)?;
        }

        Ok(path)
    }

    /// Read a benchmark result from a file
    pub fn read_result(path: impl AsRef<Path>) -> std::io::Result<BenchmarkResult> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        serde_json::from_reader(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Read all benchmark results from the raw directory
    pub fn read_all_results(&self) -> std::io::Result<Vec<BenchmarkResult>> {
        let mut results = Vec::new();

        if !self.raw_dir.exists() {
            return Ok(results);
        }

        for entry in fs::read_dir(&self.raw_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                match Self::read_result(&path) {
                    Ok(result) => results.push(result),
                    Err(e) => {
                        eprintln!("Warning: Failed to read {:?}: {}", path, e);
                    }
                }
            }
        }

        // Sort by timestamp
        results.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        Ok(results)
    }

    /// Read results from a specific time range
    pub fn read_results_in_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> std::io::Result<Vec<BenchmarkResult>> {
        let all_results = self.read_all_results()?;
        Ok(all_results
            .into_iter()
            .filter(|r| r.timestamp >= start && r.timestamp <= end)
            .collect())
    }

    /// Get the output directory path
    pub fn output_dir(&self) -> &Path {
        &self.output_dir
    }

    /// Get the raw directory path
    pub fn raw_dir(&self) -> &Path {
        &self.raw_dir
    }

    /// Clean old results (older than specified days)
    pub fn clean_old_results(&self, days: i64) -> std::io::Result<usize> {
        let cutoff = Utc::now() - chrono::Duration::days(days);
        let mut removed = 0;

        if !self.raw_dir.exists() {
            return Ok(0);
        }

        for entry in fs::read_dir(&self.raw_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                if let Ok(result) = Self::read_result(&path) {
                    if result.timestamp < cutoff {
                        fs::remove_file(&path)?;
                        removed += 1;
                    }
                }
            }
        }

        Ok(removed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[test]
    fn test_write_and_read_result() {
        let temp_dir = TempDir::new().unwrap();
        let io = BenchmarkIO::with_paths(
            temp_dir.path().join("output"),
            temp_dir.path().join("raw"),
        );

        let result = BenchmarkResult::new("test-io", json!({"value": 123}));
        let path = io.write_result(&result).unwrap();

        assert!(path.exists());

        let loaded = BenchmarkIO::read_result(&path).unwrap();
        assert_eq!(loaded.target_id, "test-io");
    }

    #[test]
    fn test_write_multiple_results() {
        let temp_dir = TempDir::new().unwrap();
        let io = BenchmarkIO::with_paths(
            temp_dir.path().join("output"),
            temp_dir.path().join("raw"),
        );

        let results = vec![
            BenchmarkResult::new("target-1", json!({"a": 1})),
            BenchmarkResult::new("target-2", json!({"b": 2})),
        ];

        let path = io.write_results(&results).unwrap();
        assert!(path.exists());

        let loaded = io.read_all_results().unwrap();
        assert_eq!(loaded.len(), 2);
    }
}
