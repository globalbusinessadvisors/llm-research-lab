//! Workflow and Pipeline Benchmark Adapters
//!
//! These adapters expose the LLM Research Lab workflow operations
//! as benchmark targets for testing reproducible experiment pipelines.

use serde_json::{json, Value};
use std::collections::HashMap;
use std::error::Error;
use std::time::Instant;

use super::BenchTarget;

/// Benchmark adapter for pipeline orchestration.
///
/// Tests the performance of DAG-based pipeline execution with
/// task dependencies.
pub struct PipelineOrchestrationBenchmark {
    stages: usize,
    tasks_per_stage: usize,
    iterations: usize,
}

impl PipelineOrchestrationBenchmark {
    pub fn new() -> Self {
        Self {
            stages: 5,
            tasks_per_stage: 4,
            iterations: 20,
        }
    }
}

impl Default for PipelineOrchestrationBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchTarget for PipelineOrchestrationBenchmark {
    fn id(&self) -> String {
        "pipeline-orchestration".to_string()
    }

    fn run(&self) -> Result<Value, Box<dyn Error>> {
        let start = Instant::now();

        let mut total_tasks_executed = 0;
        let mut stage_timings: Vec<f64> = Vec::new();

        for _ in 0..self.iterations {
            let mut pipeline_results: HashMap<String, Value> = HashMap::new();

            for stage in 0..self.stages {
                let stage_start = Instant::now();

                // Execute tasks in this stage (can be parallel in real impl)
                for task in 0..self.tasks_per_stage {
                    let task_id = format!("stage{}_task{}", stage, task);

                    // Simulate task execution with dependency check
                    if stage > 0 {
                        // Check dependencies from previous stage
                        let _deps_satisfied = (0..self.tasks_per_stage)
                            .all(|t| pipeline_results.contains_key(&format!("stage{}_task{}", stage - 1, t)));
                    }

                    // Execute task (simulated computation)
                    let result = (0..100).fold(0u64, |acc, x| acc.wrapping_add(x));
                    pipeline_results.insert(task_id, json!({"output": result}));
                    total_tasks_executed += 1;
                }

                stage_timings.push(stage_start.elapsed().as_micros() as f64);
            }
        }

        let elapsed = start.elapsed();

        // Calculate stage statistics
        let avg_stage_time = stage_timings.iter().sum::<f64>() / stage_timings.len() as f64;
        stage_timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p95_stage_time = stage_timings[(stage_timings.len() as f64 * 0.95) as usize];

        Ok(json!({
            "stages": self.stages,
            "tasks_per_stage": self.tasks_per_stage,
            "iterations": self.iterations,
            "total_tasks_executed": total_tasks_executed,
            "avg_stage_time_us": avg_stage_time,
            "p95_stage_time_us": p95_stage_time,
            "throughput_tasks_per_sec": total_tasks_executed as f64 / elapsed.as_secs_f64(),
            "avg_pipeline_ms": elapsed.as_millis() as f64 / self.iterations as f64
        }))
    }

    fn description(&self) -> Option<String> {
        Some("Benchmarks DAG pipeline orchestration with dependencies".to_string())
    }

    fn category(&self) -> Option<String> {
        Some("workflows".to_string())
    }
}

/// Benchmark adapter for task execution runtime.
///
/// Tests the overhead of the task execution framework.
pub struct TaskExecutionBenchmark {
    iterations: usize,
    task_complexity: usize,
}

impl TaskExecutionBenchmark {
    pub fn new() -> Self {
        Self {
            iterations: 100,
            task_complexity: 1000,
        }
    }
}

impl Default for TaskExecutionBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchTarget for TaskExecutionBenchmark {
    fn id(&self) -> String {
        "task-execution".to_string()
    }

    fn run(&self) -> Result<Value, Box<dyn Error>> {
        let start = Instant::now();

        let mut task_durations: Vec<f64> = Vec::new();
        let mut total_work_units = 0;

        for _ in 0..self.iterations {
            let task_start = Instant::now();

            // Simulate task context creation
            let context = HashMap::from([
                ("task_id".to_string(), json!("benchmark-task")),
                ("config".to_string(), json!({"complexity": self.task_complexity})),
            ]);

            // Execute task workload
            let result: u64 = (0..self.task_complexity)
                .map(|i| (i as f64).sqrt() as u64)
                .sum();

            // Simulate result collection
            let _output = json!({
                "result": result,
                "context": context,
            });

            let task_elapsed = task_start.elapsed();
            task_durations.push(task_elapsed.as_micros() as f64);
            total_work_units += self.task_complexity;
        }

        let elapsed = start.elapsed();

        // Calculate statistics
        let avg_duration = task_durations.iter().sum::<f64>() / task_durations.len() as f64;
        task_durations.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min_duration = task_durations[0];
        let max_duration = task_durations[task_durations.len() - 1];
        let p50_duration = task_durations[task_durations.len() / 2];
        let p99_duration = task_durations[(task_durations.len() as f64 * 0.99) as usize];

        Ok(json!({
            "iterations": self.iterations,
            "task_complexity": self.task_complexity,
            "total_work_units": total_work_units,
            "avg_task_us": avg_duration,
            "min_task_us": min_duration,
            "max_task_us": max_duration,
            "p50_task_us": p50_duration,
            "p99_task_us": p99_duration,
            "throughput_tasks_per_sec": self.iterations as f64 / elapsed.as_secs_f64(),
            "work_units_per_sec": total_work_units as f64 / elapsed.as_secs_f64()
        }))
    }

    fn description(&self) -> Option<String> {
        Some("Benchmarks task execution framework overhead".to_string())
    }

    fn category(&self) -> Option<String> {
        Some("workflows".to_string())
    }
}

/// Benchmark adapter for data loading operations.
///
/// Tests the performance of dataset loading and preprocessing.
pub struct DataLoadingBenchmark {
    record_count: usize,
    iterations: usize,
}

impl DataLoadingBenchmark {
    pub fn new() -> Self {
        Self {
            record_count: 1000,
            iterations: 50,
        }
    }
}

impl Default for DataLoadingBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchTarget for DataLoadingBenchmark {
    fn id(&self) -> String {
        "data-loading".to_string()
    }

    fn run(&self) -> Result<Value, Box<dyn Error>> {
        let start = Instant::now();

        let mut total_records_processed = 0;
        let mut load_times: Vec<f64> = Vec::new();
        let mut preprocess_times: Vec<f64> = Vec::new();

        for _ in 0..self.iterations {
            // Simulate data loading
            let load_start = Instant::now();
            let raw_data: Vec<String> = (0..self.record_count)
                .map(|i| format!("Record {} with some sample text content for processing", i))
                .collect();
            load_times.push(load_start.elapsed().as_micros() as f64);

            // Simulate preprocessing
            let preprocess_start = Instant::now();
            let _processed: Vec<Vec<&str>> = raw_data
                .iter()
                .map(|record| {
                    // Tokenization simulation
                    record.split_whitespace().collect()
                })
                .collect();
            preprocess_times.push(preprocess_start.elapsed().as_micros() as f64);

            total_records_processed += self.record_count;
        }

        let elapsed = start.elapsed();

        // Calculate statistics
        let avg_load_time = load_times.iter().sum::<f64>() / load_times.len() as f64;
        let avg_preprocess_time = preprocess_times.iter().sum::<f64>() / preprocess_times.len() as f64;

        load_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        preprocess_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        Ok(json!({
            "record_count": self.record_count,
            "iterations": self.iterations,
            "total_records_processed": total_records_processed,
            "avg_load_time_us": avg_load_time,
            "avg_preprocess_time_us": avg_preprocess_time,
            "p95_load_time_us": load_times[(load_times.len() as f64 * 0.95) as usize],
            "p95_preprocess_time_us": preprocess_times[(preprocess_times.len() as f64 * 0.95) as usize],
            "throughput_records_per_sec": total_records_processed as f64 / elapsed.as_secs_f64(),
            "avg_iteration_ms": elapsed.as_millis() as f64 / self.iterations as f64
        }))
    }

    fn description(&self) -> Option<String> {
        Some("Benchmarks dataset loading and preprocessing".to_string())
    }

    fn category(&self) -> Option<String> {
        Some("workflows".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_orchestration_benchmark() {
        let bench = PipelineOrchestrationBenchmark::new();
        let result = bench.run().unwrap();
        assert!(result.get("total_tasks_executed").is_some());
    }

    #[test]
    fn test_task_execution_benchmark() {
        let bench = TaskExecutionBenchmark::new();
        let result = bench.run().unwrap();
        assert!(result.get("throughput_tasks_per_sec").is_some());
    }

    #[test]
    fn test_data_loading_benchmark() {
        let bench = DataLoadingBenchmark::new();
        let result = bench.run().unwrap();
        assert!(result.get("total_records_processed").is_some());
    }
}
