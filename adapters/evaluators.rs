//! Evaluator Benchmark Adapters
//!
//! These adapters expose the LLM Research Lab evaluation operations
//! as benchmark targets for alpha-testing evaluators.

use serde_json::{json, Value};
use std::error::Error;
use std::time::Instant;

use super::BenchTarget;

/// Benchmark adapter for batch evaluation operations.
///
/// Tests the performance of evaluating multiple prediction-reference
/// pairs in batch mode.
pub struct BatchEvaluationBenchmark {
    batch_size: usize,
    iterations: usize,
}

impl BatchEvaluationBenchmark {
    pub fn new() -> Self {
        Self {
            batch_size: 100,
            iterations: 50,
        }
    }

    pub fn with_config(batch_size: usize, iterations: usize) -> Self {
        Self {
            batch_size,
            iterations,
        }
    }
}

impl Default for BatchEvaluationBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchTarget for BatchEvaluationBenchmark {
    fn id(&self) -> String {
        "batch-evaluation".to_string()
    }

    fn run(&self) -> Result<Value, Box<dyn Error>> {
        let start = Instant::now();

        // Generate sample evaluation pairs
        let predictions: Vec<String> = (0..self.batch_size)
            .map(|i| format!("Prediction {} with some text content", i))
            .collect();

        let references: Vec<String> = (0..self.batch_size)
            .map(|i| format!("Reference {} with matching content", i))
            .collect();

        let mut total_evaluations = 0;
        let mut total_score = 0.0;

        for _ in 0..self.iterations {
            for (pred, ref_text) in predictions.iter().zip(references.iter()) {
                // Simulate evaluation computation
                let pred_words: Vec<&str> = pred.split_whitespace().collect();
                let ref_words: Vec<&str> = ref_text.split_whitespace().collect();

                // Calculate overlap score
                let overlap = pred_words
                    .iter()
                    .filter(|w| ref_words.contains(w))
                    .count();
                let score = overlap as f64 / pred_words.len().max(ref_words.len()) as f64;

                total_score += score;
                total_evaluations += 1;
            }
        }

        let elapsed = start.elapsed();

        Ok(json!({
            "batch_size": self.batch_size,
            "iterations": self.iterations,
            "total_evaluations": total_evaluations,
            "average_score": total_score / total_evaluations as f64,
            "throughput_evals_per_sec": total_evaluations as f64 / elapsed.as_secs_f64(),
            "avg_batch_ms": elapsed.as_millis() as f64 / self.iterations as f64,
            "avg_eval_us": elapsed.as_micros() as f64 / total_evaluations as f64
        }))
    }

    fn description(&self) -> Option<String> {
        Some("Benchmarks batch evaluation of prediction-reference pairs".to_string())
    }

    fn category(&self) -> Option<String> {
        Some("evaluators".to_string())
    }
}

/// Benchmark adapter for comparative evaluation.
///
/// Tests performance of comparing multiple model outputs against each other.
pub struct ComparativeEvaluationBenchmark {
    models: usize,
    samples: usize,
    iterations: usize,
}

impl ComparativeEvaluationBenchmark {
    pub fn new() -> Self {
        Self {
            models: 5,
            samples: 50,
            iterations: 20,
        }
    }
}

impl Default for ComparativeEvaluationBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchTarget for ComparativeEvaluationBenchmark {
    fn id(&self) -> String {
        "comparative-evaluation".to_string()
    }

    fn run(&self) -> Result<Value, Box<dyn Error>> {
        let start = Instant::now();

        // Generate model outputs
        let model_outputs: Vec<Vec<String>> = (0..self.models)
            .map(|m| {
                (0..self.samples)
                    .map(|s| format!("Model {} output for sample {}", m, s))
                    .collect()
            })
            .collect();

        let mut total_comparisons = 0;
        let mut rankings: Vec<Vec<usize>> = Vec::new();

        for _ in 0..self.iterations {
            for sample_idx in 0..self.samples {
                // Compare all model outputs for this sample
                let mut scores: Vec<(usize, f64)> = model_outputs
                    .iter()
                    .enumerate()
                    .map(|(model_idx, outputs)| {
                        let output = &outputs[sample_idx];
                        // Simulate scoring (based on output length as proxy)
                        let score = output.len() as f64 + (model_idx as f64 * 0.1);
                        (model_idx, score)
                    })
                    .collect();

                // Sort by score descending
                scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                // Record ranking
                let ranking: Vec<usize> = scores.iter().map(|(idx, _)| *idx).collect();
                rankings.push(ranking);

                total_comparisons += self.models * (self.models - 1) / 2; // Pairwise comparisons
            }
        }

        let elapsed = start.elapsed();

        // Calculate win rates for each model
        let mut win_counts = vec![0usize; self.models];
        for ranking in &rankings {
            if !ranking.is_empty() {
                win_counts[ranking[0]] += 1;
            }
        }

        let win_rates: Vec<f64> = win_counts
            .iter()
            .map(|&count| count as f64 / rankings.len() as f64)
            .collect();

        Ok(json!({
            "models": self.models,
            "samples": self.samples,
            "iterations": self.iterations,
            "total_comparisons": total_comparisons,
            "win_rates": win_rates,
            "throughput_comparisons_per_sec": total_comparisons as f64 / elapsed.as_secs_f64(),
            "avg_sample_comparison_us": elapsed.as_micros() as f64 / (self.iterations * self.samples) as f64
        }))
    }

    fn description(&self) -> Option<String> {
        Some("Benchmarks comparative evaluation across multiple models".to_string())
    }

    fn category(&self) -> Option<String> {
        Some("evaluators".to_string())
    }
}

/// Benchmark adapter for LLM-as-Judge evaluation pattern.
///
/// Simulates the performance characteristics of using an LLM to judge outputs.
pub struct LLMJudgeBenchmark {
    evaluations: usize,
    simulated_latency_ms: u64,
}

impl LLMJudgeBenchmark {
    pub fn new() -> Self {
        Self {
            evaluations: 20,
            simulated_latency_ms: 5, // Simulated API latency
        }
    }
}

impl Default for LLMJudgeBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl BenchTarget for LLMJudgeBenchmark {
    fn id(&self) -> String {
        "llm-judge-evaluation".to_string()
    }

    fn run(&self) -> Result<Value, Box<dyn Error>> {
        let start = Instant::now();

        // Sample outputs to evaluate
        let outputs = vec![
            "The capital of France is Paris.",
            "Machine learning uses algorithms to learn patterns from data.",
            "Climate change is caused by greenhouse gas emissions.",
            "The Earth orbits around the Sun.",
            "Water boils at 100 degrees Celsius at sea level.",
        ];

        let criteria = vec![
            "accuracy",
            "completeness",
            "clarity",
            "relevance",
        ];

        let mut total_judgments = 0;
        let mut judgment_scores: Vec<f64> = Vec::new();

        for _ in 0..self.evaluations {
            for output in &outputs {
                for criterion in &criteria {
                    // Simulate LLM judge evaluation
                    // In real usage, this would call the LLM API
                    std::thread::sleep(std::time::Duration::from_millis(self.simulated_latency_ms));

                    // Generate simulated score based on text characteristics
                    let score = (output.len() as f64 / 50.0).min(1.0) * 0.8
                        + (criterion.len() as f64 / 20.0) * 0.2;

                    judgment_scores.push(score);
                    total_judgments += 1;
                }
            }
        }

        let elapsed = start.elapsed();

        // Calculate aggregate statistics
        let avg_score = judgment_scores.iter().sum::<f64>() / judgment_scores.len() as f64;
        let mut sorted_scores = judgment_scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median_score = sorted_scores[sorted_scores.len() / 2];
        let min_score = sorted_scores[0];
        let max_score = sorted_scores[sorted_scores.len() - 1];

        Ok(json!({
            "evaluations": self.evaluations,
            "outputs_per_eval": outputs.len(),
            "criteria": criteria.len(),
            "total_judgments": total_judgments,
            "average_score": avg_score,
            "median_score": median_score,
            "min_score": min_score,
            "max_score": max_score,
            "simulated_latency_ms": self.simulated_latency_ms,
            "actual_throughput_judgments_per_sec": total_judgments as f64 / elapsed.as_secs_f64(),
            "avg_judgment_ms": elapsed.as_millis() as f64 / total_judgments as f64
        }))
    }

    fn description(&self) -> Option<String> {
        Some("Benchmarks LLM-as-Judge evaluation pattern".to_string())
    }

    fn category(&self) -> Option<String> {
        Some("evaluators".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_evaluation_benchmark() {
        let bench = BatchEvaluationBenchmark::with_config(10, 5);
        let result = bench.run().unwrap();
        assert!(result.get("total_evaluations").is_some());
        assert_eq!(result.get("batch_size").unwrap(), 10);
    }

    #[test]
    fn test_comparative_evaluation_benchmark() {
        let bench = ComparativeEvaluationBenchmark::new();
        let result = bench.run().unwrap();
        assert!(result.get("win_rates").is_some());
    }

    #[test]
    fn test_llm_judge_benchmark() {
        let mut bench = LLMJudgeBenchmark::new();
        bench.evaluations = 2;
        bench.simulated_latency_ms = 1;
        let result = bench.run().unwrap();
        assert!(result.get("average_score").is_some());
    }
}
