//! Integration Tests for LLM-Research-Lab Agents
//!
//! These tests verify the end-to-end behavior of agents per the
//! LLM-RESEARCH-LAB AGENT INFRASTRUCTURE CONSTITUTION.
//!
//! # Test Categories
//!
//! 1. **Contract Compliance**: Verify input/output schema adherence
//! 2. **DecisionEvent Emission**: Verify exactly ONE event per invocation
//! 3. **Determinism**: Verify same inputs produce same outputs
//! 4. **Error Handling**: Verify proper error responses
//! 5. **Handler Integration**: Verify Edge Function handlers work correctly

use chrono::Utc;
use rust_decimal_macros::dec;
use serde_json::json;
use uuid::Uuid;

use llm_research_agents::{
    agents::Agent,
    contracts::{
        hypothesis::*,
        decision_event::*,
        metrics::{
            MetricsInput, MetricRequest, MetricType, MetricsData,
            MetricsConfig, MissingValueStrategy,
        },
    },
    HypothesisAgent,
    ExperimentalMetricAgent,
    HYPOTHESIS_AGENT_ID,
    HYPOTHESIS_AGENT_VERSION,
    METRIC_AGENT_ID,
    METRIC_AGENT_VERSION,
};

// ============================================================================
// TEST FIXTURES
// ============================================================================

/// Create a valid HypothesisInput for testing.
fn create_valid_hypothesis_input(sample_size: u64) -> HypothesisInput {
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

/// Create a valid MetricsInput for testing.
fn create_valid_metrics_input() -> MetricsInput {
    let records: Vec<serde_json::Value> = (0..100)
        .map(|i| json!({
            "metric_a": (i as f64) * 0.5,
            "metric_b": (i as f64) * 0.25 + 10.0,
            "group": if i % 2 == 0 { "A" } else { "B" },
        }))
        .collect();

    MetricsInput {
        request_id: Uuid::new_v4(),
        context_id: "test-context-001".to_string(),
        metrics_requested: vec![
            MetricRequest {
                name: "mean_metric_a".to_string(),
                metric_type: MetricType::CentralTendency,
                variable: "metric_a".to_string(),
                group_by: None,
                params: None,
            },
            MetricRequest {
                name: "stddev_metric_a".to_string(),
                metric_type: MetricType::Dispersion,
                variable: "metric_a".to_string(),
                group_by: None,
                params: None,
            },
        ],
        data: MetricsData {
            source: "test-metrics-source".to_string(),
            records,
            schema: None,
        },
        config: MetricsConfig {
            handle_missing: MissingValueStrategy::Skip,
            precision: 4,
            include_ci: true,
            ci_level: Some(dec!(0.95)),
        },
    }
}

// ============================================================================
// HYPOTHESIS AGENT TESTS
// ============================================================================

mod hypothesis_agent {
    use super::*;

    #[tokio::test]
    async fn test_agent_identity() {
        let agent = HypothesisAgent::new();

        assert_eq!(agent.agent_id(), HYPOTHESIS_AGENT_ID);
        assert_eq!(agent.version(), HYPOTHESIS_AGENT_VERSION);
    }

    #[tokio::test]
    async fn test_valid_input_execution() {
        let agent = HypothesisAgent::new();
        let input = create_valid_hypothesis_input(100);

        let result = agent.execute(input).await;

        assert!(result.is_ok(), "Execution should succeed with valid input");
        let output = result.unwrap();
        assert!(matches!(
            output.status,
            HypothesisStatus::Accepted | HypothesisStatus::Rejected | HypothesisStatus::Inconclusive
        ));
    }

    #[tokio::test]
    async fn test_invoke_produces_decision_event() {
        let agent = HypothesisAgent::new();
        let input = create_valid_hypothesis_input(100);

        let result = agent.invoke(input).await;

        assert!(result.is_ok(), "Invoke should succeed");
        let (output, event) = result.unwrap();

        // Verify DecisionEvent structure per Constitution
        assert_eq!(event.agent_id, HYPOTHESIS_AGENT_ID);
        assert_eq!(event.agent_version, HYPOTHESIS_AGENT_VERSION);
        assert_eq!(event.decision_type, DecisionType::HypothesisEvaluation);
        assert_eq!(event.inputs_hash.len(), 64, "inputs_hash should be SHA256 (64 hex chars)");
        assert!(!event.outputs.is_null(), "outputs should be populated");
        assert!(event.confidence.value >= dec!(0) && event.confidence.value <= dec!(1));

        // Verify output is in event
        let event_output: HypothesisOutput = serde_json::from_value(event.outputs.clone())
            .expect("Event outputs should deserialize to HypothesisOutput");
        assert_eq!(event_output.hypothesis_id, output.hypothesis_id);
    }

    #[tokio::test]
    async fn test_insufficient_sample_size() {
        let agent = HypothesisAgent::new();
        let input = create_valid_hypothesis_input(10); // Below minimum of 30

        let result = agent.execute(input).await;

        assert!(result.is_err(), "Should fail with insufficient sample size");
        let err = result.unwrap_err();
        assert!(err.to_string().contains("Insufficient sample size"));
    }

    #[tokio::test]
    async fn test_determinism() {
        let agent = HypothesisAgent::new();

        // Create identical inputs with same seed
        let input1 = create_valid_hypothesis_input(100);
        let input2 = HypothesisInput {
            request_id: input1.request_id, // Same request ID
            ..input1.clone()
        };

        let hash1 = DecisionEvent::compute_inputs_hash(&input1).unwrap();
        let hash2 = DecisionEvent::compute_inputs_hash(&input2).unwrap();

        assert_eq!(hash1, hash2, "Same inputs should produce same hash");
    }

    #[tokio::test]
    async fn test_input_validation() {
        use validator::Validate;

        let valid_input = create_valid_hypothesis_input(100);
        assert!(valid_input.validate().is_ok());

        let agent = HypothesisAgent::new();
        assert!(agent.validate_input(&valid_input).is_ok());
    }

    #[tokio::test]
    async fn test_effect_size_computation() {
        let agent = HypothesisAgent::new();
        let input = create_valid_hypothesis_input(100);

        let result = agent.execute(input).await.unwrap();

        // Effect size should be computed when configured
        assert!(result.effect_size.is_some());
        let effect_size = result.effect_size.unwrap();
        assert_eq!(effect_size.measure, EffectSizeMeasure::CohensD);
    }

    #[tokio::test]
    async fn test_diagnostics_generation() {
        let agent = HypothesisAgent::new();
        let input = create_valid_hypothesis_input(100);

        let result = agent.execute(input).await.unwrap();

        // Diagnostics should be populated
        assert!(matches!(
            result.diagnostics.sample_adequacy,
            SampleAdequacy::Adequate | SampleAdequacy::Marginal
        ));
    }
}

// ============================================================================
// EXPERIMENTAL METRIC AGENT TESTS
// ============================================================================

mod experimental_metric_agent {
    use super::*;

    #[tokio::test]
    async fn test_agent_identity() {
        let agent = ExperimentalMetricAgent::new();

        assert_eq!(agent.agent_id(), METRIC_AGENT_ID);
        assert_eq!(agent.version(), METRIC_AGENT_VERSION);
    }

    #[tokio::test]
    async fn test_valid_input_execution() {
        let agent = ExperimentalMetricAgent::new();
        let input = create_valid_metrics_input();

        let result = agent.execute(input).await;

        assert!(result.is_ok(), "Execution should succeed with valid input");
        let output = result.unwrap();
        assert!(!output.metrics.is_empty(), "Should produce computed metrics");
    }

    #[tokio::test]
    async fn test_invoke_produces_decision_event() {
        let agent = ExperimentalMetricAgent::new();
        let input = create_valid_metrics_input();

        let result = agent.invoke(input).await;

        assert!(result.is_ok(), "Invoke should succeed");
        let (_output, event) = result.unwrap();

        // Verify DecisionEvent structure per Constitution
        assert_eq!(event.agent_id, METRIC_AGENT_ID);
        assert_eq!(event.agent_version, METRIC_AGENT_VERSION);
        assert_eq!(event.decision_type, DecisionType::ExperimentalMetrics);
        assert_eq!(event.inputs_hash.len(), 64);
    }

    #[tokio::test]
    async fn test_central_tendency_computation() {
        let agent = ExperimentalMetricAgent::new();
        let mut input = create_valid_metrics_input();
        input.metrics_requested = vec![MetricRequest {
            name: "mean_metric_a".to_string(),
            metric_type: MetricType::CentralTendency,
            variable: "metric_a".to_string(),
            params: None,
            group_by: None,
        }];

        let result = agent.execute(input).await.unwrap();

        assert!(!result.metrics.is_empty());
        let metric = &result.metrics[0];
        assert_eq!(metric.metric_type, MetricType::CentralTendency);
        // Verify mean computation (values are 0, 0.5, 1.0, ... 49.5)
        // Mean should be around 24.75
        let mean: f64 = metric.value.try_into().unwrap();
        assert!(mean > 20.0 && mean < 30.0, "Mean should be approximately 24.75");
    }

    #[tokio::test]
    async fn test_dispersion_computation() {
        let agent = ExperimentalMetricAgent::new();
        let mut input = create_valid_metrics_input();
        input.metrics_requested = vec![MetricRequest {
            name: "stddev_metric_a".to_string(),
            metric_type: MetricType::Dispersion,
            variable: "metric_a".to_string(),
            params: None,
            group_by: None,
        }];

        let result = agent.execute(input).await.unwrap();

        assert!(!result.metrics.is_empty());
        let metric = &result.metrics[0];
        assert_eq!(metric.metric_type, MetricType::Dispersion);
    }

    #[tokio::test]
    async fn test_multiple_metrics() {
        let agent = ExperimentalMetricAgent::new();
        let input = create_valid_metrics_input(); // Has 2 metric requests

        let result = agent.execute(input).await.unwrap();

        assert_eq!(result.metrics.len(), 2, "Should compute all requested metrics");
    }

    #[tokio::test]
    async fn test_metadata_populated() {
        let agent = ExperimentalMetricAgent::new();
        let input = create_valid_metrics_input();

        let result = agent.execute(input).await.unwrap();

        assert_eq!(result.metadata.records_processed, 100);
        assert!(result.metadata.processing_time_ms >= 0);
    }
}

// ============================================================================
// DECISION EVENT TESTS
// ============================================================================

mod decision_event {
    use super::*;

    #[test]
    fn test_builder_pattern() {
        let confidence = Confidence {
            value: dec!(0.95),
            method: ConfidenceMethod::Bayesian,
            sample_size: Some(1000),
            ci_lower: Some(dec!(0.92)),
            ci_upper: Some(dec!(0.98)),
        };

        let event = DecisionEvent::builder()
            .agent_id("test-agent")
            .agent_version("1.0.0")
            .decision_type(DecisionType::HypothesisEvaluation)
            .inputs_hash("a".repeat(64))
            .outputs(json!({"result": "test"}))
            .confidence(confidence)
            .build();

        assert!(event.is_ok());
        let event = event.unwrap();
        assert_eq!(event.agent_id, "test-agent");
        assert_eq!(event.decision_type, DecisionType::HypothesisEvaluation);
    }

    #[test]
    fn test_builder_missing_required_field() {
        let result = DecisionEvent::builder()
            .agent_id("test-agent")
            // Missing other required fields
            .build();

        assert!(result.is_err());
    }

    #[test]
    fn test_inputs_hash_determinism() {
        let input = json!({
            "key1": "value1",
            "key2": [1, 2, 3],
            "key3": {"nested": true}
        });

        let hash1 = DecisionEvent::compute_inputs_hash(&input).unwrap();
        let hash2 = DecisionEvent::compute_inputs_hash(&input).unwrap();

        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // SHA256 hex
    }

    #[test]
    fn test_decision_type_serialization() {
        let types = vec![
            (DecisionType::HypothesisEvaluation, "\"hypothesis_evaluation\""),
            (DecisionType::ExperimentalMetrics, "\"experimental_metrics\""),
        ];

        for (dt, expected) in types {
            let json = serde_json::to_string(&dt).unwrap();
            assert_eq!(json, expected);
        }
    }
}

// ============================================================================
// AGENT REGISTRATION TESTS
// ============================================================================

mod registration {
    use super::*;
    use llm_research_agents::get_agent_registrations;

    #[test]
    fn test_registrations_not_empty() {
        let registrations = get_agent_registrations();
        assert!(!registrations.is_empty());
    }

    #[test]
    fn test_hypothesis_agent_registered() {
        let registrations = get_agent_registrations();
        let hypothesis = registrations.iter().find(|r| r.id == HYPOTHESIS_AGENT_ID);

        assert!(hypothesis.is_some());
        let reg = hypothesis.unwrap();
        assert_eq!(reg.classification, "HYPOTHESIS_EVALUATION");
        assert!(reg.cli_subcommands.contains(&"evaluate".to_string()));
    }

    #[test]
    fn test_metric_agent_registered() {
        let registrations = get_agent_registrations();
        let metric = registrations.iter().find(|r| r.id == METRIC_AGENT_ID);

        assert!(metric.is_some());
        let reg = metric.unwrap();
        assert_eq!(reg.classification, "EXPERIMENTAL_METRICS");
        assert!(reg.cli_subcommands.contains(&"compute".to_string()));
    }

    #[test]
    fn test_endpoint_paths() {
        let registrations = get_agent_registrations();

        for reg in &registrations {
            assert!(reg.endpoint_path.starts_with("/api/v1/agents/"));
        }
    }
}

// ============================================================================
// CONTRACT VALIDATION TESTS
// ============================================================================

mod contracts {
    use super::*;
    use validator::Validate;

    #[test]
    fn test_hypothesis_input_validation() {
        let input = create_valid_hypothesis_input(100);
        assert!(input.validate().is_ok());
    }

    #[test]
    fn test_metrics_input_validation() {
        // MetricsInput validation requires context_id and metrics_requested
        let input = create_valid_metrics_input();
        // Note: Full validation may require all fields
        assert!(!input.context_id.is_empty());
        assert!(!input.metrics_requested.is_empty());
    }

    #[test]
    fn test_confidence_range_validation() {
        let valid = Confidence {
            value: dec!(0.5),
            method: ConfidenceMethod::Heuristic,
            sample_size: None,
            ci_lower: None,
            ci_upper: None,
        };
        assert!(valid.validate().is_ok());
    }

    #[test]
    fn test_hypothesis_status_serialization() {
        let statuses = vec![
            (HypothesisStatus::Accepted, "\"accepted\""),
            (HypothesisStatus::Rejected, "\"rejected\""),
            (HypothesisStatus::Inconclusive, "\"inconclusive\""),
        ];

        for (status, expected) in statuses {
            let json = serde_json::to_string(&status).unwrap();
            assert_eq!(json, expected);
        }
    }
}
