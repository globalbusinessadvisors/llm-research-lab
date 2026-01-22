//! Agent commands
//!
//! CLI commands for interacting with LLM-Research-Lab agents.
//!
//! # CLI Invocation (per PROMPT 3)
//!
//! ```bash
//! # Hypothesis evaluation
//! llm-research agents hypothesis evaluate --input hypothesis.json
//! llm-research agents hypothesis evaluate --stdin
//! llm-research agents hypothesis inspect --event-id <uuid>
//!
//! # Metrics computation
//! llm-research agents metric compute --input metrics.json
//! llm-research agents metric inspect --event-id <uuid>
//! ```

use anyhow::{Context as _, Result};
use clap::{Args, Subcommand};
use std::io::{self, Read};
use std::path::PathBuf;
use uuid::Uuid;

use crate::context::Context;
use crate::output::{print_field, print_section};

/// Agent management commands
#[derive(Debug, Args)]
pub struct AgentsCommands {
    #[command(subcommand)]
    pub command: AgentsSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum AgentsSubcommand {
    /// Hypothesis evaluation agent commands
    #[command(alias = "hyp")]
    Hypothesis(HypothesisCommands),

    /// Experimental metrics agent commands
    #[command(alias = "met")]
    Metric(MetricCommands),

    /// List available agents
    List,

    /// Show agent registration info
    Info {
        /// Agent ID
        agent_id: String,
    },
}

/// Hypothesis agent commands
#[derive(Debug, Args)]
pub struct HypothesisCommands {
    #[command(subcommand)]
    pub command: HypothesisSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum HypothesisSubcommand {
    /// Evaluate a hypothesis
    Evaluate {
        /// Input file path (JSON)
        #[arg(short, long)]
        input: Option<PathBuf>,

        /// Read input from stdin
        #[arg(long)]
        stdin: bool,

        /// Output file path (optional)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Inspect a decision event
    Inspect {
        /// Decision event ID
        #[arg(long)]
        event_id: Uuid,
    },

    /// Validate input without executing
    Validate {
        /// Input file path (JSON)
        #[arg(short, long)]
        input: Option<PathBuf>,

        /// Read input from stdin
        #[arg(long)]
        stdin: bool,
    },
}

/// Metric agent commands
#[derive(Debug, Args)]
pub struct MetricCommands {
    #[command(subcommand)]
    pub command: MetricSubcommand,
}

#[derive(Debug, Subcommand)]
pub enum MetricSubcommand {
    /// Compute metrics
    Compute {
        /// Input file path (JSON)
        #[arg(short, long)]
        input: Option<PathBuf>,

        /// Read input from stdin
        #[arg(long)]
        stdin: bool,

        /// Output file path (optional)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Inspect a decision event
    Inspect {
        /// Decision event ID
        #[arg(long)]
        event_id: Uuid,
    },
}

/// Execute agent commands
pub async fn execute(ctx: &Context, cmd: AgentsCommands) -> Result<()> {
    match cmd.command {
        AgentsSubcommand::Hypothesis(hyp_cmd) => execute_hypothesis(ctx, hyp_cmd).await,
        AgentsSubcommand::Metric(met_cmd) => execute_metric(ctx, met_cmd).await,
        AgentsSubcommand::List => list_agents(ctx).await,
        AgentsSubcommand::Info { agent_id } => agent_info(ctx, &agent_id).await,
    }
}

/// Execute hypothesis agent commands
async fn execute_hypothesis(ctx: &Context, cmd: HypothesisCommands) -> Result<()> {
    match cmd.command {
        HypothesisSubcommand::Evaluate { input, stdin, output } => {
            hypothesis_evaluate(ctx, input, stdin, output).await
        }
        HypothesisSubcommand::Inspect { event_id } => {
            hypothesis_inspect(ctx, event_id).await
        }
        HypothesisSubcommand::Validate { input, stdin } => {
            hypothesis_validate(ctx, input, stdin).await
        }
    }
}

/// Execute metric agent commands
async fn execute_metric(ctx: &Context, cmd: MetricCommands) -> Result<()> {
    match cmd.command {
        MetricSubcommand::Compute { input, stdin, output } => {
            metric_compute(ctx, input, stdin, output).await
        }
        MetricSubcommand::Inspect { event_id } => {
            metric_inspect(ctx, event_id).await
        }
    }
}

/// Read input from file or stdin
fn read_input(file: Option<PathBuf>, use_stdin: bool) -> Result<String> {
    if use_stdin {
        let mut buffer = String::new();
        io::stdin().read_to_string(&mut buffer)
            .context("Failed to read from stdin")?;
        Ok(buffer)
    } else if let Some(path) = file {
        std::fs::read_to_string(&path)
            .context(format!("Failed to read file: {}", path.display()))
    } else {
        anyhow::bail!("Either --input or --stdin must be provided")
    }
}

/// Evaluate a hypothesis
async fn hypothesis_evaluate(
    ctx: &Context,
    input: Option<PathBuf>,
    use_stdin: bool,
    output: Option<PathBuf>,
) -> Result<()> {
    use llm_research_agents::{
        HypothesisAgent,
        HypothesisInput,
        RuVectorClient,
        RuVectorPersistence,
        agents::Agent,
    };

    let input_json = read_input(input, use_stdin)?;

    // Parse input
    let hypothesis_input: HypothesisInput = serde_json::from_str(&input_json)
        .context("Failed to parse hypothesis input JSON")?;

    let request_id = hypothesis_input.request_id;

    // Create agent and execute
    let agent = HypothesisAgent::new();

    let spinner = ctx.output.spinner("Evaluating hypothesis...");

    let (result, event) = agent.invoke(hypothesis_input).await
        .context("Hypothesis evaluation failed")?;

    if let Some(s) = spinner {
        s.finish_and_clear();
    }

    // Persist to ruvector-service
    let ruvector = RuVectorClient::from_env()
        .context("Failed to create RuVector client")?;

    let spinner = ctx.output.spinner("Persisting decision event...");

    match ruvector.persist_decision_event(event.clone()).await {
        Ok(persisted) => {
            if let Some(s) = spinner {
                s.finish_and_clear();
            }
            ctx.output.success(&format!(
                "Decision event persisted: {}",
                persisted.storage_ref
            ));
        }
        Err(e) => {
            if let Some(s) = spinner {
                s.finish_and_clear();
            }
            ctx.output.warn(&format!(
                "Failed to persist decision event: {}",
                e
            ));
        }
    }

    // Output result
    let output_json = serde_json::to_string_pretty(&serde_json::json!({
        "success": true,
        "request_id": request_id,
        "hypothesis_id": result.hypothesis_id,
        "status": result.status,
        "test_results": {
            "test_statistic": result.test_results.test_statistic,
            "p_value": result.test_results.p_value,
            "null_rejected": result.test_results.null_rejected,
            "decision": result.test_results.decision,
        },
        "effect_size": result.effect_size,
        "diagnostics": {
            "sample_adequacy": result.diagnostics.sample_adequacy,
            "achieved_power": result.diagnostics.achieved_power,
        },
        "recommendations": result.recommendations,
        "decision_event": {
            "id": event.id,
            "confidence": event.confidence.value,
        }
    })).context("Failed to serialize output")?;

    if let Some(output_path) = output {
        std::fs::write(&output_path, &output_json)
            .context(format!("Failed to write output to {}", output_path.display()))?;
        ctx.output.success(&format!("Output written to {}", output_path.display()));
    } else {
        println!("{}", output_json);
    }

    // Print summary
    print_section("Hypothesis Evaluation Summary");
    print_field("Status", &format!("{:?}", result.status));
    print_field("P-Value", &result.test_results.p_value.to_string());
    print_field("Decision", &result.test_results.decision);

    Ok(())
}

/// Inspect a hypothesis decision event
async fn hypothesis_inspect(ctx: &Context, event_id: Uuid) -> Result<()> {
    use llm_research_agents::{RuVectorClient, RuVectorPersistence};

    let ruvector = RuVectorClient::from_env()
        .context("Failed to create RuVector client")?;

    let spinner = ctx.output.spinner("Fetching decision event...");

    let event = ruvector.get_decision_event(event_id).await
        .context("Failed to fetch decision event")?;

    if let Some(s) = spinner {
        s.finish_and_clear();
    }

    match event {
        Some(e) => {
            print_section("Decision Event");
            print_field("ID", &e.id.to_string());
            print_field("Agent ID", &e.agent_id);
            print_field("Agent Version", &e.agent_version);
            print_field("Decision Type", &e.decision_type.to_string());
            print_field("Confidence", &e.confidence.value.to_string());
            print_field("Timestamp", &e.timestamp.to_rfc3339());
            print_field("Inputs Hash", &e.inputs_hash);

            print_section("Outputs");
            println!("{}", serde_json::to_string_pretty(&e.outputs)?);

            print_section("Constraints Applied");
            if !e.constraints_applied.scope.is_empty() {
                print_field("Scope", &e.constraints_applied.scope.join(", "));
            }
            if !e.constraints_applied.assumptions.is_empty() {
                print_field("Assumptions", &e.constraints_applied.assumptions.join(", "));
            }
            if !e.constraints_applied.limitations.is_empty() {
                print_field("Limitations", &e.constraints_applied.limitations.join(", "));
            }
        }
        None => {
            ctx.output.warn(&format!("Decision event not found: {}", event_id));
        }
    }

    Ok(())
}

/// Validate hypothesis input without executing
async fn hypothesis_validate(
    ctx: &Context,
    input: Option<PathBuf>,
    use_stdin: bool,
) -> Result<()> {
    use llm_research_agents::{HypothesisAgent, HypothesisInput, agents::Agent};
    use validator::Validate;

    let input_json = read_input(input, use_stdin)?;

    // Parse input
    let hypothesis_input: HypothesisInput = serde_json::from_str(&input_json)
        .context("Failed to parse hypothesis input JSON")?;

    // Validate
    hypothesis_input.validate()
        .context("Input validation failed")?;

    let agent = HypothesisAgent::new();
    agent.validate_input(&hypothesis_input)
        .map_err(|e| anyhow::anyhow!("Agent validation failed: {}", e))?;

    ctx.output.success("Input is valid");

    print_section("Input Summary");
    print_field("Request ID", &hypothesis_input.request_id.to_string());
    print_field("Hypothesis ID", &hypothesis_input.hypothesis.id.to_string());
    print_field("Hypothesis Name", &hypothesis_input.hypothesis.name);
    print_field("Sample Size", &hypothesis_input.experimental_data.sample_size.to_string());
    print_field("Test Method", &format!("{:?}", hypothesis_input.config.test_method));

    Ok(())
}

/// Compute metrics
async fn metric_compute(
    ctx: &Context,
    input: Option<PathBuf>,
    use_stdin: bool,
    output: Option<PathBuf>,
) -> Result<()> {
    use llm_research_agents::{
        agents::Agent,
        ExperimentalMetricAgent,
        MetricsInput,
        RuVectorClient,
        RuVectorPersistence,
    };

    let input_json = read_input(input, use_stdin)?;

    // Parse input
    let metrics_input: MetricsInput = serde_json::from_str(&input_json)
        .context("Failed to parse metrics input JSON")?;

    let request_id = metrics_input.request_id;

    // Create agent and execute
    let agent = ExperimentalMetricAgent::new();

    let spinner = ctx.output.spinner("Computing metrics...");

    let (result, event) = agent.invoke(metrics_input).await
        .map_err(|e| anyhow::anyhow!("Metrics computation failed: {}", e))?;

    if let Some(s) = spinner {
        s.finish_and_clear();
    }

    // Persist to ruvector-service
    let ruvector = RuVectorClient::from_env()
        .context("Failed to create RuVector client")?;

    let spinner = ctx.output.spinner("Persisting decision event...");

    match ruvector.persist_decision_event(event.clone()).await {
        Ok(persisted) => {
            if let Some(s) = spinner {
                s.finish_and_clear();
            }
            ctx.output.success(&format!(
                "Decision event persisted: {}",
                persisted.storage_ref
            ));
        }
        Err(e) => {
            if let Some(s) = spinner {
                s.finish_and_clear();
            }
            ctx.output.warn(&format!(
                "Failed to persist decision event: {}",
                e
            ));
        }
    }

    // Output result
    let output_json = serde_json::to_string_pretty(&serde_json::json!({
        "success": true,
        "request_id": request_id,
        "metrics": result.metrics,
        "metadata": {
            "records_processed": result.metadata.records_processed,
            "processing_time_ms": result.metadata.processing_time_ms,
        },
        "warnings": result.warnings,
        "decision_event": {
            "id": event.id,
            "confidence": event.confidence.value,
        }
    })).context("Failed to serialize output")?;

    if let Some(output_path) = output {
        std::fs::write(&output_path, &output_json)
            .context(format!("Failed to write output to {}", output_path.display()))?;
        ctx.output.success(&format!("Output written to {}", output_path.display()));
    } else {
        println!("{}", output_json);
    }

    // Print summary
    print_section("Metrics Computation Summary");
    print_field("Metrics Computed", &result.metrics.len().to_string());
    print_field("Records Processed", &result.metadata.records_processed.to_string());
    print_field("Processing Time", &format!("{}ms", result.metadata.processing_time_ms));

    Ok(())
}

/// Inspect a metric decision event
async fn metric_inspect(ctx: &Context, event_id: Uuid) -> Result<()> {
    // Reuse hypothesis inspect logic
    hypothesis_inspect(ctx, event_id).await
}

/// List available agents
async fn list_agents(ctx: &Context) -> Result<()> {
    let registrations = llm_research_agents::get_agent_registrations();

    print_section("Available Agents");

    for reg in &registrations {
        println!();
        print_field("Agent ID", &reg.id);
        print_field("Version", &reg.version);
        print_field("Classification", &reg.classification);
        print_field("CLI Command", &format!("agents {}", reg.cli_command));
        print_field("Subcommands", &reg.cli_subcommands.join(", "));
        print_field("Endpoint", &reg.endpoint_path);
    }

    ctx.output.info(&format!("\n{} agent(s) available", registrations.len()));

    Ok(())
}

/// Show agent info
async fn agent_info(ctx: &Context, agent_id: &str) -> Result<()> {
    let registrations = llm_research_agents::get_agent_registrations();

    let agent = registrations.iter().find(|r| r.id == agent_id);

    match agent {
        Some(reg) => {
            print_section("Agent Information");
            print_field("Agent ID", &reg.id);
            print_field("Version", &reg.version);
            print_field("Classification", &reg.classification);
            print_field("CLI Command", &format!("agents {}", reg.cli_command));
            print_field("Subcommands", &reg.cli_subcommands.join(", "));
            print_field("Endpoint", &reg.endpoint_path);

            // Print agent-specific info
            if reg.id == "hypothesis-agent-v1" {
                print_section("Hypothesis Agent Details");
                println!("  Purpose: Define, evaluate, and validate research hypotheses");
                println!("  Decision Type: hypothesis_evaluation");
                println!("  Scope:");
                println!("    - Define testable hypotheses");
                println!("    - Evaluate hypotheses against experimental data");
                println!("    - Emit structured hypothesis outcomes");
            } else if reg.id == "experimental-metric-agent" {
                print_section("Experimental Metric Agent Details");
                println!("  Purpose: Compute and report experimental metrics for research outcomes");
                println!("  Decision Type: experimental_metrics");
                println!("  Scope:");
                println!("    - Compute experimental metrics (central tendency, dispersion, etc.)");
                println!("    - Normalize and validate metric outputs");
                println!("    - Emit structured metric artifacts");
                println!("    - Compute confidence intervals for metrics");
                println!("    - Aggregate metrics across groups");
                println!();
                println!("  Metric Types:");
                println!("    - CentralTendency: mean, median, mode");
                println!("    - Dispersion: variance, std dev, IQR");
                println!("    - DistributionShape: skewness, kurtosis");
                println!("    - Percentile: arbitrary percentile computation");
                println!("    - Correlation: Pearson's r between variables");
                println!("    - Regression: linear regression coefficients");
                println!("    - CustomAggregation: sum, count, min, max, range");
            }
        }
        None => {
            ctx.output.warn(&format!("Agent not found: {}", agent_id));
            ctx.output.info("Use 'agents list' to see available agents");
        }
    }

    Ok(())
}
