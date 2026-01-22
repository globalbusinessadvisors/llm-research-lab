//! CLI argument parsing

use clap::{Parser, Subcommand};

use crate::commands::{
    agents::AgentsCommands, auth::AuthCommands, config::ConfigCommands, datasets::DatasetsCommands,
    evaluations::EvaluationsCommands, experiments::ExperimentsCommands, models::ModelsCommands,
    prompts::PromptsCommands, run::RunCommands,
};
use crate::output::OutputFormat;

/// LLM Research Lab CLI
///
/// A command-line tool for managing LLM research experiments, models,
/// datasets, prompts, and evaluations.
#[derive(Parser, Debug)]
#[command(name = "llm-research")]
#[command(author = "LLM Research Lab Team")]
#[command(version)]
#[command(about = "CLI for LLM Research Lab", long_about = None)]
#[command(propagate_version = true)]
pub struct Cli {
    /// Output format (table, json, yaml)
    #[arg(short, long, global = true, default_value = "table", env = "LLM_RESEARCH_OUTPUT")]
    pub output: OutputFormat,

    /// API base URL
    #[arg(long, global = true, env = "LLM_RESEARCH_API_URL")]
    pub api_url: Option<String>,

    /// API key for authentication
    #[arg(long, global = true, env = "LLM_RESEARCH_API_KEY")]
    pub api_key: Option<String>,

    /// Profile to use from config
    #[arg(short, long, global = true, env = "LLM_RESEARCH_PROFILE")]
    pub profile: Option<String>,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    pub verbose: bool,

    /// Disable colored output
    #[arg(long, global = true)]
    pub no_color: bool,

    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Research agents for hypothesis evaluation and metrics computation
    #[command(alias = "agent")]
    Agents(AgentsCommands),

    /// Manage authentication and credentials
    #[command(alias = "login")]
    Auth(AuthCommands),

    /// Manage CLI configuration
    #[command(alias = "cfg")]
    Config(ConfigCommands),

    /// Manage experiments
    #[command(alias = "exp")]
    Experiments(ExperimentsCommands),

    /// Manage models
    #[command(alias = "model")]
    Models(ModelsCommands),

    /// Manage datasets
    #[command(alias = "data")]
    Datasets(DatasetsCommands),

    /// Manage prompt templates
    #[command(alias = "prompt")]
    Prompts(PromptsCommands),

    /// Manage evaluations
    #[command(alias = "eval")]
    Evaluations(EvaluationsCommands),

    /// Run benchmark targets
    #[command(alias = "bench")]
    Run(RunCommands),
}
