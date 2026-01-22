//! LLM Research Lab CLI
//!
//! Command-line interface for managing LLM research experiments, models,
//! datasets, prompts, and evaluations.

use anyhow::Result;
use clap::Parser;

mod cli;
mod commands;
mod config;
mod context;
mod output;

use cli::{Cli, Commands};
use context::Context;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("llm_research=info".parse()?)
                .add_directive("warn".parse()?),
        )
        .with_target(false)
        .init();

    // Parse CLI arguments
    let cli = Cli::parse();

    // Create context
    let ctx = Context::new(&cli)?;

    // Execute command
    match cli.command {
        Commands::Agents(cmd) => commands::agents::execute(&ctx, cmd).await,
        Commands::Auth(cmd) => commands::auth::execute(&ctx, cmd).await,
        Commands::Config(cmd) => commands::config::execute(&ctx, cmd).await,
        Commands::Experiments(cmd) => commands::experiments::execute(&ctx, cmd).await,
        Commands::Models(cmd) => commands::models::execute(&ctx, cmd).await,
        Commands::Datasets(cmd) => commands::datasets::execute(&ctx, cmd).await,
        Commands::Prompts(cmd) => commands::prompts::execute(&ctx, cmd).await,
        Commands::Evaluations(cmd) => commands::evaluations::execute(&ctx, cmd).await,
        Commands::Run(cmd) => commands::run::execute(&ctx, cmd).await,
    }
}
