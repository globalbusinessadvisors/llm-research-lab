//! Configuration for LLM Research Lab Cloud Run Service
//!
//! # Constitution Compliance
//!
//! Per PROMPT 2, this service DOES NOT use direct database connections.
//! All persistence is via ruvector-service.
//!
//! # Phase 7 Layer 2 Hardening
//!
//! This module enforces FAIL-FAST behavior for missing configuration.
//! NO FALLBACKS for critical environment variables in production.
//!
//! Required environment variables:
//! - RUVECTOR_SERVICE_URL: RuVector service endpoint (REQUIRED - no fallback)
//! - RUVECTOR_API_KEY: RuVector authentication (REQUIRED - use Secret Manager)
//! - LLM_OBSERVATORY_ENDPOINT: Telemetry endpoint (REQUIRED in prod)
//! - PLATFORM_ENV: dev | staging | prod
//!
//! Phase 7 Agent Identity (REQUIRED in prod):
//! - AGENT_NAME: Unique agent identifier
//! - AGENT_DOMAIN: Agent domain (e.g., "research", "infrastructure")
//! - AGENT_PHASE: Must be "phase7"
//! - AGENT_LAYER: Must be "layer2"
//! - AGENT_VERSION: Semantic version of the agent

use anyhow::{anyhow, Result};
use serde::Deserialize;
use std::env;

/// Platform environment.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum PlatformEnv {
    Dev,
    Staging,
    Prod,
}

impl Default for PlatformEnv {
    fn default() -> Self {
        Self::Dev
    }
}

/// Cloud Run service configuration.
///
/// CONSTITUTION COMPLIANCE:
/// - NO database_url field (no direct SQL access)
/// - Persistence via ruvector_service_url only
///
/// PHASE 7 LAYER 2 HARDENING:
/// - REQUIRED fields FAIL FAST if missing (no fallbacks in prod)
/// - Agent identity is mandatory for distributed tracing
/// - RuVector API key required (use Secret Manager)
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// HTTP server port
    #[serde(default = "default_port")]
    pub port: u16,

    /// Platform environment
    #[serde(default)]
    pub platform_env: PlatformEnv,

    /// RuVector service URL (REQUIRED for persistence - NO FALLBACK)
    pub ruvector_service_url: String,

    /// RuVector API key (REQUIRED - use Secret Manager)
    pub ruvector_api_key: String,

    /// RuVector request timeout in seconds
    #[serde(default = "default_ruvector_timeout")]
    pub ruvector_timeout_secs: u64,

    /// RuVector max retries
    #[serde(default = "default_ruvector_retries")]
    pub ruvector_max_retries: u32,

    /// LLM Observatory endpoint for telemetry
    pub llm_observatory_endpoint: String,

    /// Telemetry endpoint (may be same as Observatory)
    pub telemetry_endpoint: String,

    /// Service name
    #[serde(default = "default_service_name")]
    pub service_name: String,

    /// Service version
    #[serde(default = "default_service_version")]
    pub service_version: String,

    /// Log level
    #[serde(default = "default_log_level")]
    pub log_level: String,

    /// Enable telemetry to stdout
    #[serde(default = "default_telemetry_stdout")]
    pub telemetry_stdout: bool,

    /// Phase 7 Agent Identity (REQUIRED in prod/staging)
    #[serde(flatten)]
    pub agent_identity: Option<AgentIdentity>,
}

fn default_port() -> u16 {
    8080
}

fn default_ruvector_timeout() -> u64 {
    30
}

fn default_ruvector_retries() -> u32 {
    3
}

fn default_service_name() -> String {
    "llm-research-lab".to_string()
}

fn default_service_version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

fn default_log_level() -> String {
    "info".to_string()
}

fn default_telemetry_stdout() -> bool {
    true
}

/// Phase 7 Agent Identity configuration.
///
/// Required for all Phase 7 Layer 2 agents.
/// These fields enable distributed tracing and identity verification.
#[derive(Debug, Clone, Deserialize)]
pub struct AgentIdentity {
    /// Unique agent name (e.g., "llm-research-lab-primary")
    pub name: String,

    /// Agent domain (e.g., "research", "infrastructure", "security")
    pub domain: String,

    /// Agent phase - MUST be "phase7" for Phase 7 compliance
    pub phase: String,

    /// Agent layer - MUST be "layer2" for Layer 2 compliance
    pub layer: String,

    /// Agent version (semantic versioning)
    pub version: String,
}

impl AgentIdentity {
    /// Load agent identity from environment variables.
    ///
    /// # Phase 7 Layer 2 Hardening
    ///
    /// FAILS FAST if required identity variables are missing or invalid.
    /// NO FALLBACKS in production mode.
    pub fn load(platform_env: &PlatformEnv) -> Result<Self> {
        let name = env::var("AGENT_NAME")
            .map_err(|_| anyhow!("AGENT_NAME environment variable is required. ABORTING STARTUP."))?;

        let domain = env::var("AGENT_DOMAIN")
            .map_err(|_| anyhow!("AGENT_DOMAIN environment variable is required. ABORTING STARTUP."))?;

        let phase = env::var("AGENT_PHASE")
            .map_err(|_| anyhow!("AGENT_PHASE environment variable is required (must be 'phase7'). ABORTING STARTUP."))?;

        let layer = env::var("AGENT_LAYER")
            .map_err(|_| anyhow!("AGENT_LAYER environment variable is required (must be 'layer2'). ABORTING STARTUP."))?;

        let version = env::var("AGENT_VERSION")
            .map_err(|_| anyhow!("AGENT_VERSION environment variable is required. ABORTING STARTUP."))?;

        // Validate phase and layer in production/staging
        if *platform_env != PlatformEnv::Dev {
            if phase != "phase7" {
                return Err(anyhow!(
                    "AGENT_PHASE must be 'phase7' in {} environment, got '{}'. ABORTING STARTUP.",
                    match platform_env {
                        PlatformEnv::Prod => "prod",
                        PlatformEnv::Staging => "staging",
                        PlatformEnv::Dev => "dev",
                    },
                    phase
                ));
            }

            if layer != "layer2" {
                return Err(anyhow!(
                    "AGENT_LAYER must be 'layer2' in {} environment, got '{}'. ABORTING STARTUP.",
                    match platform_env {
                        PlatformEnv::Prod => "prod",
                        PlatformEnv::Staging => "staging",
                        PlatformEnv::Dev => "dev",
                    },
                    layer
                ));
            }
        }

        Ok(Self {
            name,
            domain,
            phase,
            layer,
            version,
        })
    }

    /// Load with fallbacks for development mode ONLY.
    ///
    /// In dev mode, provides sensible defaults. In prod/staging, delegates to `load()`.
    pub fn load_with_dev_fallbacks(platform_env: &PlatformEnv) -> Result<Self> {
        if *platform_env == PlatformEnv::Dev {
            Ok(Self {
                name: env::var("AGENT_NAME").unwrap_or_else(|_| "llm-research-lab-dev".to_string()),
                domain: env::var("AGENT_DOMAIN").unwrap_or_else(|_| "research".to_string()),
                phase: env::var("AGENT_PHASE").unwrap_or_else(|_| "phase7".to_string()),
                layer: env::var("AGENT_LAYER").unwrap_or_else(|_| "layer2".to_string()),
                version: env::var("AGENT_VERSION").unwrap_or_else(|_| env!("CARGO_PKG_VERSION").to_string()),
            })
        } else {
            Self::load(platform_env)
        }
    }
}

impl Config {
    /// Load configuration from environment variables.
    ///
    /// # Phase 7 Layer 2 Hardening
    ///
    /// This function enforces FAIL-FAST behavior:
    /// - In prod/staging: ALL required vars must be set, NO FALLBACKS
    /// - In dev: Sensible defaults are allowed for local development
    ///
    /// Required environment variables (prod/staging):
    /// - RUVECTOR_SERVICE_URL: RuVector service endpoint
    /// - RUVECTOR_API_KEY: Authentication token (use Secret Manager)
    /// - LLM_OBSERVATORY_ENDPOINT: Telemetry endpoint
    /// - TELEMETRY_ENDPOINT: Telemetry API endpoint
    /// - AGENT_NAME, AGENT_DOMAIN, AGENT_PHASE, AGENT_LAYER, AGENT_VERSION
    pub fn load() -> Result<Self> {
        // Determine environment first (this affects validation strictness)
        let platform_env = match env::var("PLATFORM_ENV")
            .unwrap_or_else(|_| "dev".to_string())
            .as_str()
        {
            "prod" => PlatformEnv::Prod,
            "staging" => PlatformEnv::Staging,
            _ => PlatformEnv::Dev,
        };

        let is_production = platform_env != PlatformEnv::Dev;

        // RUVECTOR_SERVICE_URL - REQUIRED in prod, fallback in dev
        let ruvector_service_url = if is_production {
            env::var("RUVECTOR_SERVICE_URL")
                .map_err(|_| anyhow!(
                    "RUVECTOR_SERVICE_URL environment variable is required in {} mode. ABORTING STARTUP.",
                    if platform_env == PlatformEnv::Prod { "prod" } else { "staging" }
                ))?
        } else {
            env::var("RUVECTOR_SERVICE_URL")
                .unwrap_or_else(|_| "http://localhost:8081".to_string())
        };

        // RUVECTOR_API_KEY - REQUIRED in prod, optional in dev
        let ruvector_api_key = if is_production {
            env::var("RUVECTOR_API_KEY")
                .map_err(|_| anyhow!(
                    "RUVECTOR_API_KEY environment variable is required in {} mode (use Secret Manager). ABORTING STARTUP.",
                    if platform_env == PlatformEnv::Prod { "prod" } else { "staging" }
                ))?
        } else {
            env::var("RUVECTOR_API_KEY")
                .unwrap_or_else(|_| "dev-api-key-not-for-production".to_string())
        };

        // LLM_OBSERVATORY_ENDPOINT - REQUIRED in prod, fallback in dev
        let llm_observatory_endpoint = if is_production {
            env::var("LLM_OBSERVATORY_ENDPOINT")
                .map_err(|_| anyhow!(
                    "LLM_OBSERVATORY_ENDPOINT environment variable is required in {} mode. ABORTING STARTUP.",
                    if platform_env == PlatformEnv::Prod { "prod" } else { "staging" }
                ))?
        } else {
            env::var("LLM_OBSERVATORY_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:8082".to_string())
        };

        // TELEMETRY_ENDPOINT - REQUIRED in prod, fallback in dev
        let telemetry_endpoint = if is_production {
            env::var("TELEMETRY_ENDPOINT")
                .map_err(|_| anyhow!(
                    "TELEMETRY_ENDPOINT environment variable is required in {} mode. ABORTING STARTUP.",
                    if platform_env == PlatformEnv::Prod { "prod" } else { "staging" }
                ))?
        } else {
            env::var("TELEMETRY_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:8082/api/v1/telemetry".to_string())
        };

        // Load agent identity (uses its own validation based on platform_env)
        let agent_identity = if is_production {
            Some(AgentIdentity::load(&platform_env)?)
        } else {
            Some(AgentIdentity::load_with_dev_fallbacks(&platform_env)?)
        };

        let config = Self {
            port: env::var("LLM_RESEARCH_PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .unwrap_or(8080),
            platform_env,
            ruvector_service_url,
            ruvector_api_key,
            ruvector_timeout_secs: env::var("RUVECTOR_TIMEOUT_SECS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .unwrap_or(30),
            ruvector_max_retries: env::var("RUVECTOR_MAX_RETRIES")
                .unwrap_or_else(|_| "3".to_string())
                .parse()
                .unwrap_or(3),
            llm_observatory_endpoint,
            telemetry_endpoint,
            service_name: env::var("SERVICE_NAME")
                .unwrap_or_else(|_| default_service_name()),
            service_version: env::var("SERVICE_VERSION")
                .unwrap_or_else(|_| default_service_version()),
            log_level: env::var("LLM_RESEARCH_LOG_LEVEL")
                .unwrap_or_else(|_| default_log_level()),
            telemetry_stdout: env::var("TELEMETRY_STDOUT")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
            agent_identity,
        };

        // Log startup configuration with structured JSON
        config.log_startup_identity();

        Ok(config)
    }

    /// Log startup configuration with structured JSON including all identity fields.
    fn log_startup_identity(&self) {
        if let Some(ref identity) = self.agent_identity {
            // Use eprintln for startup logging to ensure it's visible
            // In production, this would go through tracing infrastructure
            let startup_log = serde_json::json!({
                "event": "config_loaded",
                "phase": identity.phase,
                "layer": identity.layer,
                "agent_name": identity.name,
                "agent_domain": identity.domain,
                "agent_version": identity.version,
                "platform_env": format!("{:?}", self.platform_env).to_lowercase(),
                "service_name": self.service_name,
                "service_version": self.service_version,
                "ruvector_url": self.ruvector_service_url,
                "observatory_url": self.llm_observatory_endpoint,
                "port": self.port,
            });

            eprintln!(
                "{{\"level\":\"INFO\",\"message\":\"Phase 7 Layer 2 configuration loaded\",\"config\":{}}}",
                startup_log
            );
        }
    }

    /// Validate that the configuration is complete for production deployment.
    ///
    /// Returns an error with a detailed message if any required configuration is missing.
    pub fn validate_for_production(&self) -> Result<()> {
        if self.platform_env == PlatformEnv::Dev {
            return Ok(()); // Skip strict validation in dev
        }

        // Validate RuVector URL is not localhost in production
        if self.ruvector_service_url.contains("localhost") {
            return Err(anyhow!(
                "RUVECTOR_SERVICE_URL cannot be localhost in production. Got: {}",
                self.ruvector_service_url
            ));
        }

        // Validate Observatory URL is not localhost in production
        if self.llm_observatory_endpoint.contains("localhost") {
            return Err(anyhow!(
                "LLM_OBSERVATORY_ENDPOINT cannot be localhost in production. Got: {}",
                self.llm_observatory_endpoint
            ));
        }

        // Validate agent identity is present
        if self.agent_identity.is_none() {
            return Err(anyhow!(
                "Agent identity is required in production. Set AGENT_NAME, AGENT_DOMAIN, AGENT_PHASE, AGENT_LAYER, AGENT_VERSION."
            ));
        }

        Ok(())
    }
}

impl Default for Config {
    /// Default configuration for DEVELOPMENT ONLY.
    ///
    /// WARNING: This should NEVER be used in production.
    /// Production deployments MUST use `Config::load()` which validates
    /// required environment variables.
    fn default() -> Self {
        Self {
            port: default_port(),
            platform_env: PlatformEnv::default(),
            ruvector_service_url: "http://localhost:8081".to_string(),
            ruvector_api_key: "dev-api-key-not-for-production".to_string(),
            ruvector_timeout_secs: default_ruvector_timeout(),
            ruvector_max_retries: default_ruvector_retries(),
            llm_observatory_endpoint: "http://localhost:8082".to_string(),
            telemetry_endpoint: "http://localhost:8082/api/v1/telemetry".to_string(),
            service_name: default_service_name(),
            service_version: default_service_version(),
            log_level: default_log_level(),
            telemetry_stdout: default_telemetry_stdout(),
            agent_identity: Some(AgentIdentity {
                name: "llm-research-lab-dev".to_string(),
                domain: "research".to_string(),
                phase: "phase7".to_string(),
                layer: "layer2".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.port, 8080);
        assert_eq!(config.platform_env, PlatformEnv::Dev);
    }
}
