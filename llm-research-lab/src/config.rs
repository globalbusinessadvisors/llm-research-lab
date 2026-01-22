//! Configuration for LLM Research Lab Cloud Run Service
//!
//! # Constitution Compliance
//!
//! Per PROMPT 2, this service DOES NOT use direct database connections.
//! All persistence is via ruvector-service.
//!
//! Required environment variables:
//! - RUVECTOR_SERVICE_URL: RuVector service endpoint
//! - LLM_OBSERVATORY_ENDPOINT: Telemetry endpoint
//! - PLATFORM_ENV: dev | staging | prod

use anyhow::Result;
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
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    /// HTTP server port
    #[serde(default = "default_port")]
    pub port: u16,

    /// Platform environment
    #[serde(default)]
    pub platform_env: PlatformEnv,

    /// RuVector service URL (REQUIRED for persistence)
    pub ruvector_service_url: String,

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

impl Config {
    /// Load configuration from environment variables.
    ///
    /// Required environment variables:
    /// - RUVECTOR_SERVICE_URL
    /// - LLM_OBSERVATORY_ENDPOINT
    /// - TELEMETRY_ENDPOINT
    pub fn load() -> Result<Self> {
        Ok(Self {
            port: env::var("LLM_RESEARCH_PORT")
                .unwrap_or_else(|_| "8080".to_string())
                .parse()
                .unwrap_or(8080),
            platform_env: match env::var("PLATFORM_ENV")
                .unwrap_or_else(|_| "dev".to_string())
                .as_str()
            {
                "prod" => PlatformEnv::Prod,
                "staging" => PlatformEnv::Staging,
                _ => PlatformEnv::Dev,
            },
            ruvector_service_url: env::var("RUVECTOR_SERVICE_URL")
                .unwrap_or_else(|_| "http://localhost:8081".to_string()),
            ruvector_timeout_secs: env::var("RUVECTOR_TIMEOUT_SECS")
                .unwrap_or_else(|_| "30".to_string())
                .parse()
                .unwrap_or(30),
            ruvector_max_retries: env::var("RUVECTOR_MAX_RETRIES")
                .unwrap_or_else(|_| "3".to_string())
                .parse()
                .unwrap_or(3),
            llm_observatory_endpoint: env::var("LLM_OBSERVATORY_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:8082".to_string()),
            telemetry_endpoint: env::var("TELEMETRY_ENDPOINT")
                .unwrap_or_else(|_| "http://localhost:8082/api/v1/telemetry".to_string()),
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
        })
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            port: default_port(),
            platform_env: PlatformEnv::default(),
            ruvector_service_url: "http://localhost:8081".to_string(),
            ruvector_timeout_secs: default_ruvector_timeout(),
            ruvector_max_retries: default_ruvector_retries(),
            llm_observatory_endpoint: "http://localhost:8082".to_string(),
            telemetry_endpoint: "http://localhost:8082/api/v1/telemetry".to_string(),
            service_name: default_service_name(),
            service_version: default_service_version(),
            log_level: default_log_level(),
            telemetry_stdout: default_telemetry_stdout(),
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
