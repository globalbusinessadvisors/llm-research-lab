//! RuVector Service Client
//!
//! Client for persisting DecisionEvents and research artifacts to ruvector-service.
//!
//! # Constitution Compliance
//!
//! Per PROMPT 0 (LLM-RESEARCH-LAB AGENT INFRASTRUCTURE CONSTITUTION):
//!
//! - LLM-Research-Lab does NOT own persistence
//! - ALL data is persisted via ruvector-service
//! - LLM-Research-Lab NEVER connects directly to Google SQL
//! - LLM-Research-Lab NEVER executes SQL
//!
//! This client is the ONLY authorized mechanism for data persistence.

use async_trait::async_trait;
use reqwest::{Client, StatusCode};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::time::Duration;
use thiserror::Error;
use tracing::{debug, error, info, instrument, warn};
use url::Url;
use uuid::Uuid;

use crate::contracts::DecisionEvent;

/// RuVector client configuration.
#[derive(Debug, Clone)]
pub struct RuVectorConfig {
    /// Base URL of ruvector-service
    pub base_url: Url,

    /// Authentication token
    pub auth_token: Option<String>,

    /// Request timeout
    pub timeout: Duration,

    /// Maximum retry attempts
    pub max_retries: u32,

    /// Retry backoff base (milliseconds)
    pub retry_backoff_ms: u64,
}

impl Default for RuVectorConfig {
    fn default() -> Self {
        Self {
            base_url: Url::parse("http://localhost:8080").expect("Valid default URL"),
            auth_token: None,
            timeout: Duration::from_secs(30),
            max_retries: 3,
            retry_backoff_ms: 100,
        }
    }
}

impl RuVectorConfig {
    /// Create config from environment variables.
    ///
    /// # Phase 7 Layer 2 Hardening
    ///
    /// This function FAILS FAST if required environment variables are missing.
    /// NO FALLBACKS - missing config is a fatal startup error.
    ///
    /// Required environment variables:
    /// - RUVECTOR_SERVICE_URL: Base URL of ruvector-service (REQUIRED)
    /// - RUVECTOR_API_KEY: Authentication token from Secret Manager (REQUIRED)
    ///
    /// Optional environment variables:
    /// - RUVECTOR_TIMEOUT_SECS: Request timeout (default: 30)
    pub fn from_env() -> Result<Self, RuVectorError> {
        // REQUIRED env vars - FAIL if missing (Phase 7 Layer 2 hardening)
        let base_url = std::env::var("RUVECTOR_SERVICE_URL")
            .map_err(|_| RuVectorError::Configuration(
                "RUVECTOR_SERVICE_URL environment variable is required. ABORTING STARTUP.".to_string()
            ))?;

        let auth_token = std::env::var("RUVECTOR_API_KEY")
            .map_err(|_| RuVectorError::Configuration(
                "RUVECTOR_API_KEY environment variable is required (use Secret Manager). ABORTING STARTUP.".to_string()
            ))?;

        // Optional with defaults
        let timeout_secs = std::env::var("RUVECTOR_TIMEOUT_SECS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(30);

        let parsed_url = Url::parse(&base_url).map_err(|e| {
            RuVectorError::Configuration(format!(
                "RUVECTOR_SERVICE_URL is not a valid URL: {}. ABORTING STARTUP.",
                e
            ))
        })?;

        Ok(Self {
            base_url: parsed_url,
            auth_token: Some(auth_token), // Now required, not optional
            timeout: Duration::from_secs(timeout_secs),
            ..Default::default()
        })
    }
}

/// Errors from RuVector client operations.
#[derive(Debug, Error)]
pub enum RuVectorError {
    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Connection error: {0}")]
    Connection(String),

    #[error("Request error: {0}")]
    Request(String),

    #[error("Response error: status={status}, message={message}")]
    Response { status: u16, message: String },

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Authentication error: {0}")]
    Authentication(String),

    #[error("Timeout error")]
    Timeout,

    #[error("Retry exhausted after {attempts} attempts: {last_error}")]
    RetryExhausted { attempts: u32, last_error: String },
}

impl From<reqwest::Error> for RuVectorError {
    fn from(err: reqwest::Error) -> Self {
        if err.is_timeout() {
            RuVectorError::Timeout
        } else if err.is_connect() {
            RuVectorError::Connection(err.to_string())
        } else {
            RuVectorError::Request(err.to_string())
        }
    }
}

impl From<serde_json::Error> for RuVectorError {
    fn from(err: serde_json::Error) -> Self {
        RuVectorError::Serialization(err.to_string())
    }
}

/// Response from ruvector-service.
#[derive(Debug, Serialize, Deserialize)]
pub struct RuVectorResponse<T> {
    /// Indicates success
    pub success: bool,

    /// Response data (if success)
    pub data: Option<T>,

    /// Error details (if failure)
    pub error: Option<RuVectorErrorResponse>,

    /// Request ID for correlation
    pub request_id: Option<Uuid>,
}

/// Error response from ruvector-service.
#[derive(Debug, Serialize, Deserialize)]
pub struct RuVectorErrorResponse {
    pub code: String,
    pub message: String,
    pub details: Option<serde_json::Value>,
}

/// Persisted DecisionEvent confirmation.
#[derive(Debug, Serialize, Deserialize)]
pub struct PersistedDecisionEvent {
    /// Event ID
    pub id: Uuid,

    /// Storage reference
    pub storage_ref: String,

    /// Timestamp of persistence
    pub persisted_at: chrono::DateTime<chrono::Utc>,
}

/// Trait for RuVector persistence operations.
#[async_trait]
pub trait RuVectorPersistence: Send + Sync {
    /// Persist a DecisionEvent.
    ///
    /// This is the primary persistence method for all LLM-Research-Lab agents.
    /// Every agent invocation MUST call this exactly once.
    async fn persist_decision_event(&self, event: DecisionEvent) -> Result<PersistedDecisionEvent, RuVectorError>;

    /// Retrieve a DecisionEvent by ID.
    async fn get_decision_event(&self, id: Uuid) -> Result<Option<DecisionEvent>, RuVectorError>;

    /// List DecisionEvents by agent ID.
    async fn list_decision_events_by_agent(
        &self,
        agent_id: &str,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<DecisionEvent>, RuVectorError>;

    /// Health check for ruvector-service.
    async fn health_check(&self) -> Result<bool, RuVectorError>;
}

/// HTTP client implementation for ruvector-service.
#[derive(Clone)]
pub struct RuVectorClient {
    client: Client,
    config: RuVectorConfig,
}

impl RuVectorClient {
    /// Create a new RuVector client.
    pub fn new(config: RuVectorConfig) -> Result<Self, RuVectorError> {
        let client = Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|e| RuVectorError::Configuration(e.to_string()))?;

        Ok(Self { client, config })
    }

    /// Create a client from environment variables.
    pub fn from_env() -> Result<Self, RuVectorError> {
        let config = RuVectorConfig::from_env()?;
        Self::new(config)
    }

    /// Initialize with mandatory health check - CRASHES if Ruvector unavailable.
    ///
    /// # Phase 7 Layer 2 Hardening
    ///
    /// This is the REQUIRED initialization method for production deployments.
    /// The service will FAIL TO START if Ruvector is unavailable.
    /// NO DEGRADED MODE - Ruvector is mandatory for all persistence operations.
    ///
    /// # Errors
    ///
    /// Returns `RuVectorError::Connection` if:
    /// - Ruvector service is unreachable
    /// - Health check returns non-success status
    /// - Network timeout occurs
    pub async fn init_with_health_check(config: RuVectorConfig) -> Result<Self, RuVectorError> {
        let client = Self::new(config)?;

        // Mandatory health check - FAIL FAST (Phase 7 Layer 2 hardening)
        let healthy = client.health_check().await?;
        if !healthy {
            return Err(RuVectorError::Connection(
                "Ruvector health check failed - service unavailable. ABORTING STARTUP.".to_string()
            ));
        }

        // Structured JSON startup log with identity fields
        info!(
            ruvector_url = %client.config.base_url,
            auth_configured = client.config.auth_token.is_some(),
            timeout_secs = client.config.timeout.as_secs(),
            max_retries = client.config.max_retries,
            phase = "phase7",
            layer = "layer2",
            component = "ruvector-client",
            "Ruvector client initialized successfully - health check passed"
        );

        Ok(client)
    }

    /// Initialize from environment with mandatory health check.
    ///
    /// Combines `from_env()` and `init_with_health_check()` for convenience.
    /// This is the RECOMMENDED initialization method for production.
    pub async fn init_from_env_with_health_check() -> Result<Self, RuVectorError> {
        let config = RuVectorConfig::from_env()?;
        Self::init_with_health_check(config).await
    }

    /// Build a URL for an API endpoint.
    fn build_url(&self, path: &str) -> Result<Url, RuVectorError> {
        self.config
            .base_url
            .join(path)
            .map_err(|e| RuVectorError::Configuration(e.to_string()))
    }

    /// Execute a request with retry logic.
    #[instrument(skip(self, request_builder), fields(path = %path))]
    async fn execute_with_retry<T: DeserializeOwned>(
        &self,
        path: &str,
        request_builder: impl Fn() -> reqwest::RequestBuilder,
    ) -> Result<RuVectorResponse<T>, RuVectorError> {
        let mut attempts = 0;
        let mut last_error = String::new();

        while attempts < self.config.max_retries {
            attempts += 1;

            let request = request_builder();
            let request = if let Some(ref token) = self.config.auth_token {
                request.bearer_auth(token)
            } else {
                request
            };

            match request.send().await {
                Ok(response) => {
                    let status = response.status();

                    if status.is_success() {
                        let body = response.text().await?;
                        debug!("Response body: {}", body);

                        let parsed: RuVectorResponse<T> = serde_json::from_str(&body)?;
                        return Ok(parsed);
                    }

                    // Handle specific error codes
                    match status {
                        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                            return Err(RuVectorError::Authentication(
                                "Invalid or missing authentication".to_string(),
                            ));
                        }
                        StatusCode::TOO_MANY_REQUESTS => {
                            warn!("Rate limited, retrying after backoff");
                            last_error = "Rate limited".to_string();
                        }
                        StatusCode::SERVICE_UNAVAILABLE | StatusCode::GATEWAY_TIMEOUT => {
                            warn!("Service unavailable, retrying");
                            last_error = format!("Service unavailable: {}", status);
                        }
                        _ => {
                            let body = response.text().await.unwrap_or_default();
                            return Err(RuVectorError::Response {
                                status: status.as_u16(),
                                message: body,
                            });
                        }
                    }
                }
                Err(e) => {
                    if e.is_timeout() {
                        warn!("Request timeout, attempt {}/{}", attempts, self.config.max_retries);
                        last_error = "Timeout".to_string();
                    } else if e.is_connect() {
                        warn!("Connection error, attempt {}/{}", attempts, self.config.max_retries);
                        last_error = e.to_string();
                    } else {
                        return Err(e.into());
                    }
                }
            }

            // Exponential backoff
            if attempts < self.config.max_retries {
                let backoff = Duration::from_millis(self.config.retry_backoff_ms * 2u64.pow(attempts - 1));
                tokio::time::sleep(backoff).await;
            }
        }

        Err(RuVectorError::RetryExhausted {
            attempts,
            last_error,
        })
    }
}

#[async_trait]
impl RuVectorPersistence for RuVectorClient {
    #[instrument(skip(self, event), fields(event_id = %event.id, agent_id = %event.agent_id))]
    async fn persist_decision_event(&self, event: DecisionEvent) -> Result<PersistedDecisionEvent, RuVectorError> {
        let url = self.build_url("/api/v1/decision-events")?;

        info!("Persisting DecisionEvent to ruvector-service");

        let response: RuVectorResponse<PersistedDecisionEvent> = self
            .execute_with_retry("/api/v1/decision-events", || {
                self.client.post(url.clone()).json(&event)
            })
            .await?;

        if response.success {
            let persisted = response.data.ok_or_else(|| {
                RuVectorError::Response {
                    status: 200,
                    message: "Success response without data".to_string(),
                }
            })?;

            info!(
                storage_ref = %persisted.storage_ref,
                "DecisionEvent persisted successfully"
            );

            Ok(persisted)
        } else {
            let err = response.error.unwrap_or(RuVectorErrorResponse {
                code: "UNKNOWN".to_string(),
                message: "Unknown error".to_string(),
                details: None,
            });

            error!(code = %err.code, message = %err.message, "Failed to persist DecisionEvent");

            Err(RuVectorError::Response {
                status: 400,
                message: err.message,
            })
        }
    }

    #[instrument(skip(self), fields(event_id = %id))]
    async fn get_decision_event(&self, id: Uuid) -> Result<Option<DecisionEvent>, RuVectorError> {
        let path = format!("/api/v1/decision-events/{}", id);
        let url = self.build_url(&path)?;

        let response: RuVectorResponse<DecisionEvent> = self
            .execute_with_retry(&path, || self.client.get(url.clone()))
            .await?;

        Ok(response.data)
    }

    #[instrument(skip(self), fields(agent_id = %agent_id))]
    async fn list_decision_events_by_agent(
        &self,
        agent_id: &str,
        limit: u32,
        offset: u32,
    ) -> Result<Vec<DecisionEvent>, RuVectorError> {
        let path = format!(
            "/api/v1/decision-events?agent_id={}&limit={}&offset={}",
            agent_id, limit, offset
        );
        let url = self.build_url(&path)?;

        let response: RuVectorResponse<Vec<DecisionEvent>> = self
            .execute_with_retry(&path, || self.client.get(url.clone()))
            .await?;

        Ok(response.data.unwrap_or_default())
    }

    /// Health check for ruvector-service.
    ///
    /// # Phase 7 Layer 2 Hardening
    ///
    /// This method now PROPAGATES errors instead of swallowing them.
    /// Errors are FATAL during startup - no degraded mode allowed.
    #[instrument(skip(self))]
    async fn health_check(&self) -> Result<bool, RuVectorError> {
        let url = self.build_url("/health")?;

        let response = self.client.get(url).send().await
            .map_err(|e| {
                error!(
                    error = %e,
                    ruvector_url = %self.config.base_url,
                    phase = "phase7",
                    layer = "layer2",
                    "Ruvector health check connection failed - FATAL"
                );
                RuVectorError::Connection(format!(
                    "Health check failed - cannot connect to Ruvector at {}: {}",
                    self.config.base_url, e
                ))
            })?;

        if !response.status().is_success() {
            let status = response.status();
            error!(
                status = %status,
                ruvector_url = %self.config.base_url,
                phase = "phase7",
                layer = "layer2",
                "Ruvector health check returned non-success status - FATAL"
            );
            return Err(RuVectorError::Connection(format!(
                "Health check returned non-success status: {}. Ruvector service unhealthy.",
                status
            )));
        }

        debug!(
            ruvector_url = %self.config.base_url,
            "Ruvector health check passed"
        );

        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = RuVectorConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_build_url() {
        let config = RuVectorConfig {
            base_url: Url::parse("http://example.com").unwrap(),
            ..Default::default()
        };
        let client = RuVectorClient::new(config).unwrap();

        let url = client.build_url("/api/v1/test").unwrap();
        assert_eq!(url.as_str(), "http://example.com/api/v1/test");
    }
}
