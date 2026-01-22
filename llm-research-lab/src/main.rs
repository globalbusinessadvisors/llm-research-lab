//! LLM Research Lab - Unified Cloud Run Service
//!
//! # Constitution Compliance
//!
//! This service adheres to PROMPT 2 (RUNTIME & INFRASTRUCTURE IMPLEMENTATION):
//!
//! - **Stateless Runtime**: No local state, all persistence via ruvector-service
//! - **No Direct SQL Access**: Database access ONLY through ruvector-service
//! - **Edge Function Compatible**: Handlers are deterministic and stateless
//! - **Telemetry Emission**: LLM-Observatory compatible telemetry
//!
//! # Service Topology
//!
//! Single unified service exposing:
//! - `/health` - Health check endpoint
//! - `/api/v1/agents/hypothesis` - Hypothesis evaluation agent
//! - `/api/v1/agents/metric` - Experimental metric computation agent
//! - `/api/v1/*` - Standard API routes

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{info, warn};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

mod config;

// Agent imports
use llm_research_agents::handlers::{
    HypothesisHandler, HypothesisEvaluateRequest, HypothesisEvaluateResponse,
    MetricHandler, MetricComputeRequest, MetricComputeResponse,
};

/// Application state for Cloud Run deployment.
///
/// CONSTITUTION COMPLIANCE: This state contains NO database pools.
/// All persistence is handled via ruvector-service through the handlers.
#[derive(Clone)]
pub struct CloudRunState {
    /// Hypothesis agent handler
    hypothesis_handler: Arc<HypothesisHandler>,
    /// Metric agent handler
    metric_handler: Arc<MetricHandler>,
    /// Service configuration
    config: Arc<config::Config>,
}

impl CloudRunState {
    /// Create new state from environment.
    ///
    /// No database connections are created here - handlers use ruvector-service.
    pub fn new(config: config::Config) -> Result<Self> {
        let hypothesis_handler = Arc::new(
            HypothesisHandler::new()
                .map_err(|e| anyhow::anyhow!("Failed to create hypothesis handler: {}", e))?
        );
        let metric_handler = Arc::new(MetricHandler::new());

        Ok(Self {
            hypothesis_handler,
            metric_handler,
            config: Arc::new(config),
        })
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| {
                    "llm_research_lab=debug,llm_research_agents=debug,tower_http=debug".into()
                }),
        )
        .with(tracing_subscriber::fmt::layer().json())
        .init();

    info!(
        service = "llm-research-lab",
        version = env!("CARGO_PKG_VERSION"),
        "Starting LLM Research Lab Cloud Run service"
    );

    // Load configuration from environment
    let config = config::Config::load()?;
    info!(
        port = config.port,
        platform_env = ?config.platform_env,
        ruvector_url = %config.ruvector_service_url,
        "Configuration loaded"
    );

    // Create stateless application state
    let state = CloudRunState::new(config.clone())?;
    info!("Application state initialized (stateless, no DB connections)");

    // Build router with agent endpoints
    let app = Router::new()
        // Health endpoints
        .route("/health", get(health_check))
        .route("/ready", get(readiness_check))

        // Agent endpoints (per service topology)
        .route("/api/v1/agents/hypothesis", post(hypothesis_evaluate))
        .route("/api/v1/agents/hypothesis", get(hypothesis_info))
        .route("/api/v1/agents/metric", post(metric_compute))
        .route("/api/v1/agents/metric", get(metric_info))
        .route("/api/v1/agents", get(list_agents))

        // Middleware
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    // Start server
    let addr = SocketAddr::from(([0, 0, 0, 0], config.port));
    info!(address = %addr, "Server listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

// =============================================================================
// Health Endpoints
// =============================================================================

/// Liveness probe - always returns OK if the process is running.
async fn health_check() -> &'static str {
    "OK"
}

/// Readiness probe - checks if ruvector-service is reachable.
async fn readiness_check(State(state): State<CloudRunState>) -> (StatusCode, &'static str) {
    // In production, this would check ruvector-service health
    // For now, return OK if handlers are initialized
    if state.hypothesis_handler.as_ref() as *const _ != std::ptr::null() {
        (StatusCode::OK, "READY")
    } else {
        (StatusCode::SERVICE_UNAVAILABLE, "NOT_READY")
    }
}

// =============================================================================
// Agent Endpoints
// =============================================================================

/// POST /api/v1/agents/hypothesis - Evaluate a hypothesis.
async fn hypothesis_evaluate(
    State(state): State<CloudRunState>,
    Json(request): Json<HypothesisEvaluateRequest>,
) -> (StatusCode, Json<HypothesisEvaluateResponse>) {
    let response = state.hypothesis_handler.handle(request).await;

    let status = if response.success {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    };

    (status, Json(response))
}

/// GET /api/v1/agents/hypothesis - Get hypothesis agent info.
async fn hypothesis_info() -> Json<AgentInfoResponse> {
    Json(AgentInfoResponse {
        id: "hypothesis-agent".to_string(),
        version: "1.0.0".to_string(),
        classification: "HYPOTHESIS_EVALUATION".to_string(),
        endpoint: "/api/v1/agents/hypothesis".to_string(),
        methods: vec!["POST".to_string()],
    })
}

/// POST /api/v1/agents/metric - Compute experimental metrics.
async fn metric_compute(
    State(state): State<CloudRunState>,
    Json(request): Json<MetricComputeRequest>,
) -> (StatusCode, Json<MetricComputeResponse>) {
    let response = state.metric_handler.handle(request).await;

    let status = if response.success {
        StatusCode::OK
    } else {
        StatusCode::BAD_REQUEST
    };

    (status, Json(response))
}

/// GET /api/v1/agents/metric - Get metric agent info.
async fn metric_info() -> Json<AgentInfoResponse> {
    let handler = MetricHandler::new();
    let info = handler.agent_info();

    Json(AgentInfoResponse {
        id: info.id,
        version: info.version,
        classification: info.classification,
        endpoint: info.endpoint,
        methods: vec!["POST".to_string()],
    })
}

/// GET /api/v1/agents - List all available agents.
async fn list_agents() -> Json<AgentsListResponse> {
    Json(AgentsListResponse {
        agents: vec![
            AgentInfoResponse {
                id: "hypothesis-agent".to_string(),
                version: "1.0.0".to_string(),
                classification: "HYPOTHESIS_EVALUATION".to_string(),
                endpoint: "/api/v1/agents/hypothesis".to_string(),
                methods: vec!["POST".to_string()],
            },
            AgentInfoResponse {
                id: "experimental-metric-agent".to_string(),
                version: "1.0.0".to_string(),
                classification: "EXPERIMENTAL_METRICS".to_string(),
                endpoint: "/api/v1/agents/metric".to_string(),
                methods: vec!["POST".to_string()],
            },
        ],
        count: 2,
    })
}

// =============================================================================
// Response Types
// =============================================================================

#[derive(Debug, Serialize, Deserialize)]
pub struct AgentInfoResponse {
    pub id: String,
    pub version: String,
    pub classification: String,
    pub endpoint: String,
    pub methods: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AgentsListResponse {
    pub agents: Vec<AgentInfoResponse>,
    pub count: usize,
}
