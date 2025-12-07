pub mod handlers;
pub mod middleware;
pub mod dto;
pub mod error;
pub mod security;
pub mod observability;
pub mod performance;
pub mod resilience;
pub mod reliability;
pub mod response;

// =============================================================================
// LLM-Dev-Ops Infra Re-exports (Phase 2B)
// =============================================================================
// These re-exports provide access to standardized Infra modules.
// Local implementations in observability/, performance/, resilience/, etc.
// are being migrated to use these Infra modules for ecosystem consistency.
//
// Migration Status:
// - infra_core: Available ✓
// - infra_config: Available ✓
// - infra_logging: Available (replaces observability/logging.rs)
// - infra_tracing: Available (replaces observability/tracing.rs)
// - infra_metrics: Available (replaces observability/metrics.rs)
// - infra_cache: Available (replaces performance/cache.rs)
// - infra_resilience: Available (replaces resilience/ modules)
// - infra_health: Available (replaces observability/health.rs)
// - infra_error: Available (supplements error.rs)
// =============================================================================
pub mod infra {
    pub use infra_core as core;
    pub use infra_config as config;
    pub use infra_logging as logging;
    pub use infra_tracing as tracing;
    pub use infra_metrics as metrics;
    pub use infra_cache as cache;
    pub use infra_resilience as resilience;
    pub use infra_health as health;
    pub use infra_error as error;
}

use axum::{
    routing::{get, post, put, delete},
    Router,
};
use sqlx::PgPool;
use aws_sdk_s3::Client as S3Client;
use tower_http::cors::{CorsLayer, Any};
use tower_http::trace::TraceLayer;
use std::time::Duration;

pub use error::ApiError;
pub use dto::*;
pub use security::{
    // JWT Authentication
    AuthError, AuthResult, Claims, JwtConfig, JwtService, RefreshClaims, TokenPair, TokenType,
    // Role-Based Access Control
    Role, Permission, RolePermissions, ResourceOwnership, PermissionGuard,
    // Rate Limiting
    RateLimitConfig, RateLimitError, RateLimitInfo, RateLimitKey, RateLimitLayer,
    RateLimiter, rate_limit_middleware,
    // Audit Logging
    AuditAction, AuditActor, AuditError, AuditEvent, AuditEventType, AuditLogger,
    AuditOutcome, AuditResource, AuditResult, AuditWriter, CompositeAuditWriter,
    DatabaseAuditWriter, FileAuditWriter, TracingAuditWriter, AuditMiddlewareState,
    AuditLogFilter, AuditLogQuery, AuditStatistics,
    audit_middleware, AuditMiddlewareError,
    // API Key Authentication
    ApiKey, ApiKeyService, ApiKeyUser, ApiScope,
    ExperimentPermission, ModelPermission, DatasetPermission, MetricPermission,
    RateLimitTier,
    api_key_auth_middleware, optional_api_key_auth_middleware,
    get_api_key_user, require_role, require_any_role, require_scope_permission,
    // Request Validation
    ValidatedJson, ValidationRejection, FieldError,
    validate_identifier, validate_slug, validate_json_schema, validate_s3_path,
    validate_safe_filename, validate_uuid_string, validate_no_script_tags,
    sanitize,
    // Security Headers
    SecurityHeadersConfig, ContentSecurityPolicy, FrameOptions, ReferrerPolicy,
    CorsConfig, AllowedOrigins,
    security_headers_middleware, security_headers_with_config, create_security_headers_layer,
};
pub use observability::{
    // Distributed Tracing
    create_span, current_span_id, current_trace_id, init_tracing, record_error, record_event,
    shutdown_tracing, tracing_middleware, DbSpan, SpanBuilder, TraceContextPropagation,
    TracingConfig, TracingConfigBuilder, TracingError, TracingResult,
    // Metrics
    init_metrics, metrics_handler, BusinessMetrics, DatabaseMetrics, DurationGuard,
    HttpMetrics, MetricsConfig, MetricsError, MetricsLayer, MetricsRecorder, SystemMetrics,
    increment_counter, observe_duration, set_gauge,
    // Logging
    LogConfig, LogFormat, LogRotationConfig,
    LogContext, current_context, with_context,
    SensitiveDataRedactor,
    RequestLoggingState, request_logging_middleware, create_request_logging_middleware,
    init_logging, init_default_logging,
    REQUEST_ID_HEADER, SENSITIVE_HEADERS, SENSITIVE_PATTERNS,
    // Health checks
    HealthStatus, ComponentHealth, OverallHealth, HealthCheckConfig,
    HealthCheck, PostgresHealthCheck, ClickHouseHealthCheck, S3HealthCheck,
    HealthCheckRegistry, HealthCheckState,
    liveness_handler, readiness_handler, health_handler,
};
pub use performance::{
    // Connection Pool Management
    PoolConfig, PoolConfigBuilder, PoolError, PoolHealth, PoolMonitor, PoolStatistics,
    create_clickhouse_pool, create_postgres_pool, get_postgres_pool_stats,
    // Query Result Caching
    CacheConfig, CacheConfigBuilder, CacheError, CacheKey, CacheService, CacheStatistics,
    CachedResult, EvictionPolicy, InMemoryCache, cached,
};
pub use resilience::{
    // Circuit Breaker
    CircuitBreaker, CircuitBreakerConfig, CircuitBreakerError, CircuitState,
    // Retry
    retry, retry_with_context, ConstantBackoff, ExponentialBackoff, JitterStrategy,
    LinearBackoff, RetryConfig, RetryError, RetryPolicy,
    // Timeout
    timeout_after, with_timeout, TimeoutConfig, TimeoutError, TimeoutLayer,
    // Graceful Shutdown
    ConnectionGuard, GracefulShutdown, ShutdownCoordinator, ShutdownError, ShutdownSignal,
};
pub use reliability::{
    // Database Backup
    BackupConfig, BackupError, BackupMetadata, BackupResult, BackupService, BackupStatus,
    BackupType, PostgresBackupService, S3BackupStorage,
    // Bulkhead Pattern
    Bulkhead, BulkheadConfig, BulkheadError, BulkheadMetrics, BulkheadRegistry,
    with_bulkhead,
    // Health Check Extensions
    AlertHandler, AlertSeverity, DeepHealthCheck, DependencyHealth, HealthAggregator,
    HealthAlert, HealthCheckScheduler, HealthHistory, HealthHistoryEntry, LoggingAlertHandler,
    // Load Shedding
    create_load_shedding_layer, load_shedding_middleware, LoadLevel, LoadShedder,
    LoadSheddingConfig, LoadSheddingError, LoadSheddingMiddlewareState, LoadSheddingStats,
    ResourceMetrics,
};
pub use response::{
    // Compression
    CompressionAlgorithm, CompressionConfig, CompressionConfigBuilder, CompressionLayer,
    CompressionMiddleware, ContentTypePredicate, compression_middleware,
    create_compression_layer, parse_accept_encoding,
    // Pagination
    CursorPagination, CursorPaginatedResponse, FieldSelection, PageInfo, PaginatedResponse,
    Paginator, PaginationError, PaginationLinks, PaginationParams, DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE, MIN_PAGE_SIZE,
    // Query Optimization
    FilterOperator, FilterSpec, JoinClause, JoinType,
    OptimizationHint, QueryBuilder, QueryOptimizer, SlowQueryConfig, SlowQueryLogger,
    SortDirection, SortSpec,
    // Database Indexing
    CommonIndexPatterns, IndexAnalyzer, IndexDefinition, IndexRecommendation, IndexStrategy,
    SizeImpact, TableSchema, generate_migration,
};

#[derive(Clone)]
pub struct AppState {
    pub db_pool: PgPool,
    pub s3_client: S3Client,
    pub s3_bucket: String,
}

impl AppState {
    pub fn new(db_pool: PgPool, s3_client: S3Client, s3_bucket: String) -> Self {
        Self {
            db_pool,
            s3_client,
            s3_bucket,
        }
    }
}

pub fn routes(state: AppState) -> Router {
    Router::new()
        // Experiment routes
        .route("/experiments", post(handlers::experiments::create))
        .route("/experiments", get(handlers::experiments::list))
        .route("/experiments/:id", get(handlers::experiments::get))
        .route("/experiments/:id", put(handlers::experiments::update))
        .route("/experiments/:id", delete(handlers::experiments::delete))
        .route("/experiments/:id/start", post(handlers::experiments::start))
        .route("/experiments/:id/runs", post(handlers::experiments::create_run))
        .route("/experiments/:id/runs", get(handlers::experiments::list_runs))
        .route("/experiments/:id/runs/:run_id/complete", post(handlers::experiments::complete_run))
        .route("/experiments/:id/runs/:run_id/fail", post(handlers::experiments::fail_run))

        // Model routes
        .route("/models", post(handlers::models::create))
        .route("/models", get(handlers::models::list))
        .route("/models/:id", get(handlers::models::get))
        .route("/models/:id", put(handlers::models::update))
        .route("/models/:id", delete(handlers::models::delete))
        .route("/models/providers", get(handlers::models::list_providers))

        // Dataset routes
        .route("/datasets", post(handlers::datasets::create))
        .route("/datasets", get(handlers::datasets::list))
        .route("/datasets/:id", get(handlers::datasets::get))
        .route("/datasets/:id", put(handlers::datasets::update))
        .route("/datasets/:id", delete(handlers::datasets::delete))
        .route("/datasets/:id/versions", post(handlers::datasets::create_version))
        .route("/datasets/:id/versions", get(handlers::datasets::list_versions))
        .route("/datasets/:id/upload", post(handlers::datasets::upload))
        .route("/datasets/:id/download", get(handlers::datasets::download))

        // Prompt template routes
        .route("/prompts", post(handlers::prompts::create))
        .route("/prompts", get(handlers::prompts::list))
        .route("/prompts/:id", get(handlers::prompts::get))
        .route("/prompts/:id", put(handlers::prompts::update))
        .route("/prompts/:id", delete(handlers::prompts::delete))

        // Evaluation routes
        .route("/evaluations", post(handlers::evaluations::create))
        .route("/evaluations", get(handlers::evaluations::list))
        .route("/evaluations/:id", get(handlers::evaluations::get))
        .route("/experiments/:id/metrics", get(handlers::evaluations::get_metrics))

        // Health check
        .route("/health", get(health_check))

        // Metrics endpoint
        .route("/metrics", get(observability::metrics::metrics_handler))

        // Middleware layers
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
                .max_age(Duration::from_secs(3600))
        )
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

async fn health_check() -> &'static str {
    "OK"
}
