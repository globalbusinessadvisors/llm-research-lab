# =============================================================================
# LLM Research Lab - Cloud Run Dockerfile
# =============================================================================
#
# Multi-stage build for the unified LLM Research Lab Cloud Run service.
#
# CONSTITUTION COMPLIANCE:
#   - Stateless runtime
#   - No database credentials baked in
#   - All persistence via ruvector-service
#
# SERVICE TOPOLOGY:
#   - Single binary: llm-research-lab
#   - Agent endpoints:
#     - /api/v1/agents/hypothesis
#     - /api/v1/agents/metric
#
# BUILD:
#   docker build -t llm-research-lab .
#
# RUN:
#   docker run -p 8080:8080 \
#     -e RUVECTOR_SERVICE_URL=https://ruvector-service.run.app \
#     -e LLM_OBSERVATORY_ENDPOINT=https://llm-observatory.run.app \
#     -e TELEMETRY_ENDPOINT=https://llm-observatory.run.app/api/v1/telemetry \
#     llm-research-lab
# =============================================================================

# ============================================================================
# Stage 1: Builder - Compile the Rust application
# ============================================================================
FROM rust:1.83-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy workspace manifest
COPY Cargo.toml ./

# Copy all crate manifests
COPY llm-research-lab/Cargo.toml ./llm-research-lab/
COPY llm-research-core/Cargo.toml ./llm-research-core/
COPY llm-research-agents/Cargo.toml ./llm-research-agents/

# Copy all source code
COPY llm-research-lab ./llm-research-lab
COPY llm-research-core ./llm-research-core
COPY llm-research-agents ./llm-research-agents

# Copy configuration files (if they exist)
COPY config ./config

# Build release binary with optimizations
RUN cargo build --release --bin llm-research-lab

# Strip debug symbols to reduce binary size
RUN strip /app/target/release/llm-research-lab

# ============================================================================
# Stage 2: Runtime - Minimal runtime environment
# ============================================================================
FROM debian:bookworm-slim

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r llmlab && useradd -r -g llmlab -u 1000 llmlab

# Create app directory
WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/llm-research-lab /app/llm-research-lab

# Create data directories
RUN mkdir -p /app/data /app/logs \
    && chown -R llmlab:llmlab /app

# Switch to non-root user
USER llmlab

# Expose application port
EXPOSE 8080

# Set environment variables
# CONSTITUTION: No database URLs here - provided at runtime
ENV RUST_LOG=info,llm_research_lab=debug,llm_research_agents=debug \
    LLM_RESEARCH_PORT=8080 \
    LLM_RESEARCH_LOG_LEVEL=info \
    TELEMETRY_STDOUT=true

# Health check using curl
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -sf http://localhost:8080/health || exit 1

# Run the application
ENTRYPOINT ["/app/llm-research-lab"]
CMD []
