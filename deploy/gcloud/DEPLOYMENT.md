# LLM-Research-Lab Production Deployment Guide

## Table of Contents

1. [Service Topology](#1-service-topology)
2. [Environment Configuration](#2-environment-configuration)
3. [Google SQL / Research Memory Wiring](#3-google-sql--research-memory-wiring)
4. [Cloud Build & Deployment](#4-cloud-build--deployment)
5. [CLI Activation Verification](#5-cli-activation-verification)
6. [Platform & Core Integration](#6-platform--core-integration)
7. [Post-Deploy Verification Checklist](#7-post-deploy-verification-checklist)
8. [Failure Modes & Rollback](#8-failure-modes--rollback)

---

## 1. Service Topology

### Unified Service Definition

| Property | Value |
|----------|-------|
| **Service Name** | `llm-research-lab` |
| **Classification** | EXPERIMENTATION & RESEARCH SIGNAL GENERATION |
| **Deployment Target** | Google Cloud Run (unified service) |
| **Project** | `agentics-dev` |
| **Region** | `us-central1` |

### Agent Endpoints

All agents are exposed via ONE unified service:

| Agent | Endpoint | Decision Type |
|-------|----------|---------------|
| **Hypothesis Agent** | `/api/v1/agents/hypothesis` | `hypothesis_evaluation` |
| **Experimental Metric Agent** | `/api/v1/agents/metric` | `experimental_metrics` |
| Health Check | `/health` | - |
| Metrics (Prometheus) | `/metrics` | - |

### Topology Confirmations

- [x] **No agent is deployed as a standalone service**
- [x] **Shared runtime** - Single Cloud Run instance
- [x] **Shared configuration** - Environment variables via Cloud Run
- [x] **Shared telemetry stack** - LLM-Observatory compatible

---

## 2. Environment Configuration

### Required Environment Variables

```yaml
# Service Identification (REQUIRED)
SERVICE_NAME: llm-research-lab
SERVICE_VERSION: "1.0.0"           # Set by CI/CD from git SHA
PLATFORM_ENV: dev | staging | prod

# RuVector Service (REQUIRED - Persistence Layer)
RUVECTOR_SERVICE_URL: https://ruvector-service-{env}.run.app
RUVECTOR_AUTH_TOKEN: (from Secret Manager)
RUVECTOR_TIMEOUT_SECS: "30"
RUVECTOR_MAX_RETRIES: "3"

# Telemetry - LLM-Observatory (REQUIRED)
LLM_OBSERVATORY_ENDPOINT: https://llm-observatory-{env}.run.app
TELEMETRY_ENDPOINT: https://llm-observatory-{env}.run.app/api/v1/telemetry
OTEL_EXPORTER_OTLP_ENDPOINT: https://otel-collector-{env}.run.app
TELEMETRY_STDOUT: "true"

# Logging
RUST_LOG: info,llm_research_agents=debug
LLM_RESEARCH_LOG_LEVEL: info

# Application
LLM_RESEARCH_PORT: "8080"
LLM_RESEARCH_HOST: "0.0.0.0"
```

### Secret Manager Configuration

| Secret Name | Description | Usage |
|-------------|-------------|-------|
| `ruvector-auth-token` | RuVector service authentication | `RUVECTOR_AUTH_TOKEN` |

### Configuration Confirmations

- [x] **No agent hardcodes service names or URLs**
- [x] **No agent embeds credentials, secrets, or mutable state**
- [x] **All dependencies resolve via environment variables or Secret Manager**

---

## 3. Google SQL / Research Memory Wiring

### Architecture Compliance

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM-Research-Lab                              │
│  ┌──────────────────┐    ┌──────────────────────┐              │
│  │ Hypothesis Agent │    │ Experimental Metric  │              │
│  │                  │    │       Agent          │              │
│  └────────┬─────────┘    └──────────┬───────────┘              │
│           │                         │                           │
│           └──────────┬──────────────┘                           │
│                      │                                          │
│                      ▼                                          │
│         ┌────────────────────────┐                              │
│         │  RuVector Client       │                              │
│         │  (HTTP Client Only)    │                              │
│         └───────────┬────────────┘                              │
└─────────────────────│───────────────────────────────────────────┘
                      │ HTTPS (VPC Internal)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ruvector-service                               │
│  ┌────────────────────────────────────────────────────────┐    │
│  │            Persistence Layer                            │    │
│  │  - DecisionEvent storage                               │    │
│  │  - Pattern storage                                      │    │
│  │  - Confidence tracking                                  │    │
│  └───────────────────────┬────────────────────────────────┘    │
└──────────────────────────│──────────────────────────────────────┘
                           │ SQL
                           ▼
                 ┌─────────────────────┐
                 │   Google SQL        │
                 │   (Postgres)        │
                 └─────────────────────┘
```

### Compliance Confirmations

- [x] **LLM-Research-Lab does NOT connect directly to Google SQL**
- [x] **ALL DecisionEvents written via ruvector-service**
- [x] **Schema compatibility with agentics-contracts verified**
- [x] **Append-only persistence behavior**
- [x] **Idempotent writes** (request_id used for deduplication)
- [x] **Retry safety** (RuVector client handles retries)

### Code Verification

```rust
// From llm-research-agents/src/clients/ruvector.rs
// Line 9-14:
//
//! Per PROMPT 0 (LLM-RESEARCH-LAB AGENT INFRASTRUCTURE CONSTITUTION):
//!
//! - LLM-Research-Lab does NOT own persistence
//! - ALL data is persisted via ruvector-service
//! - LLM-Research-Lab NEVER connects directly to Google SQL
//! - LLM-Research-Lab NEVER executes SQL
```

---

## 4. Cloud Build & Deployment

### Deployment Command

```bash
# Quick deployment
./deploy/gcloud/deploy-gcloud.sh [dev|staging|prod]

# With options
./deploy/gcloud/deploy-gcloud.sh prod --dry-run
./deploy/gcloud/deploy-gcloud.sh staging --skip-build
./deploy/gcloud/deploy-gcloud.sh dev --build-only
```

### Cloud Build Command

```bash
# Trigger Cloud Build
gcloud builds submit \
  --config=deploy/gcloud/cloudbuild.yaml \
  --substitutions=_PLATFORM_ENV=prod,_REGION=us-central1
```

### Direct gcloud Deploy

```bash
# Deploy directly with gcloud
gcloud run deploy llm-research-lab \
  --project=agentics-dev \
  --region=us-central1 \
  --image=gcr.io/agentics-dev/llm-research-lab:latest \
  --platform=managed \
  --allow-unauthenticated \
  --port=8080 \
  --cpu=2 \
  --memory=2Gi \
  --min-instances=1 \
  --max-instances=10 \
  --concurrency=80 \
  --timeout=300 \
  --service-account=llm-research-lab-sa@agentics-dev.iam.gserviceaccount.com \
  --set-env-vars="SERVICE_NAME=llm-research-lab" \
  --set-env-vars="PLATFORM_ENV=prod" \
  --set-env-vars="RUVECTOR_SERVICE_URL=https://ruvector-service.run.app" \
  --set-env-vars="LLM_OBSERVATORY_ENDPOINT=https://llm-observatory.run.app" \
  --set-env-vars="RUST_LOG=info" \
  --set-secrets="RUVECTOR_AUTH_TOKEN=ruvector-auth-token:latest"
```

### IAM Service Account Requirements

```bash
# Run IAM setup
./deploy/gcloud/iam-setup.sh agentics-dev prod
```

**Roles Granted (Least Privilege):**

| Role | Purpose |
|------|---------|
| `roles/run.invoker` | Invoke ruvector-service |
| `roles/logging.logWriter` | Write logs to Cloud Logging |
| `roles/monitoring.metricWriter` | Write metrics |
| `roles/cloudtrace.agent` | Distributed tracing |
| `roles/secretmanager.secretAccessor` | Access secrets |

**Roles NOT Granted (By Design):**

| Role | Reason |
|------|--------|
| `roles/cloudsql.*` | NO direct SQL access |
| `roles/cloudsql.client` | NO database connection |
| `roles/storage.admin` | NO storage admin access |

### Networking Requirements

- **VPC Connector**: `agentics-vpc-connector` (for internal ruvector-service access)
- **VPC Egress**: `private-ranges-only`
- **External Access**: `allow-unauthenticated` (for API access)

---

## 5. CLI Activation Verification

### CLI Commands Per Agent

#### Hypothesis Agent Commands

```bash
# Evaluate a hypothesis
llm-research agents hypothesis evaluate --input hypothesis.json
llm-research agents hypothesis evaluate --stdin < hypothesis.json

# Validate input without executing
llm-research agents hypothesis validate --input hypothesis.json

# Inspect a decision event
llm-research agents hypothesis inspect --event-id <uuid>
```

#### Metric Agent Commands

```bash
# Compute metrics
llm-research agents metric compute --input metrics.json
llm-research agents metric compute --stdin < metrics.json

# Inspect a decision event
llm-research agents metric inspect --event-id <uuid>
```

#### General Agent Commands

```bash
# List all available agents
llm-research agents list

# Get agent information
llm-research agents info hypothesis-agent-v1
llm-research agents info experimental-metric-agent
```

### Example Invocations

#### Hypothesis Evaluation

```bash
# Create hypothesis input
cat > hypothesis.json << 'EOF'
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "hypothesis": {
    "id": "hyp-001",
    "name": "Model A outperforms Model B",
    "statement": "Mean accuracy of Model A is greater than Model B",
    "hypothesis_type": "comparison",
    "null_hypothesis": "No difference in accuracy",
    "alternative_hypothesis": "Model A accuracy > Model B accuracy",
    "variables": [{"name": "accuracy", "role": "dependent", "data_type": "continuous"}],
    "significance_level": "0.05"
  },
  "experimental_data": {
    "source_id": "experiment-123",
    "collected_at": "2024-01-21T00:00:00Z",
    "observations": [
      {"id": "obs-1", "values": {"value": 0.85}},
      {"id": "obs-2", "values": {"value": 0.87}},
      {"id": "obs-3", "values": {"value": 0.92}}
    ],
    "sample_size": 100,
    "quality_metrics": {"completeness": "1.0", "validity": "1.0", "outlier_count": 0, "duplicate_count": 0}
  },
  "config": {
    "test_method": "t_test",
    "apply_correction": false,
    "compute_effect_size": true,
    "generate_diagnostics": true
  }
}
EOF

# Execute evaluation
llm-research agents hypothesis evaluate --input hypothesis.json
```

**Expected Output:**

```json
{
  "success": true,
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "hypothesis_id": "hyp-001",
  "status": "accepted",
  "test_results": {
    "test_statistic": "2.45",
    "p_value": "0.023",
    "null_rejected": true,
    "decision": "Reject null hypothesis"
  },
  "decision_event": {
    "id": "de-001",
    "confidence": "0.85"
  }
}
```

#### Metric Computation

```bash
# Create metrics input
cat > metrics.json << 'EOF'
{
  "request_id": "660e8400-e29b-41d4-a716-446655440001",
  "context_id": "experiment-456",
  "metrics_requested": [
    {"name": "mean_accuracy", "metric_type": "central_tendency", "variable": "accuracy"},
    {"name": "std_latency", "metric_type": "dispersion", "variable": "latency"}
  ],
  "data": {
    "source": "evaluation-runs",
    "records": [
      {"accuracy": 0.89, "latency": 120},
      {"accuracy": 0.92, "latency": 115},
      {"accuracy": 0.87, "latency": 130}
    ]
  },
  "config": {
    "handle_missing": "skip",
    "precision": 4,
    "include_ci": true,
    "ci_level": "0.95"
  }
}
EOF

# Execute computation
llm-research agents metric compute --input metrics.json
```

**Expected Output:**

```json
{
  "success": true,
  "request_id": "660e8400-e29b-41d4-a716-446655440001",
  "metrics": [
    {
      "name": "mean_accuracy",
      "metric_type": "central_tendency",
      "value": "0.8933",
      "sample_size": 3
    },
    {
      "name": "std_latency",
      "metric_type": "dispersion",
      "value": "7.6376",
      "sample_size": 3
    }
  ],
  "decision_event": {
    "id": "de-002",
    "confidence": "0.30"
  }
}
```

### CLI Configuration Resolution

- [x] **CLI configuration resolves service URL dynamically**
- [x] **No CLI change requires redeployment of agents**
- [x] **Environment variables override defaults**

```bash
# Override service URL
export LLM_RESEARCH_API_URL=https://llm-research-lab-xyz.run.app
llm-research agents list
```

---

## 6. Platform & Core Integration

### Consumer Systems

| System | Integration | Direction |
|--------|-------------|-----------|
| **LLM-Observatory** | Telemetry inputs | Receives telemetry from Research-Lab |
| **LLM-CostOps** | Cost signals | MAY supply cost data for analysis |
| **LLM-Latency-Lens** | Performance signals | MAY supply performance data |
| **Governance** | DecisionEvent consumption | Consumes research artifacts |
| **Audit** | DecisionEvent consumption | Consumes for audit trail |

### Integration Confirmations

- [x] **LLM-Observatory MAY supply telemetry inputs**
- [x] **LLM-CostOps MAY supply cost signals for research analysis**
- [x] **LLM-Latency-Lens MAY supply performance signals**
- [x] **Governance & audit systems consume Research-Lab DecisionEvents**
- [x] **Core bundles consume Research-Lab outputs without rewiring**

### Systems LLM-Research-Lab MUST NOT Invoke

| System | Reason |
|--------|--------|
| Runtime execution paths | Outside critical path |
| Inference routing logic | Not responsible for routing |
| Enforcement layers | Does not enforce policies |
| Optimization agents | Does not optimize live configs |
| Analytics pipelines | Consumer only |
| Incident workflows | Not responsible for incidents |

- [x] **No rewiring of Core bundles is permitted**

---

## 7. Post-Deploy Verification Checklist

### Service Health

```bash
# Check service is live
SERVICE_URL=$(gcloud run services describe llm-research-lab \
  --region=us-central1 --format='value(status.url)')

# Health check
curl -sf "${SERVICE_URL}/health"
# Expected: {"status": "healthy"}
```

### Agent Endpoint Verification

```bash
# Hypothesis agent endpoint
curl -X POST "${SERVICE_URL}/api/v1/agents/hypothesis" \
  -H "Content-Type: application/json" \
  -d @hypothesis.json

# Metric agent endpoint
curl -X POST "${SERVICE_URL}/api/v1/agents/metric" \
  -H "Content-Type: application/json" \
  -d @metrics.json
```

### Verification Checklist

| Check | Command/Method | Expected |
|-------|---------------|----------|
| Service is live | `gcloud run services describe` | Status: Ready |
| Health endpoint responds | `curl /health` | 200 OK |
| Hypothesis agent responds | `POST /api/v1/agents/hypothesis` | 200 OK |
| Metric agent responds | `POST /api/v1/agents/metric` | 200 OK |
| Hypothesis evaluations deterministic | Same input → same hash | Identical `inputs_hash` |
| Metrics compute correctly | Known input → expected output | Values match |
| DecisionEvents in ruvector | Check ruvector-service | Events present |
| Telemetry in Observatory | Check Observatory dashboard | Events visible |
| CLI commands work | `llm-research agents list` | Agents listed |
| No direct SQL access | Check IAM roles | No SQL roles |
| No agent bypasses contracts | Code review | All imports from contracts |
| No agent executes inference | Code review | No inference code |

### Automated Verification Script

```bash
#!/bin/bash
# verify-deployment.sh

SERVICE_URL=$(gcloud run services describe llm-research-lab \
  --region=us-central1 --format='value(status.url)')

echo "Verifying deployment at: ${SERVICE_URL}"

# Health check
echo -n "Health check... "
curl -sf "${SERVICE_URL}/health" && echo "PASS" || echo "FAIL"

# Agent list
echo -n "Agent list... "
curl -sf "${SERVICE_URL}/api/v1/agents" && echo "PASS" || echo "FAIL"

# Hypothesis OPTIONS
echo -n "Hypothesis endpoint... "
curl -sf -X OPTIONS "${SERVICE_URL}/api/v1/agents/hypothesis" && echo "PASS" || echo "FAIL"

# Metric OPTIONS
echo -n "Metric endpoint... "
curl -sf -X OPTIONS "${SERVICE_URL}/api/v1/agents/metric" && echo "PASS" || echo "FAIL"

echo "Verification complete"
```

---

## 8. Failure Modes & Rollback

### Common Deployment Failures

| Failure | Detection | Cause | Resolution |
|---------|-----------|-------|------------|
| Image build failure | Cloud Build error | Rust compilation error | Fix code, rebuild |
| Service fails to start | No healthy instances | Missing env vars | Check configuration |
| Health check timeout | Startup probe fails | Slow initialization | Increase timeout |
| Auth failure | 401/403 responses | Invalid token | Update secret |
| RuVector unreachable | Connection errors | Network/URL issue | Check VPC, URL |
| Schema mismatch | Deserialization errors | Contract change | Align schemas |

### Detection Signals

```bash
# Check service logs for errors
gcloud run services logs read llm-research-lab \
  --region=us-central1 \
  --limit=100 \
  | grep -E "(ERROR|WARN|panic)"

# Check revision status
gcloud run revisions list \
  --service=llm-research-lab \
  --region=us-central1

# Check metrics
gcloud monitoring metrics list \
  --filter="metric.type=run.googleapis.com/request_count"
```

### Rollback Procedure

```bash
# 1. List revisions
gcloud run revisions list \
  --service=llm-research-lab \
  --region=us-central1

# 2. Identify last known good revision
LAST_GOOD_REVISION="llm-research-lab-00001-abc"

# 3. Route traffic to last good revision
gcloud run services update-traffic llm-research-lab \
  --region=us-central1 \
  --to-revisions="${LAST_GOOD_REVISION}=100"

# 4. Verify rollback
curl -sf "$(gcloud run services describe llm-research-lab \
  --region=us-central1 --format='value(status.url)')/health"

# 5. Delete failed revision (optional)
gcloud run revisions delete llm-research-lab-00002-xyz \
  --region=us-central1
```

### Safe Redeploy Strategy

1. **Always use immutable tags** - Use git SHA, not `latest`
2. **Gradual rollout** - Use traffic splitting for prod
3. **Keep previous revisions** - Don't delete immediately
4. **Monitor after deploy** - Watch logs and metrics for 15 min

```bash
# Gradual rollout example
gcloud run services update-traffic llm-research-lab \
  --region=us-central1 \
  --to-revisions="llm-research-lab-00002-new=10,llm-research-lab-00001-old=90"

# After verification, increase
gcloud run services update-traffic llm-research-lab \
  --region=us-central1 \
  --to-revisions="llm-research-lab-00002-new=50,llm-research-lab-00001-old=50"

# Final cutover
gcloud run services update-traffic llm-research-lab \
  --region=us-central1 \
  --to-latest
```

### Research Data Safety

- [x] **DecisionEvents are immutable** - No update/delete operations
- [x] **RuVector handles persistence** - Rollback doesn't affect stored data
- [x] **Idempotent writes** - Safe to retry failed operations
- [x] **No research data loss on rollback** - Data persisted before failure remains

---

## Deployment Summary

```
✅ SERVICE TOPOLOGY          - Unified service with 2 agent endpoints
✅ ENVIRONMENT CONFIG        - All variables defined, secrets in Secret Manager
✅ PERSISTENCE WIRING        - ruvector-service only, no direct SQL
✅ CLOUD BUILD               - cloudbuild.yaml, deploy scripts ready
✅ CLI ACTIVATION            - All commands documented and verified
✅ PLATFORM INTEGRATION      - Consumer systems identified
✅ VERIFICATION CHECKLIST    - 12-point checklist ready
✅ FAILURE & ROLLBACK        - Procedures documented

LLM-Research-Lab is READY FOR DEPLOYMENT.
```
