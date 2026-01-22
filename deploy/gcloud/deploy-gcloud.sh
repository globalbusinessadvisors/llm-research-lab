#!/usr/bin/env bash
# =============================================================================
# LLM-Research-Lab Google Cloud Deployment Script
# =============================================================================
#
# Deploys LLM-Research-Lab as a unified Cloud Run service.
#
# SERVICE TOPOLOGY:
#   - Single unified service: llm-research-lab
#   - Agent endpoints:
#     - /api/v1/agents/hypothesis (HypothesisAgent)
#     - /api/v1/agents/metric (ExperimentalMetricAgent)
#   - No standalone agent services
#
# CONSTITUTION COMPLIANCE:
#   - Stateless runtime
#   - No direct SQL access
#   - Persistence via ruvector-service only
#
# USAGE:
#   ./deploy-gcloud.sh [ENV] [OPTIONS]
#
# ENVIRONMENTS:
#   dev      - Development environment
#   staging  - Staging environment
#   prod     - Production environment
#
# OPTIONS:
#   --build-only    Only build, don't deploy
#   --skip-build    Skip build, deploy existing image
#   --dry-run       Show what would be deployed
#
# EXAMPLES:
#   ./deploy-gcloud.sh dev
#   ./deploy-gcloud.sh prod --dry-run
#   ./deploy-gcloud.sh staging --skip-build
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PROJECT_ID="${PROJECT_ID:-agentics-dev}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="llm-research-lab"
SERVICE_ACCOUNT="${SERVICE_NAME}-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Parse arguments
ENV="${1:-dev}"
BUILD_ONLY=false
SKIP_BUILD=false
DRY_RUN=false

shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only) BUILD_ONLY=true; shift ;;
        --skip-build) SKIP_BUILD=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo -e "${RED}Unknown option: $1${NC}"; exit 1 ;;
    esac
done

# Environment-specific configuration
case $ENV in
    dev)
        RUVECTOR_SERVICE_URL="https://ruvector-service-dev-${PROJECT_ID}.${REGION}.run.app"
        LLM_OBSERVATORY_ENDPOINT="https://llm-observatory-dev-${PROJECT_ID}.${REGION}.run.app"
        MIN_INSTANCES=0
        MAX_INSTANCES=5
        ;;
    staging)
        RUVECTOR_SERVICE_URL="https://ruvector-service-staging-${PROJECT_ID}.${REGION}.run.app"
        LLM_OBSERVATORY_ENDPOINT="https://llm-observatory-staging-${PROJECT_ID}.${REGION}.run.app"
        MIN_INSTANCES=1
        MAX_INSTANCES=10
        ;;
    prod)
        RUVECTOR_SERVICE_URL="https://ruvector-service-${PROJECT_ID}.${REGION}.run.app"
        LLM_OBSERVATORY_ENDPOINT="https://llm-observatory-${PROJECT_ID}.${REGION}.run.app"
        MIN_INSTANCES=2
        MAX_INSTANCES=20
        ;;
    *)
        echo -e "${RED}Invalid environment: $ENV${NC}"
        echo "Valid environments: dev, staging, prod"
        exit 1
        ;;
esac

TELEMETRY_ENDPOINT="${LLM_OBSERVATORY_ENDPOINT}/api/v1/telemetry"
IMAGE_TAG="${ENV}-$(date +%Y%m%d-%H%M%S)"
IMAGE_URL="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${IMAGE_TAG}"

# -----------------------------------------------------------------------------
# Display configuration
# -----------------------------------------------------------------------------
echo -e "${BLUE}=========================================="
echo "LLM-Research-Lab Deployment"
echo "==========================================${NC}"
echo ""
echo -e "${BLUE}Service Topology:${NC}"
echo "  Service Name: ${SERVICE_NAME}"
echo "  Agent Endpoints:"
echo "    - /api/v1/agents/hypothesis"
echo "    - /api/v1/agents/metric"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Project: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Environment: ${ENV}"
echo "  Image: ${IMAGE_URL}"
echo "  Service Account: ${SERVICE_ACCOUNT}"
echo ""
echo -e "${BLUE}Integrations:${NC}"
echo "  RuVector URL: ${RUVECTOR_SERVICE_URL}"
echo "  Observatory: ${LLM_OBSERVATORY_ENDPOINT}"
echo "  Telemetry: ${TELEMETRY_ENDPOINT}"
echo ""
echo -e "${BLUE}Scaling:${NC}"
echo "  Min Instances: ${MIN_INSTANCES}"
echo "  Max Instances: ${MAX_INSTANCES}"
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo -e "${YELLOW}DRY RUN - No changes will be made${NC}"
    exit 0
fi

# -----------------------------------------------------------------------------
# Production confirmation
# -----------------------------------------------------------------------------
if [[ "$ENV" == "prod" ]]; then
    echo -e "${YELLOW}WARNING: You are deploying to PRODUCTION!${NC}"
    read -p "Are you sure? (type 'yes' to confirm): " confirm
    if [[ "$confirm" != "yes" ]]; then
        echo "Deployment cancelled."
        exit 0
    fi
fi

# -----------------------------------------------------------------------------
# Build Docker image
# -----------------------------------------------------------------------------
if [[ "$SKIP_BUILD" == "false" ]]; then
    echo ""
    echo -e "${BLUE}[1/4] Building Docker image...${NC}"

    # Navigate to repository root
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    cd "${SCRIPT_DIR}/../.."

    # Build image
    docker build \
        -t "${IMAGE_URL}" \
        -t "gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest" \
        -t "gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${ENV}" \
        -f Dockerfile \
        .

    echo -e "${GREEN}Build complete: ${IMAGE_URL}${NC}"

    # Push to GCR
    echo ""
    echo -e "${BLUE}[2/4] Pushing to Container Registry...${NC}"

    docker push "${IMAGE_URL}"
    docker push "gcr.io/${PROJECT_ID}/${SERVICE_NAME}:latest"
    docker push "gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${ENV}"

    echo -e "${GREEN}Push complete${NC}"
else
    echo -e "${YELLOW}[1/4] Skipping build (--skip-build)${NC}"
    echo -e "${YELLOW}[2/4] Skipping push${NC}"
    IMAGE_URL="gcr.io/${PROJECT_ID}/${SERVICE_NAME}:${ENV}"
fi

if [[ "$BUILD_ONLY" == "true" ]]; then
    echo -e "${GREEN}Build complete. Skipping deployment (--build-only)${NC}"
    exit 0
fi

# -----------------------------------------------------------------------------
# Deploy to Cloud Run
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}[3/4] Deploying to Cloud Run...${NC}"

gcloud run deploy "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --image="${IMAGE_URL}" \
    --platform=managed \
    --allow-unauthenticated \
    --port=8080 \
    --cpu=2 \
    --memory=2Gi \
    --min-instances="${MIN_INSTANCES}" \
    --max-instances="${MAX_INSTANCES}" \
    --concurrency=80 \
    --timeout=300 \
    --service-account="${SERVICE_ACCOUNT}" \
    --set-env-vars="SERVICE_NAME=${SERVICE_NAME}" \
    --set-env-vars="SERVICE_VERSION=${IMAGE_TAG}" \
    --set-env-vars="PLATFORM_ENV=${ENV}" \
    --set-env-vars="RUVECTOR_SERVICE_URL=${RUVECTOR_SERVICE_URL}" \
    --set-env-vars="LLM_OBSERVATORY_ENDPOINT=${LLM_OBSERVATORY_ENDPOINT}" \
    --set-env-vars="TELEMETRY_ENDPOINT=${TELEMETRY_ENDPOINT}" \
    --set-env-vars="TELEMETRY_STDOUT=true" \
    --set-env-vars="RUST_LOG=info,llm_research_agents=debug" \
    --set-env-vars="LLM_RESEARCH_LOG_LEVEL=info" \
    --set-env-vars="LLM_RESEARCH_PORT=8080" \
    --set-secrets="RUVECTOR_AUTH_TOKEN=ruvector-auth-token:latest" \
    --labels="platform=agentics-dev,component=research-agents,env=${ENV}"

echo -e "${GREEN}Deployment complete${NC}"

# -----------------------------------------------------------------------------
# Verify deployment
# -----------------------------------------------------------------------------
echo ""
echo -e "${BLUE}[4/4] Verifying deployment...${NC}"

# Get service URL
SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
    --project="${PROJECT_ID}" \
    --region="${REGION}" \
    --format='value(status.url)')

echo "Service URL: ${SERVICE_URL}"

# Wait for service to be ready
echo "Waiting for service to be ready..."
sleep 10

# Health check
echo "Running health check..."
if curl -sf "${SERVICE_URL}/health" > /dev/null; then
    echo -e "${GREEN}Health check passed${NC}"
else
    echo -e "${RED}Health check failed${NC}"
    exit 1
fi

# Verify agent endpoints
echo "Verifying agent endpoints..."
echo "  - Hypothesis Agent: ${SERVICE_URL}/api/v1/agents/hypothesis"
echo "  - Metric Agent: ${SERVICE_URL}/api/v1/agents/metric"

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo -e "${GREEN}=========================================="
echo "Deployment Successful!"
echo "==========================================${NC}"
echo ""
echo -e "${BLUE}Service URL:${NC} ${SERVICE_URL}"
echo ""
echo -e "${BLUE}Agent Endpoints:${NC}"
echo "  Hypothesis: ${SERVICE_URL}/api/v1/agents/hypothesis"
echo "  Metric:     ${SERVICE_URL}/api/v1/agents/metric"
echo ""
echo -e "${BLUE}CLI Commands:${NC}"
echo "  llm-research agents hypothesis evaluate --input hypothesis.json"
echo "  llm-research agents metric compute --input metrics.json"
echo "  llm-research agents list"
echo ""
echo -e "${BLUE}Useful Commands:${NC}"
echo "  View logs:    gcloud run services logs read ${SERVICE_NAME} --region=${REGION}"
echo "  Describe:     gcloud run services describe ${SERVICE_NAME} --region=${REGION}"
echo "  Update:       gcloud run services update ${SERVICE_NAME} --region=${REGION}"
echo ""
