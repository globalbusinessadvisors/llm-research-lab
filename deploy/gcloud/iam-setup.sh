#!/usr/bin/env bash
# =============================================================================
# IAM Setup for LLM-Research-Lab
# =============================================================================
#
# This script creates the service account and assigns minimum required
# permissions for LLM-Research-Lab deployment.
#
# LEAST PRIVILEGE PRINCIPLE:
#   - Service account can ONLY access what it needs
#   - No database admin permissions (no direct SQL access)
#   - Can invoke ruvector-service
#   - Can write logs and metrics
#
# USAGE:
#   ./iam-setup.sh [PROJECT_ID] [ENV]
#
# EXAMPLE:
#   ./iam-setup.sh agentics-dev prod
# =============================================================================

set -euo pipefail

PROJECT_ID="${1:-agentics-dev}"
ENV="${2:-dev}"
SERVICE_ACCOUNT_NAME="llm-research-lab-sa"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

echo "=========================================="
echo "LLM-Research-Lab IAM Setup"
echo "=========================================="
echo "Project: ${PROJECT_ID}"
echo "Environment: ${ENV}"
echo "Service Account: ${SERVICE_ACCOUNT_EMAIL}"
echo "=========================================="

# -----------------------------------------------------------------------------
# Step 1: Create service account
# -----------------------------------------------------------------------------
echo ""
echo "[1/5] Creating service account..."

if gcloud iam service-accounts describe "${SERVICE_ACCOUNT_EMAIL}" --project="${PROJECT_ID}" &>/dev/null; then
    echo "Service account already exists: ${SERVICE_ACCOUNT_EMAIL}"
else
    gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
        --project="${PROJECT_ID}" \
        --display-name="LLM Research Lab Service Account" \
        --description="Service account for LLM-Research-Lab agents (hypothesis, metrics)"

    echo "Created service account: ${SERVICE_ACCOUNT_EMAIL}"
fi

# -----------------------------------------------------------------------------
# Step 2: Assign Cloud Run invoker role (to call ruvector-service)
# -----------------------------------------------------------------------------
echo ""
echo "[2/5] Granting Cloud Run invoker role..."

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/run.invoker" \
    --condition=None \
    --quiet

echo "Granted: roles/run.invoker"

# -----------------------------------------------------------------------------
# Step 3: Assign logging and monitoring roles
# -----------------------------------------------------------------------------
echo ""
echo "[3/5] Granting logging and monitoring roles..."

# Log writer
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/logging.logWriter" \
    --condition=None \
    --quiet

echo "Granted: roles/logging.logWriter"

# Metrics writer
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/monitoring.metricWriter" \
    --condition=None \
    --quiet

echo "Granted: roles/monitoring.metricWriter"

# Trace agent
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/cloudtrace.agent" \
    --condition=None \
    --quiet

echo "Granted: roles/cloudtrace.agent"

# -----------------------------------------------------------------------------
# Step 4: Assign Secret Manager accessor role
# -----------------------------------------------------------------------------
echo ""
echo "[4/5] Granting Secret Manager accessor role..."

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None \
    --quiet

echo "Granted: roles/secretmanager.secretAccessor"

# -----------------------------------------------------------------------------
# Step 5: Create secrets (if they don't exist)
# -----------------------------------------------------------------------------
echo ""
echo "[5/5] Setting up secrets..."

# RuVector auth token secret
if gcloud secrets describe "ruvector-auth-token" --project="${PROJECT_ID}" &>/dev/null; then
    echo "Secret 'ruvector-auth-token' already exists"
else
    echo "Creating secret 'ruvector-auth-token'..."
    echo "PLACEHOLDER_TOKEN" | gcloud secrets create "ruvector-auth-token" \
        --project="${PROJECT_ID}" \
        --data-file=- \
        --replication-policy="automatic"

    echo "Created secret. Please update with actual token:"
    echo "  echo 'YOUR_TOKEN' | gcloud secrets versions add ruvector-auth-token --data-file=-"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo ""
echo "=========================================="
echo "IAM Setup Complete"
echo "=========================================="
echo ""
echo "Service Account: ${SERVICE_ACCOUNT_EMAIL}"
echo ""
echo "Roles Granted:"
echo "  - roles/run.invoker (invoke ruvector-service)"
echo "  - roles/logging.logWriter (write logs)"
echo "  - roles/monitoring.metricWriter (write metrics)"
echo "  - roles/cloudtrace.agent (distributed tracing)"
echo "  - roles/secretmanager.secretAccessor (access secrets)"
echo ""
echo "Roles NOT Granted (by design):"
echo "  - roles/cloudsql.* (NO direct SQL access)"
echo "  - roles/cloudsql.client (NO database connection)"
echo "  - roles/storage.admin (NO storage admin)"
echo ""
echo "Secrets Created:"
echo "  - ruvector-auth-token (update with actual token)"
echo ""
echo "Next Steps:"
echo "  1. Update ruvector-auth-token secret with actual value"
echo "  2. Run deployment: ./deploy-gcloud.sh ${ENV}"
echo "=========================================="
