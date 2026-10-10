#!/usr/bin/env bash
# Deploy to Google Cloud Run. Usage: PROJECT_ID=my-gcp-project ./deploy/deploy_cloud_run.sh
set -euo pipefail
: "${PROJECT_ID:?set PROJECT_ID}"
REGION="${REGION:-us-east1}"
SERVICE="${SERVICE:-faa-wildlife-dashboard}"
gcloud config set project "$PROJECT_ID"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud run deploy "$SERVICE" --source . --region "$REGION" --allow-unauthenticated \
  --memory 1Gi --cpu 1 --port 8080
gcloud run services describe "$SERVICE" --region "$REGION" --format 'value(status.url)'
