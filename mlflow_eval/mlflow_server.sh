#!/bin/bash

# =========================
# MLflow Local Server Setup
# =========================

# Directory to store MLflow artifacts (models, metrics, logs)
ARTIFACT_ROOT="./mlruns"

# SQLite file for backend metadata
BACKEND_DB="./mlflow.db"

# Server host and port
HOST="127.0.0.1"
PORT=5000

# Make sure directories exist
mkdir -p $ARTIFACT_ROOT

echo "Starting MLflow server..."
echo "Backend store URI: sqlite:///$BACKEND_DB"
echo "Default artifact root: $ARTIFACT_ROOT"
echo "Listening on http://$HOST:$PORT"

mlflow server \
  --backend-store-uri sqlite:///$BACKEND_DB \
  --host $HOST \
  --port $PORT