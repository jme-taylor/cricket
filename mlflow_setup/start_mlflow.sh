#!/bin/bash

# MLflow Tracking Server Startup Script
# For T20 Linear Regression Model Tracking

echo "Starting MLflow Tracking Server for T20 Cricket Analysis..."
echo "Experiment: male_team_level_t20"
echo "UI will be available at: http://127.0.0.1:5000"
echo ""

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

# Start MLflow server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 127.0.0.1 \
    --port 5000 \
    --serve-artifacts

echo ""
echo "MLflow server stopped."