#!/bin/bash

# MLflow Server Restart Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🔄 Restarting MLflow Server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Stop the server
"$SCRIPT_DIR/stop_mlflow.sh"

echo ""
echo "⏳ Waiting for cleanup to complete..."
sleep 3

echo ""
# Start the server
"$SCRIPT_DIR/start_mlflow.sh"