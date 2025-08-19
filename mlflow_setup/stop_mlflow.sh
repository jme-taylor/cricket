#!/bin/bash

# MLflow Server Stop Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/mlflow.pid"
PORT=5001

echo "🛑 Stopping MLflow Server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Try to stop using PID file first
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "📝 Found MLflow server with PID: $PID"
        kill "$PID"
        echo "  ✓ Sent termination signal"
        
        # Wait for graceful shutdown
        for i in {1..5}; do
            if ! kill -0 "$PID" 2>/dev/null; then
                echo "  ✓ Server stopped gracefully"
                rm "$PID_FILE"
                break
            fi
            sleep 1
        done
        
        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            echo "  ⚠️  Server didn't stop gracefully, forcing..."
            kill -9 "$PID" 2>/dev/null
            echo "  ✓ Server forcefully terminated"
        fi
        rm -f "$PID_FILE"
    else
        echo "⚠️  PID file exists but process not found"
        rm "$PID_FILE"
    fi
fi

# Clean up any remaining MLflow processes
echo "🧹 Cleaning up any remaining processes..."
pkill -f "mlflow.server:app" 2>/dev/null
pkill -f "mlflow server" 2>/dev/null
sleep 1

# Force kill any stubborn processes
pkill -9 -f "mlflow.server:app" 2>/dev/null
pkill -9 -f "mlflow server" 2>/dev/null

# Check if port is now free
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Warning: Port $PORT is still in use by another process"
    echo "   Run 'lsof -i :$PORT' to investigate"
else
    echo "✓ Port $PORT is now free"
fi

# Check for any remaining processes
if pgrep -f "mlflow" >/dev/null 2>&1; then
    echo ""
    echo "⚠️  Some MLflow-related processes may still be running:"
    ps aux | grep mlflow | grep -v grep | grep -v stop_mlflow
else
    echo "✅ All MLflow processes stopped successfully"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"