#!/bin/bash

# MLflow Tracking Server Startup Script with Automatic Cleanup
# For T20 Linear Regression Model Tracking

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/mlflow.pid"
LOG_DIR="$SCRIPT_DIR/logs"
LOG_FILE="$LOG_DIR/mlflow.log"
PORT=5001
HOST=127.0.0.1

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

# Function to cleanup zombie processes
cleanup_zombies() {
    echo "🧹 Cleaning up any existing MLflow processes..."
    
    # Kill any existing MLflow server processes
    pkill -f "mlflow.server:app" 2>/dev/null
    pkill -f "mlflow server" 2>/dev/null
    
    # Wait for processes to terminate
    sleep 2
    
    # Force kill if still running
    pkill -9 -f "mlflow.server:app" 2>/dev/null
    pkill -9 -f "mlflow server" 2>/dev/null
    
    # Remove PID file if exists
    if [ -f "$PID_FILE" ]; then
        rm "$PID_FILE"
        echo "  ✓ Removed stale PID file"
    fi
    
    echo "  ✓ Cleanup complete"
}

# Function to check if port is available
check_port() {
    if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠️  Port $PORT is in use. Attempting to free it..."
        cleanup_zombies
        sleep 2
        
        # Check again
        if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
            echo "❌ Port $PORT is still in use by another process"
            echo "   Run 'lsof -i :$PORT' to see what's using it"
            exit 1
        fi
    fi
    echo "✓ Port $PORT is available"
}

# Main startup sequence
echo "🚀 Starting MLflow Tracking Server for T20 Cricket Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Clean up any zombie processes first
cleanup_zombies

# Check port availability
check_port

# Activate virtual environment if it exists
VENV_PATH="/Users/jamie/personal/cricket/.venv"
if [ -d "$VENV_PATH" ]; then
    echo "🐍 Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
    echo "  ✓ Virtual environment activated"
fi

# Set MLflow tracking URI
export MLFLOW_TRACKING_URI="sqlite:///$SCRIPT_DIR/mlflow.db"

echo ""
echo "📊 Configuration:"
echo "  • Experiment: male_team_level_t20"
echo "  • Backend: SQLite ($SCRIPT_DIR/mlflow.db)"
echo "  • Artifacts: $SCRIPT_DIR/mlruns"
echo "  • Host: $HOST"
echo "  • Port: $PORT (changed from 5000 to avoid conflicts)"
echo "  • Logs: $LOG_FILE"
echo ""

# Start MLflow server in background and capture PID
echo "Starting server..."
nohup mlflow server \
    --backend-store-uri "sqlite:///$SCRIPT_DIR/mlflow.db" \
    --default-artifact-root "$SCRIPT_DIR/mlruns" \
    --host $HOST \
    --port $PORT \
    --serve-artifacts \
    > "$LOG_FILE" 2>&1 &

# Capture the PID
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"

# Wait for server to start
echo -n "Waiting for server to start"
for i in {1..10}; do
    sleep 1
    echo -n "."
    if curl -s -o /dev/null -w "%{http_code}" "http://$HOST:$PORT" 2>/dev/null | grep -q "200"; then
        echo ""
        echo ""
        echo "✅ MLflow server started successfully!"
        echo "🌐 UI available at: http://$HOST:$PORT"
        echo "📝 Process ID: $SERVER_PID"
        echo ""
        echo "To stop the server, run: ./stop_mlflow.sh"
        echo "To check status, run: ./status_mlflow.sh"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        exit 0
    fi
done

echo ""
echo "❌ Failed to start MLflow server"
echo "Check logs at: $LOG_FILE"
tail -20 "$LOG_FILE"
exit 1