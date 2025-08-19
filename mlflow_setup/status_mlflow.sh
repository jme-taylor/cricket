#!/bin/bash

# MLflow Server Status Check Script

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/mlflow.pid"
LOG_FILE="$SCRIPT_DIR/logs/mlflow.log"
PORT=5001
HOST=127.0.0.1

# Activate virtual environment if it exists for sqlite3 access
VENV_PATH="/Users/jamie/personal/cricket/.venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate" 2>/dev/null
fi

echo "📊 MLflow Server Status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check PID file
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    echo "📝 PID File Status:"
    echo "  • File: $PID_FILE"
    echo "  • PID: $PID"
    
    if kill -0 "$PID" 2>/dev/null; then
        echo "  • Status: ✅ Process is running"
        
        # Get process details
        echo ""
        echo "📋 Process Details:"
        ps -p "$PID" -o pid,ppid,user,start,time,command | head -2
    else
        echo "  • Status: ❌ Process not found (stale PID file)"
    fi
else
    echo "📝 PID File: Not found"
fi

echo ""
echo "🌐 Port $PORT Status:"
# Check port
if lsof -Pi :$PORT -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  • Status: ✅ Port is in use"
    echo "  • Processes using port $PORT:"
    lsof -Pi :$PORT -sTCP:LISTEN | grep -v "^COMMAND" | head -5 | while read line; do
        echo "    $line"
    done
else
    echo "  • Status: ❌ Port is free (no listener)"
fi

echo ""
echo "🔍 MLflow Processes:"
MLFLOW_PROCS=$(pgrep -f "mlflow" | wc -l | tr -d ' ')
if [ "$MLFLOW_PROCS" -gt 0 ]; then
    echo "  • Found $MLFLOW_PROCS MLflow-related process(es):"
    ps aux | grep mlflow | grep -v grep | grep -v status_mlflow | head -5 | while read line; do
        echo "    $(echo "$line" | awk '{printf "PID: %-6s CPU: %-5s MEM: %-5s CMD: %s\n", $2, $3, $4, substr($0, index($0,$11))}')"
    done
else
    echo "  • No MLflow processes found"
fi

echo ""
echo "🌐 UI Accessibility Test:"
# Test UI accessibility
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "http://$HOST:$PORT" 2>/dev/null)
if [ "$HTTP_CODE" = "200" ]; then
    echo "  • Status: ✅ UI is accessible"
    echo "  • URL: http://$HOST:$PORT"
    
    # Try to get experiment count
    EXPERIMENTS=$(sqlite3 "$SCRIPT_DIR/mlflow.db" "SELECT COUNT(*) FROM experiments;" 2>/dev/null)
    if [ -n "$EXPERIMENTS" ]; then
        echo "  • Experiments in database: $EXPERIMENTS"
    fi
    
    # Try to get run count
    RUNS=$(sqlite3 "$SCRIPT_DIR/mlflow.db" "SELECT COUNT(*) FROM runs;" 2>/dev/null)
    if [ -n "$RUNS" ]; then
        echo "  • Total runs in database: $RUNS"
    fi
elif [ -n "$HTTP_CODE" ]; then
    echo "  • Status: ⚠️  Server responded with HTTP $HTTP_CODE"
else
    echo "  • Status: ❌ UI is not accessible"
fi

echo ""
echo "📄 Log File:"
if [ -f "$LOG_FILE" ]; then
    echo "  • File: $LOG_FILE"
    echo "  • Size: $(ls -lh "$LOG_FILE" | awk '{print $5}')"
    echo "  • Last modified: $(ls -l "$LOG_FILE" | awk '{print $6, $7, $8}')"
    echo "  • Last 5 lines:"
    tail -5 "$LOG_FILE" | while IFS= read -r line; do
        echo "    $line"
    done
else
    echo "  • No log file found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Summary
echo ""
if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ MLflow server appears to be running correctly"
    echo "🌐 Access the UI at: http://$HOST:$PORT"
else
    echo "❌ MLflow server is not running properly"
    echo "💡 Run './start_mlflow.sh' to start the server"
fi