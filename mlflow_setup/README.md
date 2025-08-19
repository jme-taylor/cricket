# MLflow Setup for T20 Cricket Analysis

This directory contains the MLflow configuration and setup files for the T20 linear regression model project, with enhanced automation for process management and zombie cleanup.

## Directory Structure

```
mlflow_setup/
├── README.md                 # This file
├── mlflow_config.yaml       # Configuration settings
├── start_mlflow.sh          # Enhanced server startup script
├── stop_mlflow.sh           # Server shutdown script  
├── restart_mlflow.sh        # Server restart script
├── status_mlflow.sh         # Server status check script
├── mlflow.db               # SQLite database (created on first run)
├── mlflow.pid              # Process ID file (created when running)
├── logs/                   # Server logs directory
│   └── mlflow.log          # Server log file
├── mlruns/                 # Experiment artifacts
└── models/                 # Model registry artifacts
```

## Quick Start

### 1. Prerequisites
Ensure MLflow is installed in your virtual environment:
```bash
# From the main project directory
uv sync --locked --dev
```

### 2. Start MLflow Server
```bash
cd mlflow_setup
./start_mlflow.sh
```

The startup script will automatically:
- Kill any zombie MLflow processes
- Check port availability (uses port 5001 to avoid macOS conflicts)
- Activate the virtual environment
- Start the server with proper logging
- Verify the server is responding

### 3. Access MLflow UI
Open your browser and navigate to: **http://127.0.0.1:5001**

## Management Commands

The enhanced management system provides these scripts:

### Server Control
```bash
./start_mlflow.sh    # Start server (with automatic cleanup)
./stop_mlflow.sh     # Gracefully stop server
./restart_mlflow.sh  # Stop and restart server
./status_mlflow.sh   # Check server status and health
```

### Status Information
The status script provides comprehensive information:
- Process status and PID
- Port availability 
- UI accessibility test
- Database statistics (experiments/runs count)
- Recent log entries

## Configuration

### Experiment Settings
- **Experiment Name**: `male_team_level_t20`
- **Model Name**: `t20_runs_predictor`
- **Backend**: SQLite database
- **Artifacts**: Local filesystem

### Tracked Metrics
- R² Score (coefficient of determination)
- RMSE (root mean squared error)
- MAE (mean absolute error)  
- MAPE (mean absolute percentage error)

### Logged Parameters
- Model type and configuration
- Feature scaling method
- Data splitting strategy
- Sample points for feature extraction
- Filter criteria (T20/IT20, male matches)

## Usage

### From Python Code
```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow_setup/mlflow.db")

# Set experiment
mlflow.set_experiment("male_team_level_t20")

# Start run and log metrics
with mlflow.start_run():
    mlflow.log_param("model_type", "linear_regression")
    mlflow.log_metric("r2_score", 0.82)
    # ... train and log model
```

### From Command Line
```bash
# List experiments
mlflow experiments list

# Run MLflow project
mlflow run . -P experiment_name=male_team_level_t20

# Search runs
mlflow runs list -e male_team_level_t20
```

## Model Registry

### Register Model
```python
mlflow.sklearn.log_model(
    model, 
    "model",
    registered_model_name="t20_runs_predictor"
)
```

### Load Model
```python
# Load specific version
model = mlflow.sklearn.load_model("models:/t20_runs_predictor/1")

# Load latest production model
model = mlflow.sklearn.load_model("models:/t20_runs_predictor/Production")
```

## Troubleshooting

### Automated Problem Resolution
The enhanced scripts automatically handle common issues:

**Zombie Processes**: Automatically detected and killed during startup
**Port Conflicts**: Changed from port 5000 to 5001 to avoid macOS conflicts
**Virtual Environment**: Automatically activated if available
**Health Checks**: Server status verified before reporting success

### Manual Troubleshooting

1. **Check Server Status**
   ```bash
   ./status_mlflow.sh
   ```
   This provides comprehensive diagnostics including:
   - Process status and health
   - Port usage
   - Database connectivity
   - UI accessibility

2. **View Server Logs**
   ```bash
   tail -f logs/mlflow.log
   ```

3. **Force Stop All Processes**
   ```bash
   ./stop_mlflow.sh
   # If processes persist:
   pkill -9 -f "mlflow"
   ```

4. **Database Issues**
   ```bash
   # Check database integrity
   sqlite3 mlflow.db ".schema"
   ```

5. **Permission Issues**
   ```bash
   # Make scripts executable (should be done automatically)
   chmod +x *.sh
   ```

### Logs and Debugging
- **Server logs**: `logs/mlflow.log` (rotated automatically)
- **Process tracking**: `mlflow.pid` file
- **Database file**: `mlflow.db`
- **Artifacts**: `mlruns/` directory
- **Configuration**: `mlflow_config.yaml`

### Known Issues and Solutions

**Port 5000 Conflicts**: macOS ControlCenter uses port 5000. Scripts now use port 5001.

**Zombie Processes**: Previous MLflow instances can persist. The start script automatically cleans these up.

**Virtual Environment**: Scripts require the project's virtual environment. They automatically activate it if found at `../venv/`.

## Security Notes

This setup is configured for local development only:
- SQLite database (not suitable for production)
- No authentication/authorization
- Local filesystem storage
- Single-user access

For production deployments, consider:
- PostgreSQL/MySQL backend
- S3/GCS artifact storage
- Authentication integration
- Load balancer setup