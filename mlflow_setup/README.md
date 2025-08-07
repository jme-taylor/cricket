# MLflow Setup for T20 Cricket Analysis

This directory contains the MLflow configuration and setup files for the T20 linear regression model project.

## Directory Structure

```
mlflow_setup/
├── README.md                 # This file
├── mlflow_config.yaml       # Configuration settings
├── start_mlflow.sh          # Server startup script
├── mlflow.db               # SQLite database (created on first run)
├── mlruns/                 # Experiment artifacts
└── models/                 # Model registry artifacts
```

## Quick Start

### 1. Install MLflow
```bash
pip install mlflow
```

### 2. Start MLflow Server
```bash
cd mlflow_setup
./start_mlflow.sh
```

### 3. Access MLflow UI
Open your browser and navigate to: http://127.0.0.1:5000

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

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill existing MLflow processes
   lsof -ti:5000 | xargs kill -9
   ```

2. **Database Locked**
   ```bash
   # Check for existing connections
   ps aux | grep mlflow
   ```

3. **Permission Issues**
   ```bash
   # Ensure script is executable
   chmod +x start_mlflow.sh
   ```

### Logs and Debugging
- MLflow server logs are printed to console
- Database file: `mlflow.db`
- Artifacts stored in: `mlruns/`

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