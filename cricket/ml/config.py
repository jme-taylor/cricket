"""
Configuration for cricket ML models.

This module centralizes feature definitions and model configuration
to make it easy to add new features and modify model behavior.
"""

from typing import List, Dict, Any

# Feature configuration for men's T20 models
FEATURE_CONFIG = {
    # Raw features from the feature specification
    "base_features": [
        "runs_scored",
        "wickets_lost",
        "balls_remaining",
        "current_run_rate",
        "required_run_rate",
        "is_first_innings",
    ],
    # Mapping from specification names to actual column names
    "feature_mappings": {
        "runs_scored": "current_score",
        "wickets_lost": "wickets_fallen",
    },
    # Actual column names used in data processing
    "data_columns": [
        "current_score",
        "wickets_fallen",
        "balls_remaining",
        "current_run_rate",
        "required_run_rate",
        "is_first_innings",
    ],
}

# Model configuration for men's T20
MODEL_CONFIG = {
    # Data filtering
    "data_filters": {"gender": "male", "match_types": ["T20", "IT20"]},
    # MLflow experiment names
    "experiments": {
        "runs_model": "mens_t20_runs_model",
        "wicket_model": "mens_t20_wicket_model",
    },
    # Model registry names
    "model_names": {
        "runs_predictor": "mens_t20_runs_predictor",
        "wicket_predictor": "mens_t20_wicket_predictor",
    },
    # Data splitting ratios
    "data_splits": {"train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15},
}

# Feature engineering configuration
FEATURE_ENGINEERING_CONFIG = {
    "scaling_method": "standard",  # standard, minmax, or none
    "handle_missing": True,
    "fill_values": {
        "current_score": 0,
        "wickets_fallen": 0,
        "balls_remaining": 0,
        "current_run_rate": 0.0,
        "required_run_rate": None,  # Keep as null for first innings
        "is_first_innings": 1,
    },
}


def get_feature_columns() -> List[str]:
    """Get the list of feature columns for modeling."""
    return FEATURE_CONFIG["data_columns"].copy()


def get_mapped_feature_name(spec_name: str) -> str:
    """Map feature specification name to actual data column name."""
    return FEATURE_CONFIG["feature_mappings"].get(spec_name, spec_name)


def get_model_config(model_type: str) -> Dict[str, Any]:
    """Get configuration for a specific model type."""
    if model_type not in ["runs_model", "wicket_model"]:
        raise ValueError(f"Unknown model type: {model_type}")

    return {
        "experiment_name": MODEL_CONFIG["experiments"][model_type],
        "model_name": MODEL_CONFIG["model_names"][
            "runs_predictor" if model_type == "runs_model" else "wicket_predictor"
        ],
        "data_filters": MODEL_CONFIG["data_filters"],
        "data_splits": MODEL_CONFIG["data_splits"],
    }
