"""
Simplified training pipeline for men's T20 cricket ML models.

This module provides a streamlined approach to training both runs and wicket
prediction models with proper MLflow tracking and data preparation.
"""

import mlflow
import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

from cricket.ml.config import MODEL_CONFIG
from cricket.ml.data_preparation import T20DataPreparator, load_cricket_data
from cricket.ml.feature_engineering import FeatureEngineer
from cricket.ml.models.base_models import create_runs_predictor, create_wicket_predictor
from cricket.transformation.match import (
    get_current_score,
    get_wickets_fallen,
    get_overs_remaining,
    get_balls_remaining,
    get_current_run_rate,
    get_required_run_rate,
    get_innings_indicator,
)

logger = logging.getLogger(__name__)


class MensT20TrainingPipeline:
    """Simplified training pipeline for men's T20 cricket models."""

    def __init__(
        self,
        data_path: str,
        mlflow_tracking_uri: Optional[str] = None,
        scaling_method: str = "standard",
    ):
        """
        Initialize training pipeline.

        Parameters
        ----------
        data_path : str
            Path to cricket data parquet file
        mlflow_tracking_uri : Optional[str]
            MLflow tracking URI, defaults to local SQLite
        scaling_method : str
            Feature scaling method
        """
        self.data_path = Path(data_path)

        # Initialize components
        self.preparator = T20DataPreparator()
        self.feature_engineer = FeatureEngineer(scaling_method=scaling_method)

        # Models will be created during training
        self.runs_model = None
        self.wicket_model = None

        # Set MLflow tracking URI
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI set to: {mlflow_tracking_uri}")
        else:
            # Default to local SQLite setup
            default_uri = "sqlite:///mlflow_setup/mlflow.db"
            mlflow.set_tracking_uri(default_uri)
            logger.info(f"Using default MLflow tracking URI: {default_uri}")

        # Pipeline state
        self._pipeline_run = False
        self._results = {}

        logger.info("MensT20TrainingPipeline initialized")

    def run_training(
        self,
        train_runs_model: bool = True,
        train_wicket_model: bool = True,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, Any]:
        """
        Execute the complete training pipeline for both models.

        Parameters
        ----------
        train_runs_model : bool
            Whether to train the runs prediction model
        train_wicket_model : bool
            Whether to train the wicket prediction model
        train_ratio : float
            Training data ratio
        val_ratio : float
            Validation data ratio
        test_ratio : float
            Test data ratio

        Returns
        -------
        Dict[str, Any]
            Training results for both models
        """
        logger.info("Starting men's T20 training pipeline")

        try:
            # 1. Load and prepare data
            logger.info("Step 1: Loading and preparing cricket data")
            modeling_data = self._load_and_prepare_data()

            # 2. Split data chronologically
            logger.info("Step 2: Splitting data chronologically")
            train_data, val_data, test_data = self._split_data(
                modeling_data, train_ratio, val_ratio, test_ratio
            )

            # 3. Prepare features for modeling
            logger.info("Step 3: Engineering and scaling features")
            model_inputs = self._prepare_model_inputs(train_data, val_data, test_data)

            results = {}

            # 4. Train runs model
            if train_runs_model:
                logger.info("Step 4a: Training runs prediction model")
                runs_results = self._train_runs_model(model_inputs)
                results["runs_model"] = runs_results

            # 5. Train wicket model
            if train_wicket_model:
                logger.info("Step 4b: Training wicket prediction model")
                wicket_results = self._train_wicket_model(model_inputs)
                results["wicket_model"] = wicket_results

            self._pipeline_run = True
            self._results = results

            logger.info("Men's T20 training pipeline completed successfully")
            return results

        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise

    def _load_and_prepare_data(self) -> pl.DataFrame:
        """Load data and add all required features."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        # Load raw data
        raw_data = load_cricket_data(str(self.data_path))
        logger.info(f"Loaded {len(raw_data)} rows of cricket data")

        # Filter for men's T20/IT20 matches only
        filters = MODEL_CONFIG["data_filters"]
        data = raw_data.filter(
            (pl.col("gender") == filters["gender"])
            & (pl.col("match_type").is_in(filters["match_types"]))
        )
        logger.info(f"Filtered to {len(data)} rows for men's T20/IT20")

        # Validate filtered data
        quality_summary = self.preparator.validate_data_quality(data)
        logger.info(f"Data quality: {quality_summary['unique_matches']} matches")

        # Add all match state features
        logger.info("Adding match state features")
        data = get_current_score(data)
        data = get_wickets_fallen(data)
        data = get_overs_remaining(data)
        data = get_balls_remaining(data)
        data = get_current_run_rate(data)
        data = get_required_run_rate(data)
        data = get_innings_indicator(data)

        # Validate features were added
        self.preparator.validate_match_state_features(data)
        logger.info("All match state features added successfully")

        # Prepare modeling data with targets
        modeling_data = self.preparator.prepare_all_balls_data(data)
        logger.info(f"Prepared {len(modeling_data)} ball-level samples for modeling")

        return modeling_data

    def _split_data(
        self,
        data: pl.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split data chronologically."""
        return self.preparator.split_data_chronologically(
            data, train_ratio, val_ratio, test_ratio
        )

    def _prepare_model_inputs(
        self,
        train_data: pl.DataFrame,
        val_data: pl.DataFrame,
        test_data: pl.DataFrame,
    ) -> Dict[str, np.ndarray]:
        """Prepare features and targets for both models."""
        logger.info("Preparing model inputs")

        # Prepare features
        train_features = self.feature_engineer.prepare_features(train_data)
        val_features = self.feature_engineer.prepare_features(val_data)
        test_features = self.feature_engineer.prepare_features(test_data)

        # Scale features
        X_train = self.feature_engineer.fit_transform(train_features)
        X_val = self.feature_engineer.transform_features(val_features)
        X_test = self.feature_engineer.transform_features(test_features)

        # Extract targets for runs model (total runs)
        y_runs_train = train_data.select("total_runs_innings").to_numpy().flatten()
        y_runs_val = val_data.select("total_runs_innings").to_numpy().flatten()
        y_runs_test = test_data.select("total_runs_innings").to_numpy().flatten()

        # Extract targets for wicket model (wicket on current ball)
        # Create wicket indicator: 1 if wicket falls on this ball, 0 otherwise
        y_wicket_train = self._create_wicket_targets(train_data)
        y_wicket_val = self._create_wicket_targets(val_data)
        y_wicket_test = self._create_wicket_targets(test_data)

        logger.info(
            f"Prepared inputs - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
        )

        return {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_runs_train": y_runs_train,
            "y_runs_val": y_runs_val,
            "y_runs_test": y_runs_test,
            "y_wicket_train": y_wicket_train,
            "y_wicket_val": y_wicket_val,
            "y_wicket_test": y_wicket_test,
        }

    def _create_wicket_targets(self, data: pl.DataFrame) -> np.ndarray:
        """Create binary targets for wicket prediction."""
        # Check if a wicket falls on the current ball
        wicket_targets = (
            data.with_columns(
                ((pl.col("player_out_1") != "") | (pl.col("player_out_2") != ""))
                .cast(pl.Int32)
                .alias("wicket_this_ball")
            )
            .select("wicket_this_ball")
            .to_numpy()
            .flatten()
        )

        return wicket_targets

    def _train_runs_model(self, model_inputs: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Train the runs prediction model."""
        logger.info("Training runs prediction model")

        # Create model
        self.runs_model = create_runs_predictor()

        # Train model
        trained_model = self.runs_model.train(
            X_train=model_inputs["X_train"],
            y_train=model_inputs["y_runs_train"],
            X_val=model_inputs["X_val"],
            y_val=model_inputs["y_runs_val"],
            log_model=True,
        )

        # Evaluate on test set
        test_metrics = trained_model.evaluate(
            model_inputs["X_test"], model_inputs["y_runs_test"]
        )

        # Compile results
        results = {
            "model": trained_model,
            "test_metrics": test_metrics,
            "model_equation": trained_model.get_model_equation(),
            "training_metadata": trained_model.get_training_metadata(),
        }

        logger.info(
            f"Runs model training complete. Test R²: {test_metrics['test_r2']:.3f}"
        )
        return results

    def _train_wicket_model(
        self, model_inputs: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Train the wicket prediction model."""
        logger.info("Training wicket prediction model")

        # Create model
        self.wicket_model = create_wicket_predictor()

        # Train model
        trained_model = self.wicket_model.train(
            X_train=model_inputs["X_train"],
            y_train=model_inputs["y_wicket_train"],
            X_val=model_inputs["X_val"],
            y_val=model_inputs["y_wicket_val"],
            log_model=True,
        )

        # Evaluate on test set
        test_metrics = trained_model.evaluate(
            model_inputs["X_test"], model_inputs["y_wicket_test"]
        )

        # Compile results
        results = {
            "model": trained_model,
            "test_metrics": test_metrics,
            "training_metadata": trained_model.get_training_metadata(),
        }

        logger.info(
            f"Wicket model training complete. Test accuracy: {test_metrics['test_accuracy']:.3f}"
        )
        return results

    def get_results(self) -> Dict[str, Any]:
        """Get pipeline results."""
        if not self._pipeline_run:
            raise ValueError("Pipeline has not been run. Call run_training() first.")

        return self._results.copy()


def quick_train_models(
    data_path: str, mlflow_uri: Optional[str] = None, train_both: bool = True
) -> Dict[str, Any]:
    """
    Quick training function with default parameters.

    Parameters
    ----------
    data_path : str
        Path to cricket data
    mlflow_uri : Optional[str]
        MLflow tracking URI
    train_both : bool
        Train both models (True) or just runs model (False)

    Returns
    -------
    Dict[str, Any]
        Training results
    """
    logger.info("Starting quick men's T20 model training")

    pipeline = MensT20TrainingPipeline(
        data_path=data_path, mlflow_tracking_uri=mlflow_uri
    )

    results = pipeline.run_training(
        train_runs_model=True, train_wicket_model=train_both
    )

    # Print results summary
    print("\n=== Men's T20 Model Training Complete ===")

    if "runs_model" in results:
        runs_metrics = results["runs_model"]["test_metrics"]
        print(
            f"Runs Model - Test R²: {runs_metrics['test_r2']:.3f}, RMSE: {runs_metrics['test_rmse']:.1f}"
        )
        print(f"Runs Model Equation: {results['runs_model']['model_equation']}")

    if "wicket_model" in results:
        wicket_metrics = results["wicket_model"]["test_metrics"]
        print(f"Wicket Model - Test Accuracy: {wicket_metrics['test_accuracy']:.3f}")
        if "test_auc" in wicket_metrics:
            print(f"Wicket Model - Test AUC: {wicket_metrics['test_auc']:.3f}")

    return results
