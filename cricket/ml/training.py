import mlflow
import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional
import logging

from cricket.ml.data_preparation import T20DataPreparator, load_cricket_data
from cricket.ml.feature_engineering import FeatureEngineer
from cricket.ml.models.linear_regression import T20LinearRegression
from cricket.transformation.match import (
    get_current_score,
    get_wickets_fallen,
    get_overs_remaining,
)

logger = logging.getLogger(__name__)


class T20TrainingPipeline:
    """Complete training pipeline for T20 linear regression model."""

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
        self.model = T20LinearRegression()

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

        logger.info("T20TrainingPipeline initialized")

    def run_training(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> dict:
        """
        Execute the complete training pipeline.

        Parameters
        ----------
        train_ratio : float
            Training data ratio
        val_ratio : float
            Validation data ratio
        test_ratio : float
            Test data ratio

        Returns
        -------
        Dict
            Training results and metadata
        """
        logger.info("Starting complete T20 training pipeline")

        try:
            # 1. Load and validate data
            logger.info("Step 1: Loading cricket data")
            raw_data = self._load_data()

            # 2. Filter and prepare T20 data
            logger.info("Step 2: Filtering T20/IT20 matches")
            clean_data = self._prepare_t20_data(raw_data)

            # 3. Add match state features
            logger.info("Step 3: Adding match state features")
            features_data = self._add_match_features(clean_data)

            # 4. Create target variable and sample features
            logger.info("Step 4: Creating targets and sampling features")
            target_df, feature_samples = self._create_modeling_data(features_data)

            # 5. Split data chronologically
            logger.info("Step 5: Splitting data chronologically")
            train_data, val_data, test_data = self._split_data(
                feature_samples, target_df, train_ratio, val_ratio, test_ratio
            )

            # 6. Prepare features for modeling
            logger.info("Step 6: Engineering and scaling features")
            X_train, X_val, X_test, y_train, y_val, y_test = self._prepare_model_inputs(
                train_data, val_data, test_data, target_df
            )

            # 7. Train model with MLflow tracking
            logger.info("Step 7: Training model with MLflow")
            trained_model = self._train_model(X_train, y_train, X_val, y_val)

            # 8. Evaluate on test set
            logger.info("Step 8: Final evaluation on test set")
            test_metrics = self._evaluate_model(trained_model, X_test, y_test)

            # 9. Compile results
            results = self._compile_results(
                trained_model, test_metrics, len(X_train), len(X_val), len(X_test)
            )

            self._pipeline_run = True
            self._results = results

            logger.info("T20 training pipeline completed successfully")
            return results

        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise

    def _load_data(self) -> pl.DataFrame:
        """Load cricket data from file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        data = load_cricket_data(str(self.data_path))
        logger.info(f"Loaded {len(data)} rows of cricket data")

        return data

    def _prepare_t20_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Filter and prepare T20/IT20 data."""
        # Filter for T20/IT20 matches
        clean_data = self.preparator.filter_t20_matches(df)

        # Validate data quality
        quality_summary = self.preparator.validate_data_quality(clean_data)
        logger.info(
            f"Data quality validation completed: {quality_summary['unique_matches']} matches"
        )

        return clean_data

    def _add_match_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """Add match state features to the data."""
        logger.info("Adding match state features")

        # Add current score
        df = get_current_score(df)

        # Add wickets fallen
        df = get_wickets_fallen(df)

        # Add overs remaining
        df = get_overs_remaining(df)

        # Validate features were added using the data preparator
        self.preparator.validate_match_state_features(df)

        logger.info("Match state features added successfully")
        return df

    def _create_modeling_data(
        self, df: pl.DataFrame
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Create target variable and prepare features."""
        # Use all balls as training data
        modeling_data = self.preparator.prepare_all_balls_data(df)

        # Split into features and create a dummy target_df for compatibility
        # The modeling_data already has targets joined
        target_df = modeling_data.select(
            [
                "match_id",
                "innings_number",
                "total_runs_innings",
                "team",
                "match_type",
                "gender",
            ]
        ).unique(["match_id", "innings_number"])

        logger.info(
            f"Created {len(modeling_data)} ball-level training samples from {len(target_df)} innings"
        )

        return target_df, modeling_data

    def _split_data(
        self,
        feature_samples: pl.DataFrame,
        target_df: pl.DataFrame,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """Split data chronologically."""
        # Split the feature_samples as they already contain the targets
        train_features, val_features, test_features = (
            self.preparator.split_data_chronologically(
                feature_samples, train_ratio, val_ratio, test_ratio
            )
        )

        return (train_features, val_features, test_features)

    def _prepare_model_inputs(
        self,
        train_features: pl.DataFrame,
        val_features: pl.DataFrame,
        test_features: pl.DataFrame,
        target_df: pl.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and targets for modeling."""
        train_joined = self._join_features_with_targets(train_features, target_df)
        val_joined = self._join_features_with_targets(val_features, target_df)
        test_joined = self._join_features_with_targets(test_features, target_df)

        train_features_aligned = train_joined.drop("total_runs_innings")
        val_features_aligned = val_joined.drop("total_runs_innings")
        test_features_aligned = test_joined.drop("total_runs_innings")

        train_prep = self.feature_engineer.prepare_features(train_features_aligned)
        val_prep = self.feature_engineer.prepare_features(val_features_aligned)
        test_prep = self.feature_engineer.prepare_features(test_features_aligned)

        X_train = self.feature_engineer.fit_transform(train_prep)
        X_val = self.feature_engineer.transform_features(val_prep)
        X_test = self.feature_engineer.transform_features(test_prep)

        y_train = train_joined.select("total_runs_innings").to_numpy().flatten()
        y_val = val_joined.select("total_runs_innings").to_numpy().flatten()
        y_test = test_joined.select("total_runs_innings").to_numpy().flatten()

        logger.info(
            f"Prepared inputs - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _join_features_with_targets(
        self, features_df: pl.DataFrame, target_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Join features with targets to ensure alignment."""
        # Features already have targets when using all balls
        if "total_runs_innings" in features_df.columns:
            return features_df

        # Fallback: join features with targets on match_id and innings_number
        joined = features_df.join(
            target_df.select(["match_id", "innings_number", "total_runs_innings"]),
            on=["match_id", "innings_number"],
            how="inner",
        )

        return joined

    def _get_targets_for_features(
        self, features_df: pl.DataFrame, target_df: pl.DataFrame
    ) -> np.ndarray:
        """Get target values corresponding to feature samples."""
        # Join features with targets on match_id and innings_number
        joined = features_df.join(
            target_df.select(["match_id", "innings_number", "total_runs_innings"]),
            on=["match_id", "innings_number"],
            how="inner",
        )

        return joined.select("total_runs_innings").to_numpy().flatten()

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> T20LinearRegression:
        """Train the linear regression model."""
        trained_model = self.model.train(X_train, y_train, X_val, y_val, log_model=True)

        # Log additional pipeline metadata
        with mlflow.start_run(nested=True):
            mlflow.log_params(
                {
                    "pipeline_training_mode": "all_balls",
                    "pipeline_scaling_method": self.feature_engineer.scaling_method,
                    "pipeline_data_path": str(self.data_path),
                }
            )

        return trained_model

    def _evaluate_model(
        self, model: T20LinearRegression, X_test: np.ndarray, y_test: np.ndarray
    ) -> dict:
        """Evaluate model on test set."""
        test_metrics = model.evaluate(X_test, y_test)

        # Log to MLflow
        with mlflow.start_run(nested=True):
            mlflow.log_metrics(test_metrics)
            mlflow.log_param("evaluation_samples", len(X_test))

        return test_metrics

    def _compile_results(
        self,
        model: T20LinearRegression,
        test_metrics: dict,
        n_train: int,
        n_val: int,
        n_test: int,
    ) -> dict:
        """Compile training pipeline results."""
        results = {
            "model": model,
            "test_metrics": test_metrics,
            "feature_importance": model.get_feature_importance(),
            "model_equation": model.get_model_equation(),
            "training_metadata": model.get_training_metadata(),
            "data_splits": {
                "train_samples": n_train,
                "val_samples": n_val,
                "test_samples": n_test,
            },
            "pipeline_config": {
                "training_mode": "all_balls",
                "scaling_method": self.feature_engineer.scaling_method,
                "data_path": str(self.data_path),
            },
        }

        return results

    def get_results(self) -> dict:
        """Get pipeline results."""
        if not self._pipeline_run:
            raise ValueError("Pipeline has not been run. Call run_training() first.")

        return self._results.copy()

    def save_model(self, filepath: str) -> None:
        """Save trained model to disk."""
        if not self._pipeline_run:
            raise ValueError("Pipeline has not been run. Call run_training() first.")

        self._results["model"].save_model(filepath)
        logger.info(f"Model saved to: {filepath}")


def quick_train_t20_model(data_path: str, mlflow_uri: Optional[str] = None) -> dict:
    """
    Quick training function with default parameters.

    Parameters
    ----------
    data_path : str
        Path to cricket data
    mlflow_uri : Optional[str]
        MLflow tracking URI

    Returns
    -------
    Dict
        Training results
    """
    logger.info("Starting quick T20 model training")

    pipeline = T20TrainingPipeline(data_path=data_path, mlflow_tracking_uri=mlflow_uri)

    results = pipeline.run_training()

    test_r2 = results["test_metrics"]["test_r2"]
    test_rmse = results["test_metrics"]["test_rmse"]

    print("\n=== T20 Model Training Complete ===")
    print(f"Test R² Score: {test_r2:.3f}")
    print(f"Test RMSE: {test_rmse:.1f} runs")
    print(f"Model Equation: {results['model_equation']}")
    print("\nFeature Importance:")
    print(results["feature_importance"])

    return results
