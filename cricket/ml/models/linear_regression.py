import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class T20LinearRegression:
    """Linear regression model for T20 runs prediction with MLflow integration."""

    def __init__(
        self,
        experiment_name: str = "male_team_level_t20",
        model_name: str = "t20_runs_predictor",
        random_state: Optional[int] = 42,
    ):
        """
        Initialize T20 linear regression model.

        Parameters
        ----------
        experiment_name : str
            MLflow experiment name
        model_name : str
            Model name for registry
        random_state : Optional[int]
            Random state for reproducibility
        """
        self.model = LinearRegression(fit_intercept=True)
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.random_state = random_state

        # Feature names for interpretability
        self.feature_names = ["current_score", "wickets_fallen", "overs_remaining"]

        # Training metadata
        self._is_trained = False
        self._training_metadata = {}

        logger.info(
            f"Initialized T20LinearRegression for experiment: {experiment_name}"
        )

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        log_model: bool = True,
    ) -> "T20LinearRegression":
        """
        Train the linear regression model with MLflow tracking.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        X_val : Optional[np.ndarray]
            Validation features
        y_val : Optional[np.ndarray]
            Validation targets
        log_model : bool
            Whether to log model to MLflow

        Returns
        -------
        T20LinearRegression
            Trained model instance
        """
        logger.info("Starting model training with MLflow tracking")

        # Set experiment
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run() as run:
            # Log run metadata
            run_id = run.info.run_id
            logger.info(f"MLflow run ID: {run_id}")

            # Train model
            logger.info(f"Training on {len(X_train)} samples")
            self.model.fit(X_train, y_train)

            # Make predictions
            y_pred_train = self.model.predict(X_train)

            # Log parameters
            params = {
                "model_type": "linear_regression",
                "n_features": X_train.shape[1],
                "n_training_samples": len(X_train),
                "fit_intercept": True,
                "random_state": self.random_state,
                "feature_names": ",".join(self.feature_names),
            }

            if X_val is not None:
                params["n_validation_samples"] = len(X_val)

            mlflow.log_params(params)

            # Calculate and log training metrics
            train_metrics = self._calculate_metrics(y_train, y_pred_train, "train")
            mlflow.log_metrics(train_metrics)

            # Calculate and log validation metrics if provided
            if X_val is not None and y_val is not None:
                y_pred_val = self.model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, y_pred_val, "val")
                mlflow.log_metrics(val_metrics)

                # Log additional validation info
                mlflow.log_metric(
                    "val_train_r2_diff",
                    train_metrics["train_r2"] - val_metrics["val_r2"],
                )

            # Log model coefficients and intercept
            coeff_dict = {
                f"coeff_{name}": coeff
                for name, coeff in zip(self.feature_names, self.model.coef_)
            }
            coeff_dict["intercept"] = self.model.intercept_
            mlflow.log_params(coeff_dict)

            # Log feature importance (absolute coefficients)
            importance_dict = {
                f"importance_{name}": abs(coeff)
                for name, coeff in zip(self.feature_names, self.model.coef_)
            }
            mlflow.log_metrics(importance_dict)

            # Log model if requested
            if log_model:
                mlflow.sklearn.log_model(
                    sk_model=self.model,
                    artifact_path="model",
                    registered_model_name=self.model_name,
                    input_example=X_train[:5] if len(X_train) > 5 else X_train,
                )
                logger.info(f"Model logged to registry as: {self.model_name}")

            # Store training metadata
            self._training_metadata = {
                "run_id": run_id,
                "experiment_name": self.experiment_name,
                "model_name": self.model_name,
                "training_samples": len(X_train),
                "validation_samples": len(X_val) if X_val is not None else 0,
                "train_r2": train_metrics["train_r2"],
                "train_rmse": train_metrics["train_rmse"],
            }

            self._is_trained = True
            logger.info("Model training completed successfully")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.

        Parameters
        ----------
        X : np.ndarray
            Features for prediction

        Returns
        -------
        np.ndarray
            Predicted values
        """
        if not self._is_trained:
            raise ValueError("Model has not been trained. Call train() first.")

        logger.info(f"Making predictions for {len(X)} samples")
        predictions = self.model.predict(X)

        # Ensure predictions are non-negative (runs can't be negative)
        predictions = np.maximum(predictions, 0)

        return predictions

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
        """
        Evaluate model performance on test data.

        Parameters
        ----------
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets

        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        if not self._is_trained:
            raise ValueError("Model has not been trained. Call train() first.")

        logger.info(f"Evaluating model on {len(X_test)} test samples")

        # Make predictions
        y_pred = self.predict(X_test)

        # Calculate metrics
        test_metrics = self._calculate_metrics(y_test, y_pred, "test")

        # Log test metrics to MLflow if in active run
        if mlflow.active_run():
            mlflow.log_metrics(test_metrics)

        return test_metrics

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ) -> dict[str, float]:
        """
        Calculate regression metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True values
        y_pred : np.ndarray
            Predicted values
        prefix : str
            Prefix for metric names

        Returns
        -------
        Dict[str, float]
            Calculated metrics
        """
        metrics = {
            f"{prefix}_r2": r2_score(y_true, y_pred),
            f"{prefix}_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
        }

        # Calculate MAPE, handling division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            # Replace inf/nan with 0
            if np.isfinite(mape):
                metrics[f"{prefix}_mape"] = mape
            else:
                metrics[f"{prefix}_mape"] = 0.0

        return metrics

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on absolute coefficients.

        Returns
        -------
        pd.DataFrame
            Feature importance dataframe
        """
        if not self._is_trained:
            raise ValueError("Model has not been trained. Call train() first.")

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "coefficient": self.model.coef_,
                "abs_coefficient": np.abs(self.model.coef_),
            }
        ).sort_values("abs_coefficient", ascending=False)

        # Add intercept
        intercept_row = pd.DataFrame(
            {
                "feature": ["intercept"],
                "coefficient": [self.model.intercept_],
                "abs_coefficient": [abs(self.model.intercept_)],
            }
        )

        importance_df = pd.concat([importance_df, intercept_row], ignore_index=True)

        return importance_df

    def get_model_equation(self) -> str:
        """
        Get the linear regression equation as a string.

        Returns
        -------
        str
            Model equation
        """
        if not self._is_trained:
            raise ValueError("Model has not been trained. Call train() first.")

        equation_parts = [f"{self.model.intercept_:.3f}"]

        for name, coeff in zip(self.feature_names, self.model.coef_):
            sign = "+" if coeff >= 0 else "-"
            equation_parts.append(f"{sign} {abs(coeff):.3f} * {name}")

        equation = "total_runs = " + " ".join(equation_parts)
        return equation

    def get_training_metadata(self) -> dict:
        """
        Get training metadata.

        Returns
        -------
        Dict
            Training metadata
        """
        return self._training_metadata.copy()

    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save model to disk using MLflow.

        Parameters
        ----------
        filepath : Union[str, Path]
            Path to save model
        """
        if not self._is_trained:
            raise ValueError("Model has not been trained. Call train() first.")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        mlflow.sklearn.save_model(self.model, str(filepath))
        logger.info(f"Model saved to: {filepath}")

    @classmethod
    def load_model(cls, model_uri: str) -> "T20LinearRegression":
        """
        Load model from MLflow registry or local path.

        Parameters
        ----------
        model_uri : str
            Model URI (e.g., "models:/t20_runs_predictor/1")

        Returns
        -------
        T20LinearRegression
            Loaded model instance
        """
        logger.info(f"Loading model from: {model_uri}")

        # Load sklearn model
        sklearn_model = mlflow.sklearn.load_model(model_uri)

        # Create new instance
        instance = cls()
        instance.model = sklearn_model
        instance._is_trained = True

        logger.info("Model loaded successfully")
        return instance


def predict_innings_total(
    current_score: float,
    wickets_fallen: int,
    overs_remaining: float,
    model_uri: str = "models:/t20_runs_predictor/latest",
) -> float:
    """
    Convenience function to predict innings total from current match state.

    Parameters
    ----------
    current_score : float
        Current team score
    wickets_fallen : int
        Wickets lost so far
    overs_remaining : float
        Overs remaining in innings
    model_uri : str
        MLflow model URI

    Returns
    -------
    float
        Predicted innings total
    """
    # Load model
    model = T20LinearRegression.load_model(model_uri)

    # Prepare features
    features = np.array([[current_score, wickets_fallen, overs_remaining]])

    # Make prediction
    prediction = model.predict(features)[0]

    return round(prediction, 1)  # ty: ignore
