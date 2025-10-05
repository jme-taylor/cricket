import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from typing import Any
import logging
from abc import ABC, abstractmethod

from cricket.ml.config import get_feature_columns

logger = logging.getLogger(__name__)


class BasePredictor(ABC):
    """Abstract base class for T20 cricket predictors with MLflow integration."""

    def __init__(
        self, experiment_name: str, model_name: str, random_state: int | None = 42
    ):
        """
        Initialize base predictor.

        Parameters
        ----------
        experiment_name : str
            MLflow experiment name
        model_name : str
            Model name for registry
        random_state : int | None
            Random state for reproducibility, defaults to 42
        """
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.random_state = random_state

        # Feature configuration
        self.feature_names = get_feature_columns()

        # Training state
        self._is_trained = False
        self._training_metadata = {}

        # Model instance (to be set by subclasses)
        self.model = None

        logger.info(
            f"Initialized {self.__class__.__name__} for experiment: {experiment_name}"
        )

    @abstractmethod
    def _create_model(self):
        """Create the sklearn model instance. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ) -> dict[str, float]:
        """Calculate model-specific metrics. Must be implemented by subclasses."""
        pass

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        log_model: bool = True,
    ) -> "BasePredictor":
        """
        Train the model with MLflow tracking.

        Parameters
        ----------
        X_train : np.ndarray
            Training features
        y_train : np.ndarray
            Training targets
        X_val : np.ndarray | None
            Validation features, defaults to None
        y_val : np.ndarray | None
            Validation targets, defaults to None
        log_model : bool
            Whether to log model to MLflow

        Returns
        -------
        BasePredictor
            Trained model instance
        """
        logger.info(f"Starting {self.__class__.__name__} training with MLflow tracking")

        if self.model is None:
            self.model = self._create_model()

        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run() as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run ID: {run_id}")

            logger.info(f"Training on {len(X_train)} samples")
            self.model.fit(X_train, y_train)

            y_pred_train = self.model.predict(X_train)

            params = self._get_model_params(X_train, X_val)
            mlflow.log_params(params)

            train_metrics = self._calculate_metrics(y_train, y_pred_train, "train")
            mlflow.log_metrics(train_metrics)

            if X_val is not None and y_val is not None:
                y_pred_val = self.model.predict(X_val)
                val_metrics = self._calculate_metrics(y_val, y_pred_val, "val")
                mlflow.log_metrics(val_metrics)

                # Log validation difference for key metric
                key_metric = (
                    "r2"
                    if hasattr(self, "_is_regressor") and self._is_regressor
                    else "accuracy"
                )
                if (
                    f"train_{key_metric}" in train_metrics
                    and f"val_{key_metric}" in val_metrics
                ):
                    mlflow.log_metric(
                        f"val_train_{key_metric}_diff",
                        train_metrics[f"train_{key_metric}"]
                        - val_metrics[f"val_{key_metric}"],
                    )

            # Log model-specific parameters
            self._log_model_specific_params()

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
                **{k: v for k, v in train_metrics.items() if k.startswith("train_")},
            }

            self._is_trained = True
            logger.info(f"{self.__class__.__name__} training completed successfully")

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
            Predictions
        """
        if not self._is_trained:
            raise ValueError("Model has not been trained. Call train() first.")

        logger.info(f"Making predictions for {len(X)} samples")
        predictions = self.model.predict(X)

        return self._postprocess_predictions(predictions)

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

    def _get_model_params(
        self, X_train: np.ndarray, X_val: np.ndarray | None
    ) -> dict[str, Any]:
        """Get common model parameters for logging."""
        params = {
            "model_type": self.__class__.__name__.lower(),
            "n_features": X_train.shape[1],
            "n_training_samples": len(X_train),
            "random_state": self.random_state,
            "feature_names": ",".join(self.feature_names),
        }

        if X_val is not None:
            params["n_validation_samples"] = len(X_val)

        return params

    def _log_model_specific_params(self):
        """Log model-specific parameters. Override in subclasses if needed."""
        pass

    def _postprocess_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Post-process predictions. Override in subclasses if needed."""
        return predictions

    def get_training_metadata(self) -> dict[str, Any]:
        """Get training metadata."""
        return self._training_metadata.copy()

    @classmethod
    def load_model(cls, model_uri: str) -> "BasePredictor":
        """
        Load model from MLflow registry or local path.

        Parameters
        ----------
        model_uri : str
            Model URI (e.g., "models:/mens_t20_runs_predictor/1")

        Returns
        -------
        BasePredictor
            Loaded model instance
        """
        logger.info(f"Loading model from: {model_uri}")

        # Load sklearn model
        sklearn_model = mlflow.sklearn.load_model(model_uri)

        # Create new instance (note: experiment and model names will be default)
        instance = cls(experiment_name="loaded", model_name="loaded")
        instance.model = sklearn_model
        instance._is_trained = True

        logger.info("Model loaded successfully")
        return instance


class RunsPredictor(BasePredictor):
    """LinearRegression model for predicting total runs in T20 innings."""

    def __init__(
        self, experiment_name: str, model_name: str, random_state: int | None = 42
    ):
        super().__init__(experiment_name, model_name, random_state)
        self._is_regressor = True

    def _create_model(self):
        """Create LinearRegression model."""
        return LinearRegression(fit_intercept=True)

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ) -> dict[str, float]:
        """Calculate regression metrics."""
        metrics = {
            f"{prefix}_r2": r2_score(y_true, y_pred),
            f"{prefix}_rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            f"{prefix}_mae": mean_absolute_error(y_true, y_pred),
        }

        # Calculate MAPE, handling division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            if np.isfinite(mape):
                metrics[f"{prefix}_mape"] = mape
            else:
                metrics[f"{prefix}_mape"] = 0.0

        return metrics

    def _log_model_specific_params(self):
        """Log linear regression coefficients."""
        if hasattr(self.model, "coef_") and hasattr(self.model, "intercept_"):
            # Log coefficients
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

    def _postprocess_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Ensure predictions are non-negative (runs can't be negative)."""
        return np.maximum(predictions, 0)

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

        return "total_runs = " + " ".join(equation_parts)


class WicketPredictor(BasePredictor):
    """LogisticRegression model for predicting wicket probability on current ball."""

    def __init__(
        self, experiment_name: str, model_name: str, random_state: int | None = 42
    ):
        super().__init__(experiment_name, model_name, random_state)
        self._is_regressor = False

    def _create_model(self):
        """Create LogisticRegression model."""
        return LogisticRegression(
            random_state=self.random_state, fit_intercept=True, max_iter=1000
        )

    def _calculate_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ) -> dict[str, float]:
        """Calculate classification metrics."""
        # For probability predictions, convert to binary
        if y_pred.ndim > 1 or (
            y_pred.dtype == float and ((y_pred >= 0) & (y_pred <= 1)).all()
        ):
            y_pred_binary = (y_pred > 0.5).astype(int)
        else:
            y_pred_binary = y_pred.astype(int)

        metrics = {
            f"{prefix}_accuracy": accuracy_score(y_true, y_pred_binary),
            f"{prefix}_precision": precision_score(
                y_true, y_pred_binary, zero_division=0
            ),
            f"{prefix}_recall": recall_score(y_true, y_pred_binary, zero_division=0),
            f"{prefix}_f1": f1_score(y_true, y_pred_binary, zero_division=0),
        }

        # Add AUC if we have probabilities
        try:
            if hasattr(self.model, "predict_proba") and self._is_trained:
                # Use probability predictions for AUC
                y_prob = self.model.predict_proba(
                    X=getattr(self, "_last_X", None)
                    if hasattr(self, "_last_X")
                    else None
                )
                if y_prob is not None and y_prob.shape[1] > 1:
                    metrics[f"{prefix}_auc"] = roc_auc_score(y_true, y_prob[:, 1])
            else:
                # Use decision function or raw predictions
                metrics[f"{prefix}_auc"] = roc_auc_score(y_true, y_pred)
        except (ValueError, AttributeError):
            # Skip AUC if we can't calculate it
            pass

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict wicket probabilities."""
        if not self._is_trained:
            raise ValueError("Model has not been trained. Call train() first.")

        logger.info(f"Making probability predictions for {len(X)} samples")

        # Store X for potential AUC calculation
        self._last_X = X

        # Return probabilities for positive class (wicket)
        probabilities = self.model.predict_proba(X)[:, 1]
        return probabilities

    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """Predict binary wicket outcomes (0/1)."""
        probabilities = self.predict(X)
        return (probabilities > 0.5).astype(int)

    def _log_model_specific_params(self):
        """Log logistic regression coefficients."""
        if hasattr(self.model, "coef_") and hasattr(self.model, "intercept_"):
            # Log coefficients
            coeff_dict = {
                f"coeff_{name}": coeff
                for name, coeff in zip(self.feature_names, self.model.coef_[0])
            }
            coeff_dict["intercept"] = self.model.intercept_[0]
            mlflow.log_params(coeff_dict)

            # Log feature importance (absolute coefficients)
            importance_dict = {
                f"importance_{name}": abs(coeff)
                for name, coeff in zip(self.feature_names, self.model.coef_[0])
            }
            mlflow.log_metrics(importance_dict)


def create_runs_predictor(
    experiment_name: str | None = None, model_name: str | None = None
) -> RunsPredictor:
    """Convenience function to create a RunsPredictor with default names."""
    from cricket.ml.config import get_model_config

    config = get_model_config("runs_model")

    return RunsPredictor(
        experiment_name=experiment_name or config["experiment_name"],
        model_name=model_name or config["model_name"],
    )


def create_wicket_predictor(
    experiment_name: str | None = None, model_name: str | None = None
) -> WicketPredictor:
    """Convenience function to create a WicketPredictor with default names."""
    from cricket.ml.config import get_model_config

    config = get_model_config("wicket_model")

    return WicketPredictor(
        experiment_name=experiment_name or config["experiment_name"],
        model_name=model_name or config["model_name"],
    )
