"""
Model evaluation and visualization for T20 cricket linear regression.

This module provides:
- Comprehensive model evaluation metrics
- Visualization of model performance
- Feature importance analysis
- Model diagnostics and validation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class T20ModelEvaluator:
    """Comprehensive evaluation for T20 linear regression model."""

    def __init__(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: Optional[list] = None,
    ):
        """
        Initialize model evaluator.

        Parameters
        ----------
        model : T20LinearRegression
            Trained model instance
        X_test : np.ndarray
            Test features
        y_test : np.ndarray
            Test targets
        feature_names : Optional[list]
            Feature column names
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names or [
            "current_score",
            "wickets_fallen",
            "overs_remaining",
        ]

        # Make predictions
        self.y_pred = self.model.predict(X_test)

        # Calculate residuals
        self.residuals = self.y_test - self.y_pred

        logger.info(f"Initialized T20ModelEvaluator with {len(y_test)} test samples")

    def calculate_comprehensive_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.

        Returns
        -------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        logger.info("Calculating comprehensive evaluation metrics")

        metrics = {}

        # Basic regression metrics
        metrics["r2_score"] = r2_score(self.y_test, self.y_pred)
        metrics["adjusted_r2"] = self._calculate_adjusted_r2()
        metrics["rmse"] = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        metrics["mae"] = mean_absolute_error(self.y_test, self.y_pred)
        metrics["mape"] = self._calculate_mape()

        # Additional metrics
        metrics["max_error"] = np.max(np.abs(self.residuals))
        metrics["mean_error"] = np.mean(self.residuals)
        metrics["std_residuals"] = np.std(self.residuals)

        # Percentage of predictions within certain ranges
        metrics["within_10_runs"] = np.mean(np.abs(self.residuals) <= 10) * 100
        metrics["within_20_runs"] = np.mean(np.abs(self.residuals) <= 20) * 100
        metrics["within_30_runs"] = np.mean(np.abs(self.residuals) <= 30) * 100

        # Model performance categories
        metrics["underestimate_rate"] = (
            np.mean(self.residuals > 0) * 100
        )  # Model predicts too low
        metrics["overestimate_rate"] = (
            np.mean(self.residuals < 0) * 100
        )  # Model predicts too high

        return metrics

    def _calculate_adjusted_r2(self) -> float:
        """Calculate adjusted R-squared."""
        n = len(self.y_test)
        p = self.X_test.shape[1]  # number of features
        r2 = r2_score(self.y_test, self.y_pred)

        adj_r2 = 1 - ((1 - r2) * (n - 1)) / (n - p - 1)
        return adj_r2

    def _calculate_mape(self) -> float:
        """Calculate Mean Absolute Percentage Error, handling division by zero."""
        with np.errstate(divide="ignore", invalid="ignore"):
            mape = np.mean(np.abs((self.y_test - self.y_pred) / self.y_test)) * 100

            # Handle inf/nan values
            if np.isfinite(mape):
                return mape
            else:
                return 0.0

    def plot_predictions_vs_actual(
        self, figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Create scatter plot of predictions vs actual values.

        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        logger.info("Creating predictions vs actual plot")

        fig, ax = plt.subplots(figsize=figsize)

        # Scatter plot
        ax.scatter(self.y_test, self.y_pred, alpha=0.6, s=50)

        # Perfect prediction line
        min_val = min(self.y_test.min(), self.y_pred.min())
        max_val = max(self.y_test.max(), self.y_pred.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            "r--",
            lw=2,
            label="Perfect Prediction",
        )

        # Add R² score to plot
        r2 = r2_score(self.y_test, self.y_pred)
        ax.text(
            0.05,
            0.95,
            f"R² = {r2:.3f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            fontsize=12,
            verticalalignment="top",
        )

        ax.set_xlabel("Actual Runs", fontsize=12)
        ax.set_ylabel("Predicted Runs", fontsize=12)
        ax.set_title(
            "T20 Runs Prediction: Actual vs Predicted", fontsize=14, fontweight="bold"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Make axes equal
        ax.set_aspect("equal", adjustable="box")

        plt.tight_layout()
        return fig

    def plot_residuals(self, figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Create residual analysis plots.

        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size

        Returns
        -------
        plt.Figure
            Matplotlib figure with subplots
        """
        logger.info("Creating residual analysis plots")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Residuals vs Predicted
        ax1.scatter(self.y_pred, self.residuals, alpha=0.6)
        ax1.axhline(y=0, color="r", linestyle="--", lw=2)
        ax1.set_xlabel("Predicted Runs")
        ax1.set_ylabel("Residuals")
        ax1.set_title("Residuals vs Predicted Values")
        ax1.grid(True, alpha=0.3)

        # Histogram of residuals
        ax2.hist(self.residuals, bins=30, alpha=0.7, edgecolor="black")
        ax2.axvline(x=0, color="r", linestyle="--", lw=2)
        ax2.set_xlabel("Residuals")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Distribution of Residuals")
        ax2.grid(True, alpha=0.3)

        # Add statistics to histogram
        mean_resid = np.mean(self.residuals)
        std_resid = np.std(self.residuals)
        ax2.text(
            0.02,
            0.98,
            f"Mean: {mean_resid:.2f}\\nStd: {std_resid:.2f}",
            transform=ax2.transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()
        return fig

    def plot_feature_importance(self, figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create feature importance plot based on absolute coefficients.

        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size

        Returns
        -------
        plt.Figure
            Matplotlib figure
        """
        logger.info("Creating feature importance plot")

        # Get feature importance
        importance_df = self.model.get_feature_importance()

        fig, ax = plt.subplots(figsize=figsize)

        # Create bar plot
        bars = ax.barh(importance_df["feature"], importance_df["abs_coefficient"])

        # Color bars based on positive/negative coefficients
        for i, (bar, coeff) in enumerate(zip(bars, importance_df["coefficient"])):
            if coeff >= 0:
                bar.set_color("green")
                bar.set_alpha(0.7)
            else:
                bar.set_color("red")
                bar.set_alpha(0.7)

        ax.set_xlabel("Absolute Coefficient Value")
        ax.set_title("T20 Linear Regression: Feature Importance", fontweight="bold")
        ax.grid(True, alpha=0.3)

        # Add coefficient values as text
        for i, (feature, coeff, abs_coeff) in importance_df.iterrows():
            ax.text(
                abs_coeff + 0.01 * max(importance_df["abs_coefficient"]),
                i,
                f"{coeff:.3f}",
                va="center",
                fontweight="bold",
            )

        plt.tight_layout()
        return fig

    def plot_error_analysis(self, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create comprehensive error analysis plots.

        Parameters
        ----------
        figsize : Tuple[int, int]
            Figure size

        Returns
        -------
        plt.Figure
            Matplotlib figure with multiple subplots
        """
        logger.info("Creating error analysis plots")

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)

        # 1. Absolute errors vs actual values
        abs_errors = np.abs(self.residuals)
        ax1.scatter(self.y_test, abs_errors, alpha=0.6)
        ax1.set_xlabel("Actual Runs")
        ax1.set_ylabel("Absolute Error")
        ax1.set_title("Absolute Errors vs Actual Values")
        ax1.grid(True, alpha=0.3)

        # Add trend line
        z = np.polyfit(self.y_test, abs_errors, 1)
        p = np.poly1d(z)
        ax1.plot(sorted(self.y_test), p(sorted(self.y_test)), "r--", alpha=0.8)

        # 2. Error distribution by score ranges
        score_ranges = [(0, 120), (120, 140), (140, 160), (160, 180), (180, 250)]
        range_errors = []
        range_labels = []

        for low, high in score_ranges:
            mask = (self.y_test >= low) & (self.y_test < high)
            if np.any(mask):
                range_errors.append(abs_errors[mask])
                range_labels.append(f"{low}-{high}")

        ax2.boxplot(range_errors, labels=range_labels)
        ax2.set_xlabel("Score Ranges")
        ax2.set_ylabel("Absolute Error")
        ax2.set_title("Error Distribution by Score Range")
        ax2.grid(True, alpha=0.3)

        # 3. Percentage error vs actual
        pct_errors = (self.residuals / self.y_test) * 100
        ax3.scatter(self.y_test, pct_errors, alpha=0.6)
        ax3.axhline(y=0, color="r", linestyle="--")
        ax3.set_xlabel("Actual Runs")
        ax3.set_ylabel("Percentage Error (%)")
        ax3.set_title("Percentage Errors vs Actual Values")
        ax3.grid(True, alpha=0.3)

        # 4. Cumulative error distribution
        sorted_abs_errors = np.sort(abs_errors)
        cumulative_pct = (
            np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors) * 100
        )

        ax4.plot(sorted_abs_errors, cumulative_pct, linewidth=2)
        ax4.set_xlabel("Absolute Error (Runs)")
        ax4.set_ylabel("Cumulative Percentage (%)")
        ax4.set_title("Cumulative Error Distribution")
        ax4.grid(True, alpha=0.3)

        # Add reference lines for 10, 20, 30 run errors
        for error_threshold in [10, 20, 30]:
            if error_threshold <= max(sorted_abs_errors):
                pct_below = np.mean(abs_errors <= error_threshold) * 100
                ax4.axvline(x=error_threshold, color="red", linestyle="--", alpha=0.7)
                ax4.text(
                    error_threshold,
                    pct_below + 5,
                    f"{pct_below:.1f}%",
                    rotation=90,
                    ha="right",
                )

        plt.tight_layout()
        return fig

    def create_model_report(self) -> Dict:
        """
        Create comprehensive model evaluation report.

        Returns
        -------
        Dict
            Comprehensive evaluation report
        """
        logger.info("Creating comprehensive model report")

        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics()

        # Feature importance
        importance_df = self.model.get_feature_importance()

        # Model equation
        equation = self.model.get_model_equation()

        # Performance summary
        performance_summary = {
            "overall_performance": "Good"
            if metrics["r2_score"] >= 0.7
            else "Fair"
            if metrics["r2_score"] >= 0.5
            else "Poor",
            "prediction_accuracy": f"{metrics['within_20_runs']:.1f}% within 20 runs",
            "bias_assessment": "Unbiased"
            if abs(metrics["mean_error"]) <= 2
            else "Biased",
            "key_insights": [],
        }

        # Generate insights
        if metrics["r2_score"] >= 0.8:
            performance_summary["key_insights"].append(
                "Model explains >80% of variance - excellent performance"
            )

        if metrics["within_20_runs"] >= 80:
            performance_summary["key_insights"].append(
                "High prediction accuracy with 80%+ within 20 runs"
            )

        if abs(metrics["mean_error"]) <= 1:
            performance_summary["key_insights"].append(
                "Model predictions are well-calibrated with minimal bias"
            )

        # Compile report
        report = {
            "model_performance": metrics,
            "feature_importance": importance_df.to_dict("records"),
            "model_equation": equation,
            "performance_summary": performance_summary,
            "data_summary": {
                "test_samples": len(self.y_test),
                "actual_score_range": f"{self.y_test.min():.0f}-{self.y_test.max():.0f}",
                "predicted_score_range": f"{self.y_pred.min():.0f}-{self.y_pred.max():.0f}",
                "mean_actual": f"{self.y_test.mean():.1f}",
                "mean_predicted": f"{self.y_pred.mean():.1f}",
            },
        }

        return report

    def save_plots(self, output_dir: str = "model_evaluation_plots") -> None:
        """
        Save all evaluation plots to disk.

        Parameters
        ----------
        output_dir : str
            Directory to save plots
        """
        logger.info(f"Saving evaluation plots to: {output_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Generate and save plots
        plots = {
            "predictions_vs_actual.png": self.plot_predictions_vs_actual(),
            "residual_analysis.png": self.plot_residuals(),
            "feature_importance.png": self.plot_feature_importance(),
            "error_analysis.png": self.plot_error_analysis(),
        }

        for filename, fig in plots.items():
            filepath = output_path / filename
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"Saved plot: {filepath}")

    def print_summary(self) -> None:
        """Print a summary of model performance."""
        metrics = self.calculate_comprehensive_metrics()
        importance_df = self.model.get_feature_importance()

        print("\\n" + "=" * 50)
        print("T20 LINEAR REGRESSION MODEL EVALUATION")
        print("=" * 50)

        print("\\nModel Performance:")
        print(f"  R² Score:           {metrics['r2_score']:.3f}")
        print(f"  Adjusted R²:        {metrics['adjusted_r2']:.3f}")
        print(f"  RMSE:              {metrics['rmse']:.1f} runs")
        print(f"  MAE:               {metrics['mae']:.1f} runs")
        print(f"  MAPE:              {metrics['mape']:.1f}%")

        print("\\nPrediction Accuracy:")
        print(f"  Within 10 runs:     {metrics['within_10_runs']:.1f}%")
        print(f"  Within 20 runs:     {metrics['within_20_runs']:.1f}%")
        print(f"  Within 30 runs:     {metrics['within_30_runs']:.1f}%")

        print("\\nModel Equation:")
        print(f"  {self.model.get_model_equation()}")

        print("\\nFeature Importance:")
        for _, row in importance_df.iterrows():
            print(
                f"  {row['feature']:15}: {row['coefficient']:7.3f} (|{row['abs_coefficient']:.3f}|)"
            )

        print("\\n" + "=" * 50)
