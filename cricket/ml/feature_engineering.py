"""
Feature engineering module for T20 cricket machine learning.

This module handles:
- Feature selection and validation
- Feature scaling and normalization
- Feature transformation and encoding
- Missing value handling
"""

import polars as pl
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineer and prepare features for T20 linear regression."""

    def __init__(
        self,
        scaling_method: str = "standard",
        feature_columns: Optional[List[str]] = None,
    ):
        """
        Initialize feature engineer.

        Parameters
        ----------
        scaling_method : str
            Scaling method: 'standard', 'minmax', or 'none'
        feature_columns : Optional[List[str]]
            Specific columns to use as features
        """
        self.scaling_method = scaling_method
        self.feature_columns = feature_columns or [
            "current_score",
            "wickets_fallen",
            "balls_remaining",
            "current_run_rate",
            "required_run_rate",
            "is_first_innings",
        ]

        # Initialize scaler based on method
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_method == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaling method: {scaling_method}")

        self._scaler_fitted = False
        logger.info(f"Initialized FeatureEngineer with {scaling_method} scaling")

    def validate_features(self, df: pl.DataFrame) -> None:
        """
        Validate that required feature columns exist and have valid data.

        Parameters
        ----------
        df : pl.DataFrame
            Data to validate

        Raises
        ------
        ValueError
            If required features are missing or invalid
        """
        logger.info("Validating feature columns")

        # Check for missing columns
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")

        # Check for null values
        for col in self.feature_columns:
            null_count = df.select(pl.col(col).is_null().sum()).item()
            if null_count > 0:
                logger.warning(f"Column '{col}' has {null_count} null values")

        # Check for infinite values
        for col in self.feature_columns:
            if df.schema[col] in [pl.Float32, pl.Float64]:
                inf_count = df.select(pl.col(col).is_infinite().sum()).item()
                if inf_count > 0:
                    logger.warning(f"Column '{col}' has {inf_count} infinite values")

        logger.info("Feature validation completed")

    def prepare_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Prepare and clean features for modeling.

        Parameters
        ----------
        df : pl.DataFrame
            Raw data with features

        Returns
        -------
        pl.DataFrame
            Cleaned feature dataframe
        """
        logger.info("Preparing features for modeling")

        # Validate features first
        self.validate_features(df)

        # Select feature columns (sample_over is optional for all-balls mode)
        base_columns = ["match_id", "innings_number"]
        if "sample_over" in df.columns:
            base_columns.append("sample_over")

        feature_df = df.select(base_columns + self.feature_columns)

        # Handle missing values
        feature_df = self._handle_missing_values(feature_df)

        # Handle infinite values
        feature_df = self._handle_infinite_values(feature_df)

        # Apply feature transformations
        feature_df = self._apply_transformations(feature_df)

        logger.info(f"Prepared {len(feature_df)} feature samples")

        return feature_df

    def _handle_missing_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle missing values in feature columns."""
        logger.info("Handling missing values")

        # Fill nulls with sensible defaults
        fill_values = {
            "current_score": 0,  # No runs scored
            "wickets_fallen": 0,  # No wickets fallen
            "balls_remaining": 0,  # No balls remaining (match over)
            "current_run_rate": 0.0,  # No run rate at start
            "required_run_rate": None,  # Keep as null for first innings
            "is_first_innings": 1,  # Default to first innings
        }

        for col in self.feature_columns:
            if col in fill_values:
                df = df.with_columns(pl.col(col).fill_null(fill_values[col]))

        return df

    def _handle_infinite_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """Handle infinite values in feature columns."""
        logger.info("Handling infinite values")

        for col in self.feature_columns:
            if df.schema[col] in [pl.Float32, pl.Float64]:
                # Replace infinite values with column max/min
                col_max = df.select(
                    pl.col(col).filter(pl.col(col).is_finite()).max()
                ).item()
                col_min = df.select(
                    pl.col(col).filter(pl.col(col).is_finite()).min()
                ).item()

                df = df.with_columns(
                    pl.when(pl.col(col).is_infinite() & (pl.col(col) > 0))
                    .then(col_max)
                    .when(pl.col(col).is_infinite() & (pl.col(col) < 0))
                    .then(col_min)
                    .otherwise(pl.col(col))
                    .alias(col)
                )

        return df

    def _apply_transformations(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply feature transformations if needed."""
        logger.info("Applying feature transformations")

        # For now, no additional transformations beyond scaling
        # Future enhancements could include:
        # - Log transformations for skewed features
        # - Polynomial features
        # - Interaction terms

        return df

    def fit_scaler(self, X: Union[pl.DataFrame, np.ndarray]) -> "FeatureEngineer":
        """
        Fit the scaler on training data.

        Parameters
        ----------
        X : Union[pl.DataFrame, np.ndarray]
            Training features

        Returns
        -------
        FeatureEngineer
            Self for method chaining
        """
        if self.scaler is None:
            logger.info("No scaling method specified, skipping scaler fitting")
            return self

        logger.info(f"Fitting {self.scaling_method} scaler")

        # Convert to numpy if needed
        if isinstance(X, pl.DataFrame):
            X_array = X.select(self.feature_columns).to_numpy()
        else:
            X_array = X

        self.scaler.fit(X_array)
        self._scaler_fitted = True

        logger.info("Scaler fitted successfully")
        return self

    def transform_features(self, X: Union[pl.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform features using fitted scaler.

        Parameters
        ----------
        X : Union[pl.DataFrame, np.ndarray]
            Features to transform

        Returns
        -------
        np.ndarray
            Scaled features
        """
        if self.scaler is None:
            logger.info("No scaling method specified, returning raw features")
            if isinstance(X, pl.DataFrame):
                return X.select(self.feature_columns).to_numpy()
            else:
                return X

        if not self._scaler_fitted:
            raise ValueError("Scaler has not been fitted. Call fit_scaler() first.")

        logger.info(f"Transforming features using {self.scaling_method} scaler")

        # Convert to numpy if needed
        if isinstance(X, pl.DataFrame):
            X_array = X.select(self.feature_columns).to_numpy()
        else:
            X_array = X

        X_scaled = self.scaler.transform(X_array)

        logger.info(f"Transformed {len(X_scaled)} feature samples")
        return X_scaled

    def fit_transform(self, X: Union[pl.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Fit scaler and transform features in one step.

        Parameters
        ----------
        X : Union[pl.DataFrame, np.ndarray]
            Training features

        Returns
        -------
        np.ndarray
            Scaled features
        """
        logger.info("Fitting scaler and transforming features")
        self.fit_scaler(X)
        return self.transform_features(X)

    def get_feature_names(self) -> List[str]:
        """
        Get list of feature column names.

        Returns
        -------
        List[str]
            Feature column names
        """
        return self.feature_columns.copy()

    def get_feature_stats(self, df: pl.DataFrame) -> dict:
        """
        Get descriptive statistics for features.

        Parameters
        ----------
        df : pl.DataFrame
            Data with features

        Returns
        -------
        dict
            Feature statistics
        """
        logger.info("Calculating feature statistics")

        stats = {}

        for col in self.feature_columns:
            if col in df.columns:
                col_stats = df.select(
                    [
                        pl.col(col).count().alias("count"),
                        pl.col(col).mean().alias("mean"),
                        pl.col(col).std().alias("std"),
                        pl.col(col).min().alias("min"),
                        pl.col(col).quantile(0.25).alias("q25"),
                        pl.col(col).median().alias("median"),
                        pl.col(col).quantile(0.75).alias("q75"),
                        pl.col(col).max().alias("max"),
                        pl.col(col).is_null().sum().alias("nulls"),
                    ]
                ).to_dicts()[0]

                stats[col] = col_stats

        return stats

    def create_feature_correlation_matrix(self, df: pl.DataFrame) -> dict:
        """
        Calculate correlation matrix for features.

        Parameters
        ----------
        df : pl.DataFrame
            Data with features

        Returns
        -------
        dict
            Correlation matrix as nested dictionary
        """
        logger.info("Calculating feature correlation matrix")

        # Convert to pandas for correlation calculation

        feature_df = df.select(self.feature_columns).to_pandas()

        correlation_matrix = feature_df.corr()

        # Convert to nested dictionary
        corr_dict = {}
        for i, col1 in enumerate(self.feature_columns):
            corr_dict[col1] = {}
            for j, col2 in enumerate(self.feature_columns):
                corr_dict[col1][col2] = correlation_matrix.iloc[i, j]

        return corr_dict
