import polars as pl
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class T20DataPreparator:
    """Prepare T20/IT20 data for linear regression modeling."""

    def __init__(self):
        """
        Initialize data preparator for all-balls training.
        """
        # Only require columns needed for filtering - match state features will be added later
        self.required_columns = [
            "match_id",
            "innings_number",
            "match_type",
            "gender",
            "runs",
            "delivery",
            "team",
        ]
        # These columns will be added by the transformation layer
        self.match_state_columns = [
            "current_score",
            "wickets_fallen",
            "overs_remaining",
        ]

    def filter_t20_matches(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Filter for T20/IT20 male matches with complete data.

        Parameters
        ----------
        df : pl.DataFrame
            Raw cricket data with match information

        Returns
        -------
        pl.DataFrame
            Filtered dataframe with only T20/IT20 male matches
        """
        logger.info("Filtering for T20/IT20 male matches")

        # Validate required columns exist
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Apply filters
        filtered_df = df.filter(
            (pl.col("match_type").is_in(["T20", "IT20"]))
            & (pl.col("gender") == "male")
            & (pl.col("runs").is_not_null())
            & (pl.col("delivery").is_not_null())
        )

        # Log filtering results
        original_matches = df.select("match_id").n_unique()
        filtered_matches = filtered_df.select("match_id").n_unique()

        logger.info(f"Filtered from {original_matches} to {filtered_matches} matches")
        logger.info(f"Total balls: {len(filtered_df)}")

        return filtered_df

    def validate_match_state_features(self, df: pl.DataFrame) -> None:
        """
        Validate that match state features are present in the data.

        Parameters
        ----------
        df : pl.DataFrame
            Data to validate

        Raises
        ------
        ValueError
            If required match state features are missing
        """
        missing_features = [
            col for col in self.match_state_columns if col not in df.columns
        ]
        if missing_features:
            raise ValueError(
                f"Missing required match state features: {missing_features}"
            )
        logger.info("All required match state features are present")

    def create_target_variable(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Create innings-level target variable (total runs scored).

        Parameters
        ----------
        df : pl.DataFrame
            Ball-by-ball data

        Returns
        -------
        pl.DataFrame
            Innings-level data with target variable
        """
        logger.info("Creating innings-level target variables")

        # Create target by summing runs per innings
        target_df = df.group_by(["match_id", "innings_number"]).agg(
            [
                pl.col("runs").sum().alias("total_runs_innings"),
                pl.col("match_type").first(),
                pl.col("gender").first(),
                pl.col("team").first(),
                pl.col("delivery").count().alias("balls_faced"),
            ]
        )

        # Filter complete innings (minimum 60 balls = 10 overs)
        target_df = target_df.filter(pl.col("balls_faced") >= 60)

        logger.info(f"Created {len(target_df)} innings targets")
        logger.info(
            f"Average runs per innings: {target_df['total_runs_innings'].mean():.1f}"
        )

        return target_df

    def prepare_all_balls_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Prepare all balls as training samples with their innings totals.
        Each ball becomes a training example: (current_state) → innings_total.

        Parameters
        ----------
        df : pl.DataFrame
            Ball-by-ball data with match state features

        Returns
        -------
        pl.DataFrame
            All balls with innings totals joined as targets
        """
        logger.info("Preparing all balls as training data")

        # Validate match state features are present
        self.validate_match_state_features(df)

        # Create innings totals
        innings_totals = df.group_by(["match_id", "innings_number"]).agg(
            [
                pl.col("runs").sum().alias("total_runs_innings"),
                pl.col("delivery").count().alias("balls_faced"),
                pl.col("team").first(),
                pl.col("match_type").first(),
                pl.col("gender").first(),
            ]
        )

        # Filter for complete innings (minimum 60 balls = 10 overs)
        complete_innings = innings_totals.filter(pl.col("balls_faced") >= 60)

        logger.info(
            f"Found {len(complete_innings)} complete innings out of {len(innings_totals)} total"
        )

        # Join back to ball data to get all balls with their innings totals
        balls_with_targets = df.join(
            complete_innings.select(
                ["match_id", "innings_number", "total_runs_innings"]
            ),
            on=["match_id", "innings_number"],
            how="inner",
        )

        # Select relevant columns for modeling
        modeling_data = balls_with_targets.select(
            [
                "match_id",
                "innings_number",
                "delivery",
                "current_score",
                "wickets_fallen",
                "overs_remaining",
                "team",
                "match_type",
                "gender",
                "total_runs_innings",
            ]
        )

        logger.info(
            f"Created {len(modeling_data)} ball-level training samples from {len(complete_innings)} complete innings"
        )
        logger.info(
            f"Average samples per innings: {len(modeling_data) / len(complete_innings):.1f}"
        )

        return modeling_data

    def validate_data_quality(self, df: pl.DataFrame) -> dict:
        """
        Validate data quality and return summary statistics.

        Parameters
        ----------
        df : pl.DataFrame
            Data to validate

        Returns
        -------
        dict
            Data quality summary
        """
        logger.info("Validating data quality")

        quality_summary = {
            "total_rows": len(df),
            "unique_matches": df.select("match_id").n_unique(),
            "null_counts": {},
            "value_ranges": {},
            "duplicate_rows": df.is_duplicated().sum(),
        }

        # Check for null values in key columns
        key_columns = [
            "match_id",
            "innings_number",
            "runs",
            "current_score",
            "wickets_fallen",
            "overs_remaining",
        ]

        for col in key_columns:
            if col in df.columns:
                null_count = df.select(pl.col(col).is_null().sum()).item()
                quality_summary["null_counts"][col] = null_count

        # Check value ranges for numeric columns
        numeric_columns = ["runs", "current_score", "wickets_fallen", "overs_remaining"]
        for col in numeric_columns:
            if col in df.columns:
                col_stats = df.select(
                    [
                        pl.col(col).min().alias("min"),
                        pl.col(col).max().alias("max"),
                        pl.col(col).mean().alias("mean"),
                    ]
                ).to_dicts()[0]
                quality_summary["value_ranges"][col] = col_stats

        # Log quality summary
        logger.info(f"Data quality summary: {quality_summary}")

        return quality_summary

    def split_data_chronologically(
        self,
        df: pl.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Split data chronologically based on match dates.

        Parameters
        ----------
        df : pl.DataFrame
            Data to split
        train_ratio : float
            Proportion for training data
        val_ratio : float
            Proportion for validation data
        test_ratio : float
            Proportion for test data

        Returns
        -------
        tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
            Training, validation, and test datasets
        """
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Data split ratios must sum to 1.0")

        logger.info(
            f"Splitting data chronologically: train={train_ratio}, val={val_ratio}, test={test_ratio}"
        )

        # Sort by match_id (assuming match_ids are chronological)
        sorted_df = df.sort("match_id")

        # Calculate split indices
        n_total = len(sorted_df)
        train_end = int(n_total * train_ratio)
        val_end = int(n_total * (train_ratio + val_ratio))

        # Split data
        train_df = sorted_df[:train_end]
        val_df = sorted_df[train_end:val_end]
        test_df = sorted_df[val_end:]

        logger.info(
            f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
        )

        return train_df, val_df, test_df


def load_cricket_data(data_path: str) -> pl.DataFrame:
    """
    Load cricket data from parquet file and join with match metadata.

    Parameters
    ----------
    data_path : str
        Path to the cricket data file

    Returns
    -------
    pl.DataFrame
        Loaded cricket data with match metadata joined
    """
    logger.info(f"Loading cricket data from: {data_path}")

    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load ball-level data
    df = pl.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    # Load match metadata and join
    metadata_path = Path(data_path).parent / "match_metadata.parquet"
    if metadata_path.exists():
        logger.info(f"Joining with match metadata from: {metadata_path}")
        metadata_df = pl.read_parquet(metadata_path)

        # Join on match_id to add match_type and gender
        df = df.join(
            metadata_df.select(["match_id", "match_type", "gender"]),
            on="match_id",
            how="left",
        )
        logger.info(f"Joined data now has {len(df.columns)} columns")
    else:
        logger.warning(f"Match metadata file not found at: {metadata_path}")

    return df
