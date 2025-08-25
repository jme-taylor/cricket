"""
Match state feature calculations.

This module provides functions to calculate match state features like:
- Current score at each ball
- Wickets fallen at each ball
- Overs remaining in the innings
"""

import polars as pl


def get_current_score(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate cumulative score at each ball.
    Score represents the total BEFORE the current ball.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Ball-by-ball data with 'runs', 'match_id', 'innings_number' columns

    Returns
    -------
    pl.DataFrame
        Input dataframe with additional 'current_score' column
    """
    # Sort by match, innings, and delivery to ensure correct order for cumulative calculations
    sorted_df = dataframe.sort(["match_id", "innings_number", "delivery"])

    # Calculate cumulative score within each match-innings group
    result = sorted_df.with_columns(
        pl.col("runs")
        .cum_sum()
        .shift(1, fill_value=0)
        .over(["match_id", "innings_number"])
        .alias("current_score")
    )

    # Return in original order if needed (maintain original row order)
    # For now, return sorted order as it's more logical for match analysis
    return result


def get_wickets_fallen(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate cumulative wickets fallen at each ball.
    Handles both player_out_1 and player_out_2 for double wickets.
    Wickets represent total fallen BEFORE the current ball.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Ball-by-ball data with 'player_out_1', 'player_out_2', 'match_id', 'innings_number' columns

    Returns
    -------
    pl.DataFrame
        Input dataframe with additional 'wickets_fallen' column
    """
    # Sort by match, innings, and delivery to ensure correct order
    sorted_df = dataframe.sort(["match_id", "innings_number", "delivery"])

    # Create wicket indicator columns
    df = sorted_df.with_columns(
        [
            (pl.col("player_out_1") != "").cast(pl.Int32).alias("wicket_1"),
            (pl.col("player_out_2") != "")
            .cast(pl.Int32)
            .fill_null(0)
            .alias("wicket_2"),
        ]
    )

    # Sum wickets per ball and calculate cumulative
    df = df.with_columns(
        (pl.col("wicket_1") + pl.col("wicket_2")).alias("wickets_this_ball")
    )

    # Calculate cumulative wickets (before current ball) within each innings
    df = df.with_columns(
        pl.col("wickets_this_ball")
        .cum_sum()
        .shift(1, fill_value=0)
        .over(["match_id", "innings_number"])
        .alias("wickets_fallen")
    )

    # Cap at 10 wickets (all out)
    df = df.with_columns(
        pl.when(pl.col("wickets_fallen") > 10)
        .then(10)
        .otherwise(pl.col("wickets_fallen"))
        .alias("wickets_fallen")
    )

    # Clean up temporary columns
    return df.drop(["wicket_1", "wicket_2", "wickets_this_ball"])


def get_overs_remaining(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate overs remaining in the innings at each ball.
    Accounts for match format and D/L adjustments.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Ball-by-ball data with 'match_type', 'target_overs', 'wides', 'noballs',
        'match_id', 'innings_number' columns

    Returns
    -------
    pl.DataFrame
        Input dataframe with additional 'overs_remaining' column
    """
    # Sort by match, innings, and delivery to ensure correct order
    sorted_df = dataframe.sort(["match_id", "innings_number", "delivery"])

    # Map match type to total overs
    df = sorted_df.with_columns(
        pl.when(pl.col("match_type").is_in(["T20", "IT20"]))
        .then(20.0)
        .when(pl.col("match_type").is_in(["ODI", "ODM"]))
        .then(50.0)
        .when(pl.col("match_type") == "MDM")
        .then(40.0)
        .when(pl.col("match_type") == "The Hundred")
        .then(16.67)
        .when(pl.col("match_type") == "Test")
        .then(None)  # Will be handled later
        .otherwise(20.0)  # Default to T20
        .alias("total_overs_available")
    )

    # Use target_overs if available (D/L scenarios), otherwise use format default
    df = df.with_columns(
        pl.when(pl.col("target_overs") > 0)
        .then(pl.col("target_overs"))
        .otherwise(pl.col("total_overs_available"))
        .alias("innings_overs")
    )

    # Calculate legal deliveries (exclude wides and no-balls from over count)
    df = df.with_columns(
        pl.when((pl.col("wides") > 0) | (pl.col("noballs") > 0))
        .then(0)
        .otherwise(1)
        .alias("legal_delivery")
    )

    # Calculate cumulative legal deliveries within each innings
    df = df.with_columns(
        pl.col("legal_delivery")
        .cum_sum()
        .over(["match_id", "innings_number"])
        .alias("balls_bowled")
    )

    # Convert to overs and calculate remaining
    df = df.with_columns(
        (pl.col("innings_overs") - (pl.col("balls_bowled") / 6.0)).alias(
            "overs_remaining"
        )
    )

    # Handle Test matches (unlimited overs) - set to null
    df = df.with_columns(
        pl.when(pl.col("match_type") == "Test")
        .then(None)
        .otherwise(pl.col("overs_remaining"))
        .alias("overs_remaining")
    )

    # Clean up temporary columns
    return df.drop(
        ["total_overs_available", "innings_overs", "legal_delivery", "balls_bowled"]
    )


def get_balls_remaining(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate balls remaining in the innings at each ball.
    Converts overs_remaining to balls (overs * 6).

    Parameters
    ----------
    dataframe : pl.DataFrame
        Ball-by-ball data with 'overs_remaining' column

    Returns
    -------
    pl.DataFrame
        Input dataframe with additional 'balls_remaining' column
    """
    return dataframe.with_columns(
        (pl.col("overs_remaining") * 6.0).alias("balls_remaining")
    )


def get_current_run_rate(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate current run rate (runs per over) at each ball.
    Rate calculated from runs scored before current ball and balls faced.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Ball-by-ball data with 'current_score' and 'overs_remaining' columns

    Returns
    -------
    pl.DataFrame
        Input dataframe with additional 'current_run_rate' column
    """
    # Calculate overs completed (20 - overs_remaining for T20)
    df = dataframe.with_columns(
        pl.when(pl.col("match_type").is_in(["T20", "IT20"]))
        .then(20.0 - pl.col("overs_remaining"))
        .when(pl.col("match_type").is_in(["ODI", "ODM"]))
        .then(50.0 - pl.col("overs_remaining"))
        .otherwise(20.0 - pl.col("overs_remaining"))  # Default T20
        .alias("overs_completed")
    )

    # Calculate run rate (avoid division by zero)
    df = df.with_columns(
        pl.when(pl.col("overs_completed") > 0)
        .then(pl.col("current_score") / pl.col("overs_completed"))
        .otherwise(0.0)
        .alias("current_run_rate")
    )

    return df.drop("overs_completed")


def get_required_run_rate(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate required run rate for second innings (chasing team).
    Set to NULL for first innings.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Ball-by-ball data with match state columns

    Returns
    -------
    pl.DataFrame
        Input dataframe with additional 'required_run_rate' column
    """
    # For now, set to NULL for all innings
    # TODO: Implement target calculation for second innings
    return dataframe.with_columns(
        pl.when(pl.col("innings_number") == 2)
        .then(None)  # Will calculate later when target data available
        .otherwise(None)
        .alias("required_run_rate")
    )


def get_innings_indicator(dataframe: pl.DataFrame) -> pl.DataFrame:
    """
    Create binary indicator for first innings.
    1 = first innings, 0 = second innings.

    Parameters
    ----------
    dataframe : pl.DataFrame
        Ball-by-ball data with 'innings_number' column

    Returns
    -------
    pl.DataFrame
        Input dataframe with additional 'is_first_innings' column
    """
    return dataframe.with_columns(
        (pl.col("innings_number") == 1).cast(pl.Int32).alias("is_first_innings")
    )
