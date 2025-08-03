import polars as pl

from cricket.transformation.utils import get_all_matches_data


def get_batter_outs(dataframe: pl.DataFrame) -> pl.DataFrame:
    return (
        dataframe.with_columns(
            pl.when(pl.col("batter") == pl.col("player_out_1"))
            .then(True)
            .when(pl.col("batter") == pl.col("player_out_2"))
            .then(True)
            .otherwise(False)
            .over("match_id")
            .alias("out")
        )
        .with_columns(
            pl.when(pl.col("batter") == pl.col("player_out_1"))
            .then(pl.col("kind_1"))
            .when(pl.col("batter") == pl.col("player_out_2"))
            .then(pl.col("kind_2"))
            .otherwise(None)
            .over("match_id")
            .alias("how_out")
        )
        .filter(pl.col("out") == True)  # noqa: E712
    )


def get_batter_scores(dataframe: pl.DataFrame) -> pl.DataFrame:
    return dataframe.group_by(["batter_id", "match_id", "innings_number"]).agg(
        pl.sum("batter_runs").alias("total_runs")
    )


def get_batter_balls_faced(dataframe: pl.DataFrame) -> pl.DataFrame:
    return (
        dataframe.filter(pl.col("wides") == 0)
        .group_by(["batter_id", "match_id", "innings_number"])
        .agg(pl.count("delivery").alias("balls_faced"))
    )


def get_fours(dataframe: pl.DataFrame) -> pl.DataFrame:
    return (
        dataframe.filter(pl.col("batter_runs") == 4)
        .group_by(["batter_id", "match_id", "innings_number"])
        .agg(pl.count("delivery").alias("fours"))
    )


def get_sixes(dataframe: pl.DataFrame) -> pl.DataFrame:
    return (
        dataframe.filter(pl.col("batter_runs") == 6)
        .group_by(["batter_id", "match_id", "innings_number"])
        .agg(pl.count("delivery").alias("sixes"))
    )


def join_all_batter_data(dataframe: pl.DataFrame) -> pl.DataFrame:
    type_data_sorted = dataframe.sort(["start_date", "innings_number", "delivery"])
    batter_outs = get_batter_outs(type_data_sorted)
    batter_scores = get_batter_scores(type_data_sorted)
    batter_balls_faced = get_batter_balls_faced(type_data_sorted)
    fours = get_fours(type_data_sorted)
    sixes = get_sixes(type_data_sorted)
    return (
        batter_scores.join(
            batter_balls_faced, on=["batter_id", "match_id", "innings_number"]
        )
        .join(batter_outs, on=["batter_id", "match_id", "innings_number"])
        .join(fours, on=["batter_id", "match_id", "innings_number"])
        .join(sixes, on=["batter_id", "match_id", "innings_number"])
    )


def get_batter_data(match_type: str) -> pl.DataFrame:
    RETURN_COLUMNS = [
        "batter_id",
        "match_id",
        "innings_number",
        "total_runs",
        "balls_faced",
        "out",
        "how_out",
        "fours",
        "sixes",
    ]
    all_match_data = get_all_matches_data(match_type)
    batter_data = join_all_batter_data(all_match_data)
    return batter_data.select(RETURN_COLUMNS)


def get_earliest_ball_faced(dataframe: pl.DataFrame) -> pl.DataFrame:
    return dataframe.group_by(["match_id", "batter_id", "innings_number"]).agg(
        pl.min("delivery").alias("earliest_batter_delivery")
    )


def get_earliest_ball_as_non_striker(dataframe: pl.DataFrame) -> pl.DataFrame:
    return (
        dataframe.group_by(["match_id", "non_striker_id", "innings_number"])
        .agg(pl.min("delivery").alias("earliest_non_striker_delivery"))
        .rename({"non_striker_id": "batter_id"})
    )


def get_batter_arrival(dataframe: pl.DataFrame) -> pl.DataFrame:
    earliest_batter_delivery = get_earliest_ball_faced(dataframe)
    earliest_non_striker_delivery = get_earliest_ball_as_non_striker(dataframe)
    batter_arrival = earliest_batter_delivery.join(
        earliest_non_striker_delivery,
        on=["match_id", "batter_id", "innings_number"],
        how="left",
    )
    batter_arrival = batter_arrival.with_columns(
        pl.when(
            pl.col("earliest_non_striker_delivery") < pl.col("earliest_batter_delivery")
        )
        .then(pl.col("earliest_non_striker_delivery"))
        .otherwise(pl.col("earliest_batter_delivery"))
        .alias("earliest_delivery")
    )
    batter_arrival = batter_arrival.sort(
        [
            "match_id",
            "innings_number",
            "earliest_delivery",
            "earliest_batter_delivery",
        ]
    ).with_columns(
        pl.col("batter_id")
        .cum_count()
        .over(["match_id", "innings_number"])
        .alias("batting_number")
    )
    return batter_arrival
