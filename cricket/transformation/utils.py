import polars as pl

from cricket.constants import DATA_FOLDER


def get_ball_level_data() -> pl.DataFrame:
    return pl.read_parquet(DATA_FOLDER.joinpath("ball_level_data.parquet"))


def get_match_metadata() -> pl.DataFrame:
    return pl.read_parquet(DATA_FOLDER.joinpath("match_metadata.parquet"))


def join_to_metadata(
    ball_level_data: pl.DataFrame, match_metadata: pl.DataFrame
) -> pl.DataFrame:
    return ball_level_data.join(match_metadata, on="match_id")


def filter_by_type(dataframe: pl.DataFrame, match_type: list[str]) -> pl.DataFrame:
    return dataframe.filter(pl.col("match_type").is_in(match_type))


def get_all_matches_data() -> pl.DataFrame:
    ball_level_data = get_ball_level_data()
    match_metadata = get_match_metadata()
    all_match_data = join_to_metadata(ball_level_data, match_metadata)
    return all_match_data
