import polars as pl


def get_innings_team(dataframe: pl.DataFrame, innings_num: int) -> pl.DataFrame:
    team_name = dataframe.filter(pl.col("innings_number") == innings_num).item(0, "team")
    return team_name