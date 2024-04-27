from typing import Dict

import polars as pl

from cricket.over_processing import Over


class Innings:
    """
    Parse data about a single innings in a cricket match into a dictionary format.

    Takes some raw data in dictionary format and can parse various information about this innings,
    such as the team, the powerplays, and the target. It can also produce a dictionary of this
    information.

    Attributes
    ----------
    innings_data : Dict
        The raw data about the innings in dictionary format.
    innings_num : int
        The innings number of the innings in the match
    team : str
        The team batting in the innings
    powerplays : List
        The deliveries in which powerplays were active in this innings. Not supplied in class initialisation.
    target: Dict
        The target set for the innings, if it exists. Not supplied in class initialisation.
    """

    def __init__(self, innings_data: Dict, innings_num: int):
        self.innings_data = innings_data
        self.innings_num = innings_num
        self.team = self.innings_data["team"]
        self.powerplays = self.innings_data.get("powerplays", [])
        self.target = self.innings_data.get("target", None)
        self.innings_df = pl.DataFrame()

    def power_play_check(self) -> None:
        """
        Check if the delivery was in a powerplay and add this information to the ball dictionary.

        Parameters
        ----------
        ball : Dict
            The ball dictionary to add the powerplay information to.
        """
        self.innings_df = self.innings_df.with_columns(
            pl.lit(False).alias("powerplay")
        )
        for powerplay in self.powerplays:
            self.innings_df = self.innings_df.with_columns(
                pl.when(
                    (pl.col("delivery") >= powerplay["from"])
                    & (pl.col("delivery") <= powerplay["to"])
                )
                .then(True)
                .otherwise(False)
                .alias("powerplay")
            )

    def target_check(self) -> None:
        """
        Check if the delivery was in a powerplay and add this information to the ball dictionary.

        Parameters
        ----------
        ball : Dict
            The ball dictionary to add the powerplay information to.
        """
        if self.target is None:
            ball["target_runs"] = 0
            ball["target_overs"] = 0.0
        else:
            self.innings_df = self.innings_df.with_columns(
                [
                    pl.lit(self.target["runs"]).alias("target_runs"),
                    pl.lit(float(self.target["overs"])).alias("target_overs"),
                ]
            )

    def parse_innings_data(self) -> pl.DataFrame:
        """
        Parse the raw innings data into a list of ball dictionaries.

        Returns
        -------
        List
            A list of ball dictionaries containing the parsed data about each delivery in the innings.
        """
        for over_data in self.innings_data["overs"]:
            over = Over(over_data)
            over_data = over.parse_over_data()
            self.innings_df = pl.concat(
                [self.innings_df, over_data], how="diagonal"
            )

        self.innings_df = self.innings_df.with_columns(
            [
                pl.lit(self.team).alias("team"),
                pl.lit(self.innings_num).alias("innings_number"),
            ]
        )
        self.power_play_check()
        self.target_check()
        return self.innings_df
