import datetime
from typing import Dict, List, Optional

import polars as pl

from cricket.data_processing import load_json
from cricket.innings_processing import Innings


class Match:
    """
    Process a full match file into ball by ball data. Takes an input of a filepath, and an ID for the match,
    and will parse the match data into a list of dictionaries, each representing a ball in the match. Will also
    lookup player names against the registry to provide a player ID for players involved in each delivery.
    """

    def __init__(self, match_filepath, match_id) -> None:
        self.match_filepath = match_filepath
        self.match_data = load_json(self.match_filepath)
        self.match_id = match_id
        self.player_registry = self.match_data["info"]["registry"]["people"]

    def lookup_player(self, player_name: str) -> Optional[str]:
        """
        Lookup a player name against the matches player registry to return a player ID.
        If no match is found, the return value will be None.

        Parameters
        ----------
        player_name : str
            The player name to lookup against the player registry.

        Returns
        -------
        Optional[str]
            String with player ID if there is a match, None if there is no match.
        """
        return self.player_registry.get(player_name, None)

    def get_match_metadata(self) -> pl.DataFrame:
        """
        Get the match metadata from the match data. This will return a dictionary containing the metadata
        for the match in question. This will include the match ID, match type, city, venue, balls per over,
        gender, dates, teams and toss. All of these fields are optional, and will return None if they are not
        present in the match data.

        Returns
        -------
        match_metadata : Dict
            A dictionary containing the metadata for the match
        """
        match_metadata = {}
        match_metadata["match_id"] = self.match_id
        match_metadata["match_type"] = self.match_data.get("info", {}).get(
            "match_type", None
        )
        match_metadata["city"] = self.match_data.get("info", {}).get(
            "city", None
        )
        match_metadata["venue"] = self.match_data.get("info", {}).get(
            "venue", None
        )
        match_metadata["balls_per_over"] = self.match_data.get("info", {}).get(
            "balls_per_over", None
        )
        match_metadata["gender"] = self.match_data.get("info", {}).get(
            "gender", None
        )
        dates = self.match_data.get("info", {}).get("dates", [])
        dates_parsed = [
            datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates
        ]
        match_metadata["start_date"] = min(dates_parsed).strftime("%Y-%m-%d")
        match_metadata["end_date"] = max(dates_parsed).strftime("%Y-%m-%d")
        teams = self.match_data.get("info", {}).get("teams", [])
        try:
            match_metadata["team_1"] = teams[0]
        except IndexError:
            match_metadata["team_1"] = None
        try:
            match_metadata["team_2"] = teams[1]
        except IndexError:
            match_metadata["team_2"] = None
        toss = self.match_data.get("info", {}).get("toss", {})
        match_metadata["toss_winner"] = toss.get("winner", None)
        match_metadata["toss_decision"] = toss.get("decision", None)
        return pl.from_dict(match_metadata)

    def parse_match_data(self) -> List[Dict]:
        """
        Parse the match data into a list of dictionaries, each representing a ball in the match.

        Returns
        -------
        List[Dict]
            A list of dictionaries, each representing a ball in the match.
        """
        match_data = pl.DataFrame()
        for innings_num, innings_raw in enumerate(self.match_data["innings"]):
            innings = Innings(
                innings_raw,
                innings_num + 1,
            )
            forfeit = innings_raw.get("forfeited", False)

            if forfeit:
                continue
            match_data = pl.concat(
                [match_data, innings.parse_innings_data()], how="diagonal"
            )

        match_data = match_data.with_columns(
            pl.lit(self.match_id).alias("match_id")
        )
        return match_data


from cricket.constants import PROJECT_ROOT

DATA_DIR = PROJECT_ROOT.joinpath("tests").joinpath("test_match_processing")

test_match = Match(
    DATA_DIR.joinpath("innings_forfeit_match_input.json"), match_id=5
)
test_match_data = test_match.parse_match_data()
test_match_data.write_parquet(
    DATA_DIR.joinpath("innings_forfeit_match_output.parquet")
)
test_match_metadata = test_match.get_match_metadata()
test_match_metadata.write_parquet(
    DATA_DIR.joinpath("innings_forfeit_match_metadata.parquet")
)
