import datetime
from typing import Dict, List, Optional, Tuple

from cricket.extraction.data_processing import load_json
from cricket.extraction.innings_processing import Innings


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

    def get_match_dates(self) -> Tuple:
        dates = self.match_data.get("info", {}).get("dates", None)
        dates_parsed = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]
        start_date = min(dates_parsed)
        end_date = max(dates_parsed)
        return start_date, end_date

    def get_match_teams(self) -> Tuple:
        teams_raw = self.match_data.get("info", {}).get("teams", [])
        team_parsed = []
        for team_num in [0, 1]:
            try:
                team = teams_raw[team_num]
            except IndexError:
                team = None
            team_parsed.append(team)
        return team_parsed[0], team_parsed[1]

    def get_match_toss(self) -> Tuple:
        toss_data = self.match_data.get("info", {}).get("toss", {})
        toss_winner = toss_data.get("winner", None)
        toss_decision = toss_data.get("decision", None)
        return toss_winner, toss_decision

    def get_match_metadata(self) -> Dict:
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
        match_metadata["city"] = self.match_data.get("info", {}).get("city", None)
        match_metadata["venue"] = self.match_data.get("info", {}).get("venue", None)
        match_metadata["balls_per_over"] = self.match_data.get("info", {}).get(
            "balls_per_over", None
        )
        match_metadata["gender"] = self.match_data.get("info", {}).get("gender", None)
        (
            match_metadata["start_date"],
            match_metadata["end_date"],
        ) = self.get_match_dates()
        (
            match_metadata["team_1"],
            match_metadata["team_2"],
        ) = self.get_match_teams()
        (
            match_metadata["toss_winner"],
            match_metadata["toss_decision"],
        ) = self.get_match_toss()
        return match_metadata

    def parse_match_data(self) -> List[Dict]:
        """
        Parse the match data into a list of dictionaries, each representing a ball in the match.

        Returns
        -------
        List[Dict]
            A list of dictionaries, each representing a ball in the match.
        """
        innings_data = []
        for innings_num, innings_raw in enumerate(self.match_data["innings"]):
            innings = Innings(
                innings_raw,
                innings_num + 1,
            )
            forfeit = innings_raw.get("forfeited", False)

            if forfeit:
                continue
            innings_data.extend(innings.parse_innings_data())
        for ball in innings_data:
            ball["match_id"] = self.match_id
            ball["batter_id"] = self.lookup_player(ball["batter"])
            ball["non_striker_id"] = self.lookup_player(ball["non_striker"])
            ball["bowler_id"] = self.lookup_player(ball["bowler"])
        return innings_data
