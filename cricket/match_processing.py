from typing import Dict, List, Optional

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
        match_metadata["dates"] = self.match_data.get("info", {}).get(
            "dates", None
        )
        match_metadata["teams"] = self.match_data.get("info", {}).get(
            "teams", None
        )
        match_metadata["toss"] = self.match_data.get("info", {}).get(
            "toss", None
        )
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
