import datetime
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, computed_field

from pathlib import Path
from cricket.extraction.data_processing import load_json
from cricket.extraction.innings_processing import Innings, InningsData


class TossData(BaseModel):
    """Toss information"""

    winner: Optional[str] = None
    decision: Optional[str] = None


class MatchInfo(BaseModel):
    """Match information from JSON"""

    match_type: Optional[str] = None
    city: Optional[str] = None
    venue: Optional[str] = None
    balls_per_over: Optional[int] = None
    gender: Optional[str] = None
    dates: Optional[List[str]] = None
    teams: Optional[List[str]] = None
    toss: Optional[Dict] = None
    registry: Dict


class MatchData(BaseModel):
    """Complete match data from JSON"""

    info: MatchInfo
    innings: List[Dict]


class Match(BaseModel):
    """
    Process a full match file into ball by ball data using Pydantic validation.

    Uses computed fields for derived data and validation for match processing.
    """

    match_filepath: Path
    match_id: Union[str, int]
    raw_data: Optional[MatchData] = None

    def __init__(self, match_filepath=None, match_id=None, **kwargs):
        """
        Initialize Match with file path and match ID.
        Maintains backward compatibility with existing API.
        """
        if match_filepath is not None:
            kwargs["match_filepath"] = (
                Path(match_filepath)
                if not isinstance(match_filepath, Path)
                else match_filepath
            )
        if match_id is not None:
            kwargs["match_id"] = match_id

        super().__init__(**kwargs)

    def model_post_init(self, __context) -> None:
        """Load and validate match data after model initialization"""
        if self.raw_data is None:
            json_data = load_json(self.match_filepath)
            self.raw_data = MatchData(**json_data)

    @computed_field
    @property
    def player_registry(self) -> Dict[str, str]:
        """Player registry from match data"""
        return self.raw_data.info.registry.get("people", {})

    @computed_field
    @property
    def match_dates(
        self,
    ) -> Tuple[Optional[datetime.datetime], Optional[datetime.datetime]]:
        """Start and end dates of the match"""
        if not self.raw_data.info.dates:
            return None, None
        dates_parsed = [
            datetime.datetime.strptime(date, "%Y-%m-%d")
            for date in self.raw_data.info.dates
        ]
        return min(dates_parsed), max(dates_parsed)

    @computed_field
    @property
    def match_teams(self) -> Tuple[Optional[str], Optional[str]]:
        """The two teams playing"""
        teams = self.raw_data.info.teams or []
        team1 = teams[0] if len(teams) > 0 else None
        team2 = teams[1] if len(teams) > 1 else None
        return team1, team2

    @computed_field
    @property
    def match_toss(self) -> Tuple[Optional[str], Optional[str]]:
        """Toss winner and decision"""
        if not self.raw_data.info.toss:
            return None, None
        toss_data = TossData(**self.raw_data.info.toss)
        return toss_data.winner, toss_data.decision

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
        Get the match metadata from the match data.

        Returns
        -------
        Dict
            A dictionary containing the metadata for the match
        """
        start_date, end_date = self.match_dates
        team1, team2 = self.match_teams
        toss_winner, toss_decision = self.match_toss

        return {
            "match_id": self.match_id,
            "match_type": self.raw_data.info.match_type,
            "city": self.raw_data.info.city,
            "venue": self.raw_data.info.venue,
            "balls_per_over": self.raw_data.info.balls_per_over,
            "gender": self.raw_data.info.gender,
            "start_date": start_date,
            "end_date": end_date,
            "team_1": team1,
            "team_2": team2,
            "toss_winner": toss_winner,
            "toss_decision": toss_decision,
        }

    def parse_match_data(self) -> List[Dict]:
        """
        Parse the match data into a list of dictionaries, each representing a ball in the match.

        Returns
        -------
        List[Dict]
            A list of dictionaries, each representing a ball in the match.
        """
        innings_data = []
        for innings_num, innings_raw in enumerate(self.raw_data.innings):
            # Skip forfeited innings
            if innings_raw.get("forfeited", False):
                continue

            innings_data_model = InningsData(**innings_raw)
            innings = Innings(
                raw_data=innings_data_model,
                innings_num=innings_num + 1,
            )
            innings_data.extend(innings.parse_innings_data())

        # Add match-level data to each ball
        for ball in innings_data:
            ball["match_id"] = self.match_id
            ball["batter_id"] = self.lookup_player(ball["batter"])
            ball["non_striker_id"] = self.lookup_player(ball["non_striker"])
            ball["bowler_id"] = self.lookup_player(ball["bowler"])

        return innings_data
