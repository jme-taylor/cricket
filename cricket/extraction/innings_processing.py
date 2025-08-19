from typing import Dict, List, Optional
from pydantic import BaseModel, computed_field, ConfigDict, Field

from cricket.extraction.over_processing import Over, OverData


class PowerplayData(BaseModel):
    """Powerplay period data"""

    model_config = ConfigDict(populate_by_name=True)

    from_: float | None = Field(
        None, alias="from"
    )  # Using from_ because 'from' is a Python keyword
    to: float


class TargetData(BaseModel):
    """Target data for the innings"""

    runs: int
    overs: float


class InningsData(BaseModel):
    """Raw innings data from JSON"""

    team: str
    overs: List[Dict]
    powerplays: Optional[List[Dict]] = None
    target: Optional[Dict] = None


class Innings(BaseModel):
    """
    Parse data about a single innings in a cricket match using Pydantic validation.

    Uses computed fields to handle team data and methods for powerplay/target checks.
    """

    raw_data: InningsData
    innings_num: int

    def __init__(
        self,
        raw_data: dict | InningsData | None = None,
        innings_num: int | None = None,
        **kwargs,
    ):
        """
        Initialize Innings with either dictionary or InningsData.
        Maintains backward compatibility with existing API.
        """
        if raw_data is not None and not isinstance(raw_data, InningsData):
            # Convert dict to InningsData for backward compatibility
            raw_data = InningsData(**raw_data)  # ty: ignore

        if raw_data is not None:
            kwargs["raw_data"] = raw_data
        if innings_num is not None:
            kwargs["innings_num"] = innings_num

        super().__init__(**kwargs)

    @computed_field
    @property
    def team(self) -> str:
        """The team batting in the innings"""
        return self.raw_data.team

    @computed_field
    @property
    def powerplays(self) -> List[PowerplayData]:
        """List of powerplay periods"""
        if not self.raw_data.powerplays:
            return []
        return [PowerplayData(**pp) for pp in self.raw_data.powerplays]  # ty: ignore

    @computed_field
    @property
    def target(self) -> Optional[TargetData]:
        """Target for the innings if it exists"""
        if not self.raw_data.target:
            return None
        return TargetData(**self.raw_data.target)  # ty: ignore

    def power_play_check(self, ball: Dict) -> None:
        """
        Check if the delivery was in a powerplay and add this information to the ball dictionary.

        Parameters
        ----------
        ball : Dict
            The ball dictionary to add the powerplay information to.
        """
        if len(self.powerplays) == 0:
            ball["powerplay"] = False
            return

        ball["powerplay"] = any(
            ball["delivery"] >= pp.from_ and ball["delivery"] <= pp.to
            for pp in self.powerplays
        )

    def target_check(self, ball: Dict) -> None:
        """
        Check if there's a target and add target information to the ball dictionary.

        Parameters
        ----------
        ball : Dict
            The ball dictionary to add the target information to.
        """
        if self.target is None:
            ball["target_runs"] = 0
            ball["target_overs"] = 0.0
        else:
            ball["target_runs"] = self.target.runs
            ball["target_overs"] = self.target.overs

    def parse_innings_data(self) -> List[Dict]:
        """
        Parse the raw innings data into a list of ball dictionaries.

        Returns
        -------
        List[Dict]
            A list of ball dictionaries containing the parsed data about each delivery in the innings.
        """
        innings_data = []
        for over_data in self.raw_data.overs:
            over_data_model = OverData(**over_data)  # ty: ignore
            over = Over(raw_data=over_data_model)
            innings_data.extend(over.parse_over_data())

        for ball in innings_data:
            ball["team"] = self.team
            ball["innings_number"] = self.innings_num
            self.power_play_check(ball)
            self.target_check(ball)
        return innings_data
