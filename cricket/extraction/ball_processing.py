from pydantic import BaseModel, computed_field


class BallData(BaseModel):
    """Raw ball data from JSON"""

    batter: str
    non_striker: str
    bowler: str
    runs: dict[str, int]
    extras: dict[str, int] | None = None
    wickets: list | None = None


class Ball(BaseModel):
    """
    Parse data about a single delivery in a cricket match using Pydantic validation.

    Uses computed fields to derive additional data like runs breakdown and wicket information.
    """

    raw_data: BallData
    over_num: int | None = None
    ball_num: int | None = None
    ball_num_including_extras: int | None = None
    delivery: float | None = None

    def __init__(self, raw_data: dict | BallData | None = None, **kwargs):
        """
        Initialize Ball with either dictionary or BallData.
        Maintains backward compatibility with existing API.
        """
        if raw_data is not None and not isinstance(raw_data, BallData):
            raw_data = BallData(**raw_data)  # ty: ignore

        if raw_data is not None:
            kwargs["raw_data"] = raw_data

        super().__init__(**kwargs)

    @computed_field
    @property
    def runs(self) -> int:
        """Total runs scored on this delivery"""
        return self.raw_data.runs["total"]

    @computed_field
    @property
    def batter_runs(self) -> int:
        """Runs scored by the batter"""
        return self.raw_data.runs["batter"]

    @computed_field
    @property
    def extras(self) -> int:
        """Runs from extras"""
        return self.raw_data.runs["extras"]

    @computed_field
    @property
    def wides(self) -> int:
        """Wide balls"""
        return self.raw_data.extras.get("wides", 0) if self.raw_data.extras else 0

    @computed_field
    @property
    def noballs(self) -> int:
        """No balls"""
        return self.raw_data.extras.get("noballs", 0) if self.raw_data.extras else 0

    @computed_field
    @property
    def byes(self) -> int:
        """Bye runs"""
        return self.raw_data.extras.get("byes", 0) if self.raw_data.extras else 0

    @computed_field
    @property
    def legbyes(self) -> int:
        """Leg bye runs"""
        return self.raw_data.extras.get("legbyes", 0) if self.raw_data.extras else 0

    @computed_field
    @property
    def penalty(self) -> int:
        """Penalty runs"""
        return self.raw_data.extras.get("penalty", 0) if self.raw_data.extras else 0

    @computed_field
    @property
    def player_out_1(self) -> str:
        """First player out (if any)"""
        wickets = self.raw_data.wickets or []
        return wickets[0]["player_out"] if len(wickets) > 0 else ""

    @computed_field
    @property
    def kind_1(self) -> str:
        """Dismissal type for first wicket"""
        wickets = self.raw_data.wickets or []
        return wickets[0]["kind"] if len(wickets) > 0 else ""

    @computed_field
    @property
    def player_out_2(self) -> str:
        """Second player out (if any)"""
        wickets = self.raw_data.wickets or []
        return wickets[1]["player_out"] if len(wickets) > 1 else ""

    @computed_field
    @property
    def kind_2(self) -> str:
        """Dismissal type for second wicket"""
        wickets = self.raw_data.wickets or []
        return wickets[1]["kind"] if len(wickets) > 1 else ""

    @computed_field
    @property
    def batter(self) -> str:
        """Batter name"""
        return self.raw_data.batter

    @computed_field
    @property
    def non_striker(self) -> str:
        """Non-striker name"""
        return self.raw_data.non_striker

    @computed_field
    @property
    def bowler(self) -> str:
        """Bowler name"""
        return self.raw_data.bowler

    # Compatibility methods for existing tests
    def get_batter(self) -> dict:
        """Get the batter and non-striker id for the delivery."""
        return {
            "batter": self.batter,
            "non_striker": self.non_striker,
        }

    def get_runs(self) -> dict:
        """Get the runs scored, batter runs and runs from extras"""
        return {
            "runs": self.runs,
            "batter_runs": self.batter_runs,
            "extras": self.extras,
        }

    def get_bowler(self) -> dict:
        """Get the bowler id for the delivery."""
        return {"bowler": self.bowler}

    def get_extras(self) -> dict:
        """Get the specific extras from the delivery, if they exist"""
        return {
            "wides": self.wides,
            "noballs": self.noballs,
            "byes": self.byes,
            "legbyes": self.legbyes,
            "penalty": self.penalty,
        }

    def get_wickets(self) -> dict:
        """Get wicket information"""
        return {
            "player_out_1": self.player_out_1,
            "kind_1": self.kind_1,
            "player_out_2": self.player_out_2,
            "kind_2": self.kind_2,
        }

    @computed_field
    @property
    def ball_data(self) -> dict:
        """Ball data as dictionary - for backwards compatibility"""
        return {
            "batter": self.batter,
            "non_striker": self.non_striker,
            "runs": self.runs,
            "batter_runs": self.batter_runs,
            "extras": self.extras,
            "bowler": self.bowler,
            "wides": self.wides,
            "noballs": self.noballs,
            "byes": self.byes,
            "legbyes": self.legbyes,
            "penalty": self.penalty,
            "player_out_1": self.player_out_1,
            "kind_1": self.kind_1,
            "player_out_2": self.player_out_2,
            "kind_2": self.kind_2,
        }

    def get_ball_data(self) -> dict:
        """Return ball data as dictionary for compatibility"""
        data = self.ball_data.copy()
        if self.over_num is not None:
            data["over_num"] = self.over_num
        if self.ball_num is not None:
            data["ball_num"] = self.ball_num
        if self.ball_num_including_extras is not None:
            data["ball_num_including_extras"] = self.ball_num_including_extras
        if self.delivery is not None:
            data["delivery"] = self.delivery
        return data
