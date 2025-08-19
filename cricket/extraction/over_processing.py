from typing import Dict, List
from pydantic import BaseModel, computed_field

from cricket.extraction.ball_processing import Ball, BallData


class OverData(BaseModel):
    """Raw over data from JSON"""

    over: int
    deliveries: List[Dict]


class Over(BaseModel):
    """
    Parse and format cricket over-related data using Pydantic validation.

    Uses computed fields to process deliveries into Ball instances with proper numbering.
    """

    raw_data: OverData

    def __init__(self, raw_data: dict | OverData | None = None, **kwargs):
        """
        Initialize Over with either dictionary or OverData.
        Maintains backward compatibility with existing API.
        """
        if raw_data is not None and not isinstance(raw_data, OverData):
            # Convert dict to OverData for backward compatibility
            raw_data = OverData(**raw_data) # ty: ignore

        if raw_data is not None:
            kwargs["raw_data"] = raw_data

        super().__init__(**kwargs)

    @computed_field
    @property
    def over_num(self) -> int:
        """The number of the over"""
        return self.raw_data.over

    @computed_field
    @property
    def deliveries(self) -> List[Dict]:
        """List of delivery dictionaries"""
        return self.raw_data.deliveries

    def parse_over_data(self) -> List[Dict]:
        """
        Parse raw data into ball by ball format for an over.

        Returns
        -------
        List[Dict]
            A list of dictionaries, each containing data of a ball in the over.
        """
        ball_num = 1
        ball_num_including_extras = 1
        over_data = []

        for delivery in self.deliveries:
            ball_data = BallData(**delivery) # ty: ignore
            ball = Ball(
                raw_data=ball_data,
                over_num=self.over_num,
                ball_num=ball_num,
                ball_num_including_extras=ball_num_including_extras,
                delivery=float(f"{self.over_num}.{ball_num}"),
            )

            ball_dict = ball.get_ball_data()

            # Check if this delivery had wides or no balls
            if ball.wides > 0 or ball.noballs > 0:
                over_data.append(ball_dict)
            else:
                over_data.append(ball_dict)
                ball_num += 1
            ball_num_including_extras += 1

        return over_data
