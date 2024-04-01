from typing import Dict, List

import polars as pl

from cricket.ball_processing import Ball


class Over:
    """
    Parse and format cricket over-related data.

    This class processes cricket over data provided in a dictionary format and
    offers a method to parse the data into a structured format. Each ball
    within the over is processed and detailed information about the delivery is
    extracted. The method `parse_over_data` compiles a list of dictionaries,
    each representing a ball's detailed data including its number, extras, and
    other relevant information.

    Attributes
    ----------
    over_num : int
        The number of the over.
    deliveries : List[Dict]
        A list of dictionaries, each containing data of a ball in the over.
    """

    def __init__(self, over_data: Dict):
        self.over_num = over_data["over"]
        self.deliveries = over_data["deliveries"]

    def parse_over_data(self) -> List:
        """
        Parse raw data into ball by ball format for an over.

        Returns
        -------
        List
            A list of dictionaries, each containing data of a ball in the over.
        """
        ball_num = 1
        ball_num_including_extras = 1
        over_data = pl.DataFrame()
        for delivery in self.deliveries:
            over_float = f"{self.over_num}.{ball_num}"
            ball = Ball(
                delivery,
            )
            ball_data = ball.get_ball_data()
            ball_data = ball_data.with_columns(
                [
                    pl.lit(self.over_num).alias("over_num"),
                    pl.lit(ball_num_including_extras).alias(
                        "ball_num_including_extras"
                    ),
                    pl.lit(ball_num).alias("ball_num"),
                    pl.lit(float(over_float)).alias("delivery"),
                ]
            )
            over_data = pl.concat([over_data, ball_data], how="diagonal")
            if (
                ball_data["wides"].sum() == 0
                and ball_data["noballs"].sum() == 0
            ):
                ball_num += 1
            ball_num_including_extras += 1
        return over_data
