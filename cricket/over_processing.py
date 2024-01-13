from typing import Dict, List

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
        over_data = []
        for delivery in self.deliveries:
            ball = Ball(
                delivery,
            )
            ball.get_ball_data()
            delivery_data = ball.ball_data
            delivery_data["over_num"] = self.over_num
            delivery_data[
                "ball_num_including_extras"
            ] = ball_num_including_extras
            delivery_data["ball_num"] = ball_num
            over_float = f"{self.over_num}.{ball_num}"
            delivery_data["delivery"] = float(over_float)
            if "wides" in ball.ball_data or "noballs" in ball.ball_data:
                over_data.append(delivery_data)
            else:
                over_data.append(delivery_data)
                ball_num += 1
            ball_num_including_extras += 1
        return over_data
