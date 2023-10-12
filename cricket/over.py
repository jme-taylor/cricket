from typing import List

from cricket.ball import Ball


class Over:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.over_data: List = []

    def get_over_deliveries(self) -> List:
        ball_num = 1
        for ball in self.data["deliveries"]:
            ball = Ball(ball)
            ball_data = ball.get_ball_data()
            ball_data["ball_number"] = ball_num
            self.over_data.append(ball_data)
            if ["wides", "noballs"] not in ball_data.keys():
                ball_num += 1

        return self.over_data
