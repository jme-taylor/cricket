from typing import Dict


class Ball:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.ball_data: Dict = {}

    def get_batter(self) -> None:
        batter = self.data["batter"]
        self.ball_data["batter"] = batter

    def is_wicket(self) -> bool:
        if "wickets" in self.data:
            return True
        else:
            return False

    def get_wickets(self) -> None:
        if self.is_wicket():
            self.ball_data["wicket_count"] = len(self.data["wickets"])
            self.ball_data["wickets"] = self.data["wickets"]
        else:
            self.ball_data["wicket_count"] = 0
            self.ball_data["wickets"] = []
