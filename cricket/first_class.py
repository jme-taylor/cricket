from cricket.data_processing import load_json_file


class TestPlayerMatchRecord:
    def __init__(self, name: str, team: str, match_json: dict) -> None:
        self.name = name
        self.team = team
        self.match_json = match_json

    def get_batting_statistics(self) -> tuple[list, list]:
        innings_scores = list()
        innings_balls = list()

        for innings in self.match_json["innings"]:
            if innings["team"] != self.team:
                continue
            else:
                score = 0
                balls = 0
                for over in innings["overs"]:
                    for ball in over["deliveries"]:
                        if ball["batter"] == self.name:
                            score += ball["runs"]["batter"]
                            balls += 1
            innings_scores.append(score)
            innings_balls.append(balls)
        return innings_scores, innings_balls

    def get_bowling_statistics(self) -> tuple[list, list]:
        innings_overs = list()
        innings_wickets = list()
        innings_conceded = list()


class TestMatchRecord:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.match_json = load_json_file(filename)
