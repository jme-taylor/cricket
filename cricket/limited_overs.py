from dataclasses import dataclass 

@dataclass
class LimitedOversMatch:
    

@dataclass
class LimitedOversInning:

@dataclass
class LimitedOversBall:


class LimitedOversPlayerMatchRecord:
    def __init__(self, player_name, team, match_json):
        self.player_name = player_name
        self.team = team
        self.match_json = match_json

    def _get_balls_faced(self) -> None:
        self.balls_faced = []
        innings = self.match_json["innings"]
        for inning in innings:
            if self.team != inning["team"]:
                continue
            else:
                for over in inning["overs"]:
                    for delivery in over["deliveries"]:
                        if delivery["batter"] == self.player_name:
                            self.balls_faced.append(delivery)

    def get_batting_statistics(self) -> None:
        self._get_balls_faced()

        score = 0
        balls_faced = 0
        fours = 0
        sixes = 0
        how_out = None
        bowler = None

        for ball in self.balls_faced:
            score += ball["runs"]["batter"]
            balls_faced += 1
            if ball["runs"]["batter"] == 4:
                fours += 1
            if ball["runs"]["batter"] == 6:
                sixes += 1
            try:
                if (
                    ball["wickets"]
                    and ball["wickets"][0]["player_out"] == self.player_name
                ):
                    how_out = ball["wickets"][0]["kind"]
                    if how_out not in ["run out", "retired hurt"]:
                        bowler = ball["bowler"]
            except KeyError:
                pass
        self.batting_statistics = {
            "score": score,
            "balls": balls_faced,
            "fours": fours,
            "sixes": sixes,
            "how_out": how_out,
            "bowler": bowler,
        }

    def get_bowling_statistics(self) -> None:
        for inning in self.match_json["innings"]:
            if inning["team"] == self.team:
                pass
            else:
                overs_bowled = 0
                runs_conceded = 0
                extras = 0
                wides = 0
                no_balls = 0 
                leg_byes = 0
                byes = 0
                wickets = 0
                maidens = 0
                wickets_detail = []
                for over in inning["overs"]:
                    delivery_count = 0
                    for delivery in over["deliveries"]:
                        runs_conceded += delivery["runs"]["total"]
                        try:
                            extras = delivery["extras"]
                            extras += delivery["runs"]["extras"]
                        try:
                            if delivery["wickets"]:
                                wickets += 1
                                wicket_detail = {
                                    "kind": delivery["wickets"][0]["kind"],
                                    "batsmen": delivery["wickets"][0]["player_out"],
                                }
                                wickets_detail.append(wicket_detail)
                        except KeyError:
                            pass
                    if runs_conceded == 0:
                        maidens += 1
