class Ball:
    def __init__(self, json) -> None:
        self.json = json

    def run_data(self) -> dict:
        batter_name = self.json["batter"]
        batter_runs = self.json["runs"]["batter"]

        try:
            non_boundary = self.json["runs"]["non_boundary"]
        except KeyError:
            non_boundary = False

        if not non_boundary and batter_runs == 4:
            batter_fours = 1
        elif not non_boundary and batter_runs == 6:
            batter_sixes = 1

        runs_data = {
            "batter": batter_name,
            "runs": batter_runs,
            "fours": batter_fours,
            "sixes": batter_sixes,
        }
        return runs_data

    def wicket_data(self) -> dict:
        batter_out = self.json["wicket"]["player_out"]
        how_out = self.json["wicket"]["kind"]
