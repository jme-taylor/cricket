from typing import Dict


class Ball:
    """
    Parse data about a single delivery in a cricket match into a dictionary format.
    Takes some raw data in dictionary format and can parse various information about this delivery,
    such as the runs scored, the batter, the bowler, the extras, and the wickets. It can also produce
    a dictionary of this information.

    Attributes
    ----------
    raw_data : Dict
        The raw data about the delivery in dictionary format.
    ball_data : Dict
        The parsed data about the delivery in dictionary format. This is the output of the class, and isn't supplied when instantiating the class.
    """

    def __init__(
        self,
        raw_data: Dict,
    ) -> None:
        self.raw_data = raw_data
        self.ball_data = {}

    def get_batter(self) -> None:
        """Get the batter and non-striker id for the delivery."""
        return {
            "batter": self.raw_data["batter"],
            "non_striker": self.raw_data["non_striker"],
        }

    def get_runs(self) -> None:
        """Get the runs scored, batter runs and runs from extras"""
        return {
            "runs": self.raw_data["runs"]["total"],
            "batter_runs": self.raw_data["runs"]["batter"],
            "extras": self.raw_data["runs"]["extras"],
        }

    def get_bowler(self) -> None:
        """Get the boweler id for the delivery."""
        return {"bowler": self.raw_data["bowler"]}

    def get_extras(self) -> None:
        """Get the specific extras from the delivery, if they exist"""
        EXTRAS_TYPES = ["wides", "noballs", "byes", "legbyes", "penalty"]
        return {
            extra: self.raw_data.get("extras", {}).get(extra, 0)
            for extra in EXTRAS_TYPES
        }

    def get_wickets(self) -> Dict:
        wicket_data = {}
        wickets = self.raw_data.get("wickets", [])
        for wicket_number in [0, 1]:
            try:
                wicket_data[f"player_out_{wicket_number + 1}"] = wickets[
                    wicket_number
                ]["player_out"]
                wicket_data[f"kind_{wicket_number + 1}"] = wickets[
                    wicket_number
                ]["kind"]
            except IndexError:
                wicket_data[f"player_out_{wicket_number + 1}"] = ""
                wicket_data[f"kind_{wicket_number + 1}"] = ""
        return wicket_data

    def get_ball_data(self) -> Dict:
        """Combine all the parsed data into a single dictionary, and return it."""
        self.ball_data.update(self.get_batter())
        self.ball_data.update(self.get_runs())
        self.ball_data.update(self.get_bowler())
        self.ball_data.update(self.get_extras())
        self.ball_data.update(self.get_wickets())
        return self.ball_data
