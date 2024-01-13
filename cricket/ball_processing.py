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
        self.ball_data: Dict = {}

    def get_batter(self) -> None:
        """Get the batter and non-striker id for the delivery."""
        self.ball_data["batter"] = self.raw_data["batter"]
        self.ball_data["non_striker"] = self.raw_data["non_striker"]

    def get_runs(self) -> None:
        """Get the runs scored, batter runs and runs from extras"""
        self.ball_data["runs"] = self.raw_data["runs"]["total"]
        self.ball_data["batter_runs"] = self.raw_data["runs"]["batter"]
        self.ball_data["extras"] = self.raw_data["runs"]["extras"]

    def get_bowler(self) -> None:
        """Get the boweler id for the delivery."""
        self.ball_data["bowler"] = self.raw_data["bowler"]

    def get_extras(self) -> None:
        """Get the specific extras from the delivery, if they exist"""
        try:
            for key, value in self.raw_data["extras"].items():
                self.ball_data[key] = value
        except KeyError:
            pass

    def parse_wicket(self, wicket: Dict, wicket_num: int) -> Dict:
        """Parse all wickets from the delivery into a dictionary format."""
        parsed_wicket = {}
        parsed_wicket[f"player_out_{wicket_num}"] = wicket["player_out"]
        parsed_wicket[f"kind_{wicket_num}"] = wicket["kind"]
        return parsed_wicket

    def get_wickets(self) -> None:
        if "wickets" in self.raw_data:
            self.ball_data["wicket"] = True
            self.ball_data["wicket_count"] = len(self.raw_data["wickets"])
            for wicket_num, wicket in enumerate(self.raw_data["wickets"]):
                parsed_wicket = self.parse_wicket(wicket, wicket_num)
                self.ball_data.update(parsed_wicket)
        else:
            self.ball_data["wicket"] = False
            self.ball_data["wicket_count"] = 0

    def get_ball_data(self) -> Dict:
        """Combine all the parsed data into a single dictionary, and return it."""
        self.get_batter()
        self.get_runs()
        self.get_bowler()
        self.get_extras()
        self.get_wickets()
        return self.ball_data
