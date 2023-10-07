from typing import Dict


class Ball:
    """
    This class takes a ball data dictionary and returns a dictionary with
    the relevant information formatted in a way that is easier to work with.
    It will get information about the batter, runs scored, extras, bowler and
    wickets via the associated "get" methods. All of this information will be
    returned in a single dictionary if the "get_ball_data" method is called.

    Parameters
    ----------
    data : dict
        A dictionary containing information about a ball.
    """

    def __init__(self, data: dict) -> None:
        self.data = data
        self.ball_data: Dict = {}

    def get_batter(self) -> None:
        """
        This method will get the batter and non-striker from the ball data
        dictionary and add them to the ball_data dictionary.
        """
        self.ball_data["batter"] = self.data["batter"]
        self.ball_data["non_striker"] = self.data["non_striker"]

    def get_runs(self) -> None:
        """
        This method will get the runs scored, batter runs and extras from the
        ball data dictionary and add them to the ball_data dictionary.
        """
        self.ball_data["runs"] = self.data["runs"]["total"]
        self.ball_data["batter_runs"] = self.data["runs"]["batter"]
        self.ball_data["extras"] = self.data["runs"]["extras"]

    def get_bowler(self) -> None:
        """
        This method will get the bowler from the ball data dictionary and add
        them to the ball_data dictionary.
        """
        self.ball_data["bowler"] = self.data["bowler"]

    def get_extras(self) -> None:
        """
        This method will get the extras from the ball data dictionary and add
        them to the ball_data dictionary. If there are no extras, it will
        not add anything.
        """
        try:
            for key, value in self.data["extras"].items():
                self.ball_data[key] = value
        except KeyError:
            pass

    def get_wickets(self) -> None:
        """
        This method will get the wickets from the ball data dictionary and add
        them to the ball_data dictionary. If there are no wickets, it will
        not add anything.
        """
        if "wickets" in self.data:
            self.ball_data["wicket"] = True
            self.ball_data["wicket_count"] = len(self.data["wickets"])
            self.ball_data["wickets"] = self.data["wickets"]
        else:
            self.ball_data["wicket"] = False
            self.ball_data["wicket_count"] = 0
            self.ball_data["wickets"] = []

    def get_ball_data(self) -> Dict:
        """
        This method will call all of the "get" methods and return the
        ball_data dictionary.

        Returns
        -------
        Dict
            A dictionary containing information about a ball.
        """
        self.get_batter()
        self.get_runs()
        self.get_bowler()
        self.get_extras()
        self.get_wickets()
        return self.ball_data
