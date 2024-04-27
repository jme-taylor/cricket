from typing import Dict, List

from cricket.over_processing import Over


class Innings:
    """
    Parse data about a single innings in a cricket match into a dictionary format.

    Takes some raw data in dictionary format and can parse various information about this innings,
    such as the team, the powerplays, and the target. It can also produce a dictionary of this
    information.

    Attributes
    ----------
    innings_data : Dict
        The raw data about the innings in dictionary format.
    innings_num : int
        The innings number of the innings in the match
    team : str
        The team batting in the innings
    powerplays : List
        The deliveries in which powerplays were active in this innings. Not supplied in class initialisation.
    target: Dict
        The target set for the innings, if it exists. Not supplied in class initialisation.
    """

    def __init__(self, innings_data: Dict, innings_num: int):
        self.innings_data = innings_data
        self.innings_num = innings_num
        self.team = self.innings_data["team"]
        self.powerplays = self.innings_data.get("powerplays", [])
        self.target = self.innings_data.get("target", None)

    def power_play_check(self, ball: Dict) -> None:
        """
        Check if the delivery was in a powerplay and add this information to the ball dictionary.

        Parameters
        ----------
        ball : Dict
            The ball dictionary to add the powerplay information to.
        """
        if len(self.powerplays) == 0:
            ball["powerplay"] = False
        for powerplay in self.powerplays:
            if (
                ball["delivery"] >= powerplay["from"]
                and ball["delivery"] <= powerplay["to"]
            ):
                ball["powerplay"] = True
            else:
                ball["powerplay"] = False

    def target_check(self, ball: Dict) -> None:
        """
        Check if the delivery was in a powerplay and add this information to the ball dictionary.

        Parameters
        ----------
        ball : Dict
            The ball dictionary to add the powerplay information to.
        """
        if self.target is None:
            ball["target_runs"] = 0
            ball["target_overs"] = 0.0
        else:
            ball["target_runs"] = self.target["runs"]
            ball["target_overs"] = self.target["overs"]

    def parse_innings_data(self) -> List:
        """
        Parse the raw innings data into a list of ball dictionaries.

        Returns
        -------
        List
            A list of ball dictionaries containing the parsed data about each delivery in the innings.
        """
        innings_data = []
        for over_data in self.innings_data["overs"]:
            over = Over(over_data)
            innings_data.extend(over.parse_over_data())

        for ball in innings_data:
            ball["team"] = self.team
            ball["innings_number"] = self.innings_num
            self.power_play_check(ball)
            self.target_check(ball)
        return innings_data
