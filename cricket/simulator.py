import random

from cricket.logging_config import logger

class MatchSimulator:
    def __init__(self, seed: int | None = None):
        self.current_score = 0
        self.wickets = 0
        self.balls_bowled = 0
        self.overs_remaining = 20.0
        self.rng = random.Random(seed)
        self.outcomes = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "6",
            "wicket",
        ]
        self.default_weights = {
            "0": 0.3,
            "1": 0.2,
            "2": 0.1,
            "3": 0.05,
            "4": 0.15,
            "6": 0.1,
            "wicket": 0.1,
        }

    def get_ball_weights(self) -> dict:
        return self.default_weights

    def simulate_ball(self, weights: dict) -> int:
        if list(weights.keys()) != self.outcomes:
            raise ValueError("Weights must match outcomes")
        probabilities = list(weights.values())

        outcome = self.rng.choices(self.outcomes, weights=probabilities, k=1)[0]

        if outcome == "wicket":
            self.wickets += 1
            logger.info(f"{self.current_score}-{self.wickets}")
            runs = 0
        else:
            runs = int(outcome)
            self.current_score += runs

        self.balls_bowled += 1
        self.overs_remaining -= 1 / 6

        return runs

    def simulate_innings(self) -> dict:
        while self.wickets < 10 and self.balls_bowled < 120:
            weights = self.get_ball_weights()
            self.simulate_ball(weights)

        return {
            "current_score": self.current_score,
            "wickets": self.wickets,
            "balls_bowled": self.balls_bowled,
            "overs_remaining": round(self.overs_remaining, 10),
        }


if __name__ == "__main__":
    simulator = MatchSimulator()
    print(simulator.simulate_innings())
