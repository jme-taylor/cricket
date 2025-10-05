from cricket.simulator import MatchSimulator


class TestSimulateBall:
    def test_simulate_ball_returns_runs_scored(self):
        # Given a simulator with a known seed and weights favoring runs
        simulator = MatchSimulator(seed=42)
        weights = {
            "0": 0.1,
            "1": 0.1,
            "2": 0.1,
            "3": 0.1,
            "4": 0.2,
            "6": 0.3,
            "wicket": 0.1,
        }

        # When we simulate a ball
        result = simulator.simulate_ball(weights)

        # Then the result should be a valid outcome (0-6 or wicket returns 0)
        assert result in [0, 1, 2, 3, 4, 6]

    def test_simulate_ball_increments_score_for_runs(self):
        # Given a simulator with weights guaranteeing a 4
        simulator = MatchSimulator(seed=100)
        weights = {
            "0": 0.0,
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
            "4": 1.0,
            "6": 0.0,
            "wicket": 0.0,
        }
        initial_score = simulator.current_score

        # When we simulate a ball
        runs = simulator.simulate_ball(weights)

        # Then the score should increase by the runs scored
        assert simulator.current_score == initial_score + runs
        assert runs == 4

    def test_simulate_ball_increments_wickets_on_dismissal(self):
        # Given a simulator with weights guaranteeing a wicket
        simulator = MatchSimulator(seed=50)
        weights = {
            "0": 0.0,
            "1": 0.0,
            "2": 0.0,
            "3": 0.0,
            "4": 0.0,
            "6": 0.0,
            "wicket": 1.0,
        }
        initial_wickets = simulator.wickets

        # When we simulate a ball
        runs = simulator.simulate_ball(weights)

        # Then wickets should increment and runs should be 0
        assert simulator.wickets == initial_wickets + 1
        assert runs == 0

    def test_simulate_ball_increments_balls_bowled(self):
        # Given a simulator
        simulator = MatchSimulator(seed=10)
        weights = {
            "0": 0.3,
            "1": 0.2,
            "2": 0.1,
            "3": 0.05,
            "4": 0.15,
            "6": 0.1,
            "wicket": 0.1,
        }
        initial_balls = simulator.balls_bowled

        # When we simulate a ball
        simulator.simulate_ball(weights)

        # Then balls bowled should increment
        assert simulator.balls_bowled == initial_balls + 1

    def test_simulate_ball_decrements_overs_remaining(self):
        # Given a simulator starting with 20 overs
        simulator = MatchSimulator(seed=15)
        weights = {
            "0": 0.3,
            "1": 0.2,
            "2": 0.1,
            "3": 0.05,
            "4": 0.15,
            "6": 0.1,
            "wicket": 0.1,
        }
        initial_overs = simulator.overs_remaining

        # When we simulate a ball
        simulator.simulate_ball(weights)

        # Then overs remaining should decrease by 1/6 (one ball)
        expected_overs = initial_overs - (1 / 6)
        assert abs(simulator.overs_remaining - expected_overs) < 0.001

    def test_deterministic_outcomes_with_seed(self):
        # Given two simulators with the same seed
        simulator1 = MatchSimulator(seed=999)
        simulator2 = MatchSimulator(seed=999)
        weights = {
            "0": 0.3,
            "1": 0.2,
            "2": 0.1,
            "3": 0.05,
            "4": 0.15,
            "6": 0.1,
            "wicket": 0.1,
        }

        # When we simulate multiple balls on each
        results1 = [simulator1.simulate_ball(weights) for _ in range(10)]
        results2 = [simulator2.simulate_ball(weights) for _ in range(10)]

        # Then the outcomes should be identical
        assert results1 == results2


class TestSimulateInnings:
    def test_innings_completes_at_ten_wickets(self):
        # Given a simulator with weights heavily favoring wickets
        simulator = MatchSimulator(seed=200)
        simulator.default_weights = {
            "0": 0.1,
            "1": 0.05,
            "2": 0.05,
            "3": 0.0,
            "4": 0.05,
            "6": 0.05,
            "wicket": 0.7,
        }

        # When we simulate an innings
        result = simulator.simulate_innings()

        # Then the innings should end with 10 wickets
        assert result["wickets"] == 10
        assert result["balls_bowled"] <= 120

    def test_innings_completes_at_120_balls(self):
        # Given a simulator with weights avoiding wickets
        simulator = MatchSimulator(seed=300)
        simulator.default_weights = {
            "0": 0.3,
            "1": 0.2,
            "2": 0.15,
            "3": 0.1,
            "4": 0.15,
            "6": 0.1,
            "wicket": 0.0,
        }

        # When we simulate an innings
        result = simulator.simulate_innings()

        # Then the innings should complete 120 balls
        assert result["balls_bowled"] == 120
        assert result["wickets"] < 10
        assert result["overs_remaining"] == 0

    def test_innings_returns_complete_match_state(self):
        # Given a simulator
        simulator = MatchSimulator(seed=400)

        # When we simulate an innings
        result = simulator.simulate_innings()

        # Then the result should contain all match state
        assert "current_score" in result
        assert "wickets" in result
        assert "balls_bowled" in result
        assert "overs_remaining" in result
        assert result["current_score"] >= 0
        assert 0 <= result["wickets"] <= 10
        assert 0 <= result["balls_bowled"] <= 120
        assert result["overs_remaining"] >= 0
