from cricket.ball import Ball
from cricket.constants import PROJECT_ROOT

test_non_scoring_ball = {
    "batter": "ME Trescothick",
    "bowler": "IDR Bradshaw",
    "non_striker": "VS Solanki",
    "runs": {"batter": 0, "extras": 0, "total": 0},
}

test_wicket = {
    "batter": "VS Solanki",
    "bowler": "IDR Bradshaw",
    "non_striker": "ME Trescothick",
    "runs": {"batter": 0, "extras": 0, "total": 0},
    "wickets": [
        {
            "player_out": "VS Solanki",
            "fielders": [{"name": "CO Browne"}],
            "kind": "caught",
        }
    ],
}


def test_get_batsman():
    non_scoring_ball = Ball(test_non_scoring_ball)
    non_scoring_ball.get_batter()
    wicket = Ball(test_wicket)
    wicket.get_batter()
    assert non_scoring_ball.ball_data["batter"] == "ME Trescothick"
    assert wicket.ball_data["batter"] == "VS Solanki"


def test_wickets():
    non_scoring_ball = Ball(test_non_scoring_ball)
    wicket = Ball(test_wicket)
    assert non_scoring_ball.is_wicket() is False
    assert wicket.is_wicket() is True


def test_get_wickets():
    non_scoring_ball = Ball(test_non_scoring_ball)
    non_scoring_ball.get_wickets()
    wicket = Ball(test_wicket)
    wicket.get_wickets()
    assert non_scoring_ball.ball_data["wicket_count"] == 0
    assert non_scoring_ball.ball_data["wickets"] == []
    assert wicket.ball_data["wicket_count"] == 1
    assert wicket.ball_data["wickets"] == [
        {
            "player_out": "VS Solanki",
            "fielders": [{"name": "CO Browne"}],
            "kind": "caught",
        }
    ]
