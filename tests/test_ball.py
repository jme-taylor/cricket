import pytest

from cricket.ball import Ball


# Define fixtures for different ball scenarios
@pytest.fixture
def non_scoring_ball():
    return Ball(
        {
            "batter": "ME Trescothick",
            "bowler": "IDR Bradshaw",
            "non_striker": "VS Solanki",
            "runs": {"batter": 0, "extras": 0, "total": 0},
        }
    )


@pytest.fixture
def wicket_ball():
    return Ball(
        {
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
    )


@pytest.fixture
def wide_ball():
    return Ball(
        {
            "batter": "MP Vaughan",
            "bowler": "IDR Bradshaw",
            "extras": {"wides": 1},
            "non_striker": "ME Trescothick",
            "runs": {"batter": 0, "extras": 1, "total": 1},
        }
    )


def test_get_batsman(non_scoring_ball, wicket_ball, wide_ball):
    non_scoring_ball.get_batter()
    assert non_scoring_ball.ball_data["batter"] == "ME Trescothick"
    assert non_scoring_ball.ball_data["non_striker"] == "VS Solanki"
    wicket_ball.get_batter()
    assert wicket_ball.ball_data["batter"] == "VS Solanki"
    assert wicket_ball.ball_data["non_striker"] == "ME Trescothick"
    wide_ball.get_batter()
    assert wide_ball.ball_data["batter"] == "MP Vaughan"
    assert wide_ball.ball_data["non_striker"] == "ME Trescothick"


def test_get_runs(non_scoring_ball, wicket_ball, wide_ball):
    non_scoring_ball.get_runs()
    assert non_scoring_ball.ball_data["runs"] == 0
    assert non_scoring_ball.ball_data["batter_runs"] == 0
    assert non_scoring_ball.ball_data["extras"] == 0
    wicket_ball.get_runs()
    assert wicket_ball.ball_data["runs"] == 0
    assert wicket_ball.ball_data["batter_runs"] == 0
    assert wicket_ball.ball_data["extras"] == 0
    wide_ball.get_runs()
    assert wide_ball.ball_data["runs"] == 1
    assert wide_ball.ball_data["batter_runs"] == 0
    assert wide_ball.ball_data["extras"] == 1


def test_get_bowler(non_scoring_ball, wicket_ball, wide_ball):
    non_scoring_ball.get_bowler()
    assert non_scoring_ball.ball_data["bowler"] == "IDR Bradshaw"
    wicket_ball.get_bowler()
    assert wicket_ball.ball_data["bowler"] == "IDR Bradshaw"
    wide_ball.get_bowler()
    assert wide_ball.ball_data["bowler"] == "IDR Bradshaw"


def test_get_extras(non_scoring_ball, wicket_ball, wide_ball):
    non_scoring_ball.get_extras()
    assert non_scoring_ball.ball_data == {}
    wicket_ball.get_extras()
    assert wicket_ball.ball_data == {}
    with pytest.raises(KeyError):
        assert wicket_ball.ball_data["wides"] == 0
    wide_ball.get_extras()
    assert wide_ball.ball_data["wides"] == 1


def test_get_wickets(non_scoring_ball, wicket_ball, wide_ball):
    non_scoring_ball.get_wickets()
    assert non_scoring_ball.ball_data["wicket"] is False
    assert non_scoring_ball.ball_data["wicket_count"] == 0
    assert non_scoring_ball.ball_data["wickets"] == []
    wicket_ball.get_wickets()
    assert wicket_ball.ball_data["wicket"] is True
    assert wicket_ball.ball_data["wicket_count"] == 1
    assert wicket_ball.ball_data["wickets"] == [
        {
            "player_out": "VS Solanki",
            "fielders": [{"name": "CO Browne"}],
            "kind": "caught",
        }
    ]
    wide_ball.get_wickets()
    assert wide_ball.ball_data["wicket"] is False
    assert wide_ball.ball_data["wicket_count"] == 0
    assert wide_ball.ball_data["wickets"] == []


def test_get_ball_data(non_scoring_ball, wicket_ball, wide_ball):
    non_scoring_ball.get_ball_data()
    assert non_scoring_ball.ball_data == {
        "batter": "ME Trescothick",
        "non_striker": "VS Solanki",
        "runs": 0,
        "batter_runs": 0,
        "extras": 0,
        "bowler": "IDR Bradshaw",
        "wicket": False,
        "wicket_count": 0,
        "wickets": [],
    }
    wicket_ball.get_ball_data()
    assert wicket_ball.ball_data == {
        "batter": "VS Solanki",
        "non_striker": "ME Trescothick",
        "runs": 0,
        "batter_runs": 0,
        "extras": 0,
        "bowler": "IDR Bradshaw",
        "wicket": True,
        "wicket_count": 1,
        "wickets": [
            {
                "player_out": "VS Solanki",
                "fielders": [{"name": "CO Browne"}],
                "kind": "caught",
            }
        ],
    }
    wide_ball.get_ball_data()
    assert wide_ball.ball_data == {
        "batter": "MP Vaughan",
        "non_striker": "ME Trescothick",
        "runs": 1,
        "batter_runs": 0,
        "extras": 1,
        "bowler": "IDR Bradshaw",
        "wicket": False,
        "wicket_count": 0,
        "wickets": [],
        "wides": 1,
    }
