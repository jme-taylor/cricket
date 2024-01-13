import pytest

from cricket.ball_processing import Ball


# Define fixtures for different ball scenarios
@pytest.fixture
def non_scoring_ball():
    return Ball(
        {
            "batter": "JL Langer",
            "bowler": "AR Caddick",
            "non_striker": "ML Hayden",
            "runs": {"batter": 0, "extras": 0, "total": 0},
        },
    )


@pytest.fixture
def wicket_ball():
    return Ball(
        {
            "batter": "N Hussain",
            "bowler": "SCG MacGill",
            "non_striker": "JP Crawley",
            "runs": {"batter": 0, "extras": 0, "total": 0},
            "wickets": [
                {
                    "player_out": "N Hussain",
                    "fielders": [{"name": "ML Hayden"}],
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
            "bowler": "B Lee",
            "extras": {"wides": 1},
            "non_striker": "RWT Key",
            "runs": {"batter": 0, "extras": 1, "total": 1},
        }
    )


@pytest.fixture
def no_ball():
    return Ball(
        {
            "batter": "ML Hayden",
            "bowler": "C White",
            "extras": {"noballs": 1},
            "non_striker": "JL Langer",
            "runs": {"batter": 0, "extras": 1, "total": 1},
        }
    )


@pytest.fixture
def leg_bye():
    return Ball(
        {
            "batter": "ML Hayden",
            "bowler": "AR Caddick",
            "extras": {"legbyes": 1},
            "non_striker": "JL Langer",
            "runs": {"batter": 0, "extras": 1, "total": 1},
        },
    )


@pytest.fixture
def bye():
    return Ball(
        {
            "batter": "ME Trescothick",
            "bowler": "SCG MacGill",
            "extras": {"byes": 2},
            "non_striker": "MA Butcher",
            "runs": {"batter": 0, "extras": 2, "total": 2},
        },
    )


@pytest.fixture
def penalty():
    return Ball(
        {
            "batter": "Habibul Bashar",
            "bowler": "MJ Hoggard",
            "extras": {"penalty": 5},
            "non_striker": "Hannan Sarkar",
            "runs": {"batter": 0, "extras": 5, "total": 5},
        },
    )


def test_get_batsman(
    non_scoring_ball, wicket_ball, wide_ball, no_ball, leg_bye, bye, penalty
):
    non_scoring_ball.get_batter()
    assert non_scoring_ball.ball_data["batter"] == "JL Langer"
    assert non_scoring_ball.ball_data["non_striker"] == "ML Hayden"
    wicket_ball.get_batter()
    assert wicket_ball.ball_data["batter"] == "N Hussain"
    assert wicket_ball.ball_data["non_striker"] == "JP Crawley"
    wide_ball.get_batter()
    assert wide_ball.ball_data["batter"] == "MP Vaughan"
    assert wide_ball.ball_data["non_striker"] == "RWT Key"
    no_ball.get_batter()
    assert no_ball.ball_data["batter"] == "ML Hayden"
    assert no_ball.ball_data["non_striker"] == "JL Langer"
    leg_bye.get_batter()
    assert leg_bye.ball_data["batter"] == "ML Hayden"
    assert leg_bye.ball_data["non_striker"] == "JL Langer"
    bye.get_batter()
    assert bye.ball_data["batter"] == "ME Trescothick"
    assert bye.ball_data["non_striker"] == "MA Butcher"
    penalty.get_batter()
    assert penalty.ball_data["batter"] == "Habibul Bashar"
    assert penalty.ball_data["non_striker"] == "Hannan Sarkar"


def test_get_runs(
    non_scoring_ball, wicket_ball, wide_ball, no_ball, leg_bye, bye, penalty
):
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
    no_ball.get_runs()
    assert no_ball.ball_data["runs"] == 1
    assert no_ball.ball_data["batter_runs"] == 0
    assert no_ball.ball_data["extras"] == 1
    leg_bye.get_runs()
    assert leg_bye.ball_data["runs"] == 1
    assert leg_bye.ball_data["batter_runs"] == 0
    assert leg_bye.ball_data["extras"] == 1
    bye.get_runs()
    assert bye.ball_data["runs"] == 2
    assert bye.ball_data["batter_runs"] == 0
    assert bye.ball_data["extras"] == 2
    penalty.get_runs()
    assert penalty.ball_data["runs"] == 5
    assert penalty.ball_data["batter_runs"] == 0
    assert penalty.ball_data["extras"] == 5


def test_get_bowler(
    non_scoring_ball, wicket_ball, wide_ball, no_ball, leg_bye, bye, penalty
):
    non_scoring_ball.get_bowler()
    assert non_scoring_ball.ball_data["bowler"] == "AR Caddick"
    wicket_ball.get_bowler()
    assert wicket_ball.ball_data["bowler"] == "SCG MacGill"
    wide_ball.get_bowler()
    assert wide_ball.ball_data["bowler"] == "B Lee"
    no_ball.get_bowler()
    assert no_ball.ball_data["bowler"] == "C White"
    leg_bye.get_bowler()
    assert leg_bye.ball_data["bowler"] == "AR Caddick"
    bye.get_bowler()
    assert bye.ball_data["bowler"] == "SCG MacGill"
    penalty.get_bowler()
    assert penalty.ball_data["bowler"] == "MJ Hoggard"


def test_get_extras(
    non_scoring_ball, wicket_ball, wide_ball, no_ball, leg_bye, bye, penalty
):
    non_scoring_ball.get_extras()
    assert non_scoring_ball.ball_data == {}
    with pytest.raises(KeyError):
        assert non_scoring_ball.ball_data["wides"] == 0
        assert non_scoring_ball.ball_data["noballs"] == 0
        assert non_scoring_ball.ball_data["legbyes"] == 0
        assert non_scoring_ball.ball_data["penalty"] == 0
        assert non_scoring_ball.ball_data["byes"] == 0
    wicket_ball.get_extras()
    assert wicket_ball.ball_data == {}
    with pytest.raises(KeyError):
        assert wicket_ball.ball_data["wides"] == 0
        assert wicket_ball.ball_data["noballs"] == 0
        assert wicket_ball.ball_data["legbyes"] == 0
        assert wicket_ball.ball_data["penalty"] == 0
        assert wicket_ball.ball_data["byes"] == 0
    wide_ball.get_extras()
    assert wide_ball.ball_data["wides"] == 1
    with pytest.raises(KeyError):
        assert wide_ball.ball_data["noballs"] == 0
        assert wide_ball.ball_data["legbyes"] == 0
        assert wide_ball.ball_data["penalty"] == 0
        assert wide_ball.ball_data["byes"] == 0
    no_ball.get_extras()
    assert no_ball.ball_data["noballs"] == 1
    with pytest.raises(KeyError):
        assert no_ball.ball_data["wides"] == 0
        assert no_ball.ball_data["legbyes"] == 0
        assert no_ball.ball_data["penalty"] == 0
        assert no_ball.ball_data["byes"] == 0
    leg_bye.get_extras()
    assert leg_bye.ball_data["legbyes"] == 1
    with pytest.raises(KeyError):
        assert leg_bye.ball_data["noballs"] == 0
        assert leg_bye.ball_data["wides"] == 0
        assert leg_bye.ball_data["penalty"] == 0
        assert leg_bye.ball_data["byes"] == 0
    bye.get_extras()
    assert bye.ball_data["byes"] == 2
    with pytest.raises(KeyError):
        assert bye.ball_data["noballs"] == 0
        assert bye.ball_data["wides"] == 0
        assert bye.ball_data["legbyes"] == 0
        assert bye.ball_data["penalty"] == 0
    penalty.get_extras()
    assert penalty.ball_data["penalty"] == 5
    with pytest.raises(KeyError):
        assert penalty.ball_data["noballs"] == 0
        assert penalty.ball_data["wides"] == 0
        assert penalty.ball_data["legbyes"] == 0
        assert penalty.ball_data["byes"] == 0


def test_get_wickets(
    non_scoring_ball, wicket_ball, wide_ball, no_ball, leg_bye, bye, penalty
):
    non_scoring_ball.get_wickets()
    assert non_scoring_ball.ball_data["wicket"] is False
    assert non_scoring_ball.ball_data["wicket_count"] == 0
    with pytest.raises(KeyError):
        assert non_scoring_ball.ball_data["player_out_0"] == "N Hussain"
    wicket_ball.get_wickets()
    assert wicket_ball.ball_data["wicket"] is True
    assert wicket_ball.ball_data["wicket_count"] == 1
    assert wicket_ball.ball_data["player_out_0"] == "N Hussain"
    assert wicket_ball.ball_data["kind_0"] == "caught"
    with pytest.raises(KeyError):
        assert wicket_ball.ball_data["player_out_1"] == "N Hussain"
    wide_ball.get_wickets()
    assert wide_ball.ball_data["wicket"] is False
    assert wide_ball.ball_data["wicket_count"] == 0
    with pytest.raises(KeyError):
        assert wide_ball.ball_data["player_out_0"] == "ML Hayden"
    no_ball.get_wickets()
    assert no_ball.ball_data["wicket"] is False
    assert no_ball.ball_data["wicket_count"] == 0
    with pytest.raises(KeyError):
        assert no_ball.ball_data["player_out_0"] == "ML Hayden"
    leg_bye.get_wickets()
    assert leg_bye.ball_data["wicket"] is False
    assert leg_bye.ball_data["wicket_count"] == 0
    with pytest.raises(KeyError):
        assert leg_bye.ball_data["player_out_0"] == "ML Hayden"
    bye.get_wickets()
    assert bye.ball_data["wicket"] is False
    assert bye.ball_data["wicket_count"] == 0
    with pytest.raises(KeyError):
        assert bye.ball_data["player_out_0"] == "ML Hayden"
    penalty.get_wickets()
    assert penalty.ball_data["wicket"] is False
    assert penalty.ball_data["wicket_count"] == 0
    with pytest.raises(KeyError):
        assert penalty.ball_data["player_out_0"] == "ML Hayden"


def test_get_ball_data(
    non_scoring_ball, wicket_ball, wide_ball, no_ball, leg_bye, bye, penalty
):
    non_scoring_ball.get_ball_data()
    assert non_scoring_ball.ball_data == {
        "batter": "JL Langer",
        "non_striker": "ML Hayden",
        "runs": 0,
        "batter_runs": 0,
        "extras": 0,
        "bowler": "AR Caddick",
        "wicket": False,
        "wicket_count": 0,
    }
    wicket_ball.get_ball_data()
    assert wicket_ball.ball_data == {
        "batter": "N Hussain",
        "non_striker": "JP Crawley",
        "runs": 0,
        "batter_runs": 0,
        "extras": 0,
        "bowler": "SCG MacGill",
        "wicket": True,
        "wicket_count": 1,
        "player_out_0": "N Hussain",
        "kind_0": "caught",
    }
    wide_ball.get_ball_data()
    assert wide_ball.ball_data == {
        "batter": "MP Vaughan",
        "non_striker": "RWT Key",
        "runs": 1,
        "batter_runs": 0,
        "extras": 1,
        "bowler": "B Lee",
        "wides": 1,
        "wicket": False,
        "wicket_count": 0,
    }
    no_ball.get_ball_data()
    assert no_ball.ball_data == {
        "batter": "ML Hayden",
        "non_striker": "JL Langer",
        "runs": 1,
        "batter_runs": 0,
        "extras": 1,
        "bowler": "C White",
        "noballs": 1,
        "wicket": False,
        "wicket_count": 0,
    }
    leg_bye.get_ball_data()
    assert leg_bye.ball_data == {
        "batter": "ML Hayden",
        "non_striker": "JL Langer",
        "runs": 1,
        "batter_runs": 0,
        "extras": 1,
        "bowler": "AR Caddick",
        "legbyes": 1,
        "wicket": False,
        "wicket_count": 0,
    }
    bye.get_ball_data()
    assert bye.ball_data == {
        "batter": "ME Trescothick",
        "non_striker": "MA Butcher",
        "runs": 2,
        "batter_runs": 0,
        "extras": 2,
        "bowler": "SCG MacGill",
        "byes": 2,
        "wicket": False,
        "wicket_count": 0,
    }
    penalty.get_ball_data()
    assert penalty.ball_data == {
        "batter": "Habibul Bashar",
        "non_striker": "Hannan Sarkar",
        "runs": 5,
        "batter_runs": 0,
        "extras": 5,
        "bowler": "MJ Hoggard",
        "penalty": 5,
        "wicket": False,
        "wicket_count": 0,
    }
