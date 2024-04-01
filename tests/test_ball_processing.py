import polars as pl
import pytest
from polars.testing import assert_frame_equal

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


def test_get_batter(
    non_scoring_ball, wicket_ball, wide_ball, no_ball, leg_bye, bye, penalty
):
    non_scoring_batter_info = non_scoring_ball.get_batter()
    assert non_scoring_batter_info["batter"] == "JL Langer"
    assert non_scoring_batter_info["non_striker"] == "ML Hayden"
    wicket_batter_info = wicket_ball.get_batter()
    assert wicket_batter_info["batter"] == "N Hussain"
    assert wicket_batter_info["non_striker"] == "JP Crawley"
    wide_ball_batter_info = wide_ball.get_batter()
    assert wide_ball_batter_info["batter"] == "MP Vaughan"
    assert wide_ball_batter_info["non_striker"] == "RWT Key"
    no_ball_batter_info = no_ball.get_batter()
    assert no_ball_batter_info["batter"] == "ML Hayden"
    assert no_ball_batter_info["non_striker"] == "JL Langer"
    leg_bye_batter_info = leg_bye.get_batter()
    assert leg_bye_batter_info["batter"] == "ML Hayden"
    assert leg_bye_batter_info["non_striker"] == "JL Langer"
    bye_batter_info = bye.get_batter()
    assert bye_batter_info["batter"] == "ME Trescothick"
    assert bye_batter_info["non_striker"] == "MA Butcher"
    penalty_batter_info = penalty.get_batter()
    assert penalty_batter_info["batter"] == "Habibul Bashar"
    assert penalty_batter_info["non_striker"] == "Hannan Sarkar"


def test_get_runs(
    non_scoring_ball, wicket_ball, wide_ball, no_ball, leg_bye, bye, penalty
):
    non_scoring_run_info = non_scoring_ball.get_runs()
    assert non_scoring_run_info["runs"] == 0
    assert non_scoring_run_info["batter_runs"] == 0
    assert non_scoring_run_info["extras"] == 0
    wicket_ball_run_info = wicket_ball.get_runs()
    assert wicket_ball_run_info["runs"] == 0
    assert wicket_ball_run_info["batter_runs"] == 0
    assert wicket_ball_run_info["extras"] == 0
    wide_ball_run_info = wide_ball.get_runs()
    assert wide_ball_run_info["runs"] == 1
    assert wide_ball_run_info["batter_runs"] == 0
    assert wide_ball_run_info["extras"] == 1
    no_ball_run_info = no_ball.get_runs()
    assert no_ball_run_info["runs"] == 1
    assert no_ball_run_info["batter_runs"] == 0
    assert no_ball_run_info["extras"] == 1
    leg_bye_run_info = leg_bye.get_runs()
    assert leg_bye_run_info["runs"] == 1
    assert leg_bye_run_info["batter_runs"] == 0
    assert leg_bye_run_info["extras"] == 1
    bye_run_info = bye.get_runs()
    assert bye_run_info["runs"] == 2
    assert bye_run_info["batter_runs"] == 0
    assert bye_run_info["extras"] == 2
    penalty_run_info = penalty.get_runs()
    assert penalty_run_info["runs"] == 5
    assert penalty_run_info["batter_runs"] == 0
    assert penalty_run_info["extras"] == 5


def test_get_bowler(
    non_scoring_ball, wicket_ball, wide_ball, no_ball, leg_bye, bye, penalty
):
    non_scoring_bowler_info = non_scoring_ball.get_bowler()
    assert non_scoring_bowler_info["bowler"] == "AR Caddick"
    wicket_bowler_info = wicket_ball.get_bowler()
    assert wicket_bowler_info["bowler"] == "SCG MacGill"
    wide_bowler_info = wide_ball.get_bowler()
    assert wide_bowler_info["bowler"] == "B Lee"
    no_ball_bowler_info = no_ball.get_bowler()
    assert no_ball_bowler_info["bowler"] == "C White"
    leg_bye_bowler_info = leg_bye.get_bowler()
    assert leg_bye_bowler_info["bowler"] == "AR Caddick"
    bye_bowler_info = bye.get_bowler()
    assert bye_bowler_info["bowler"] == "SCG MacGill"
    penalty_bowler = penalty.get_bowler()
    assert penalty_bowler["bowler"] == "MJ Hoggard"


def test_get_extras(
    non_scoring_ball, wicket_ball, wide_ball, no_ball, leg_bye, bye, penalty
):
    non_scoring_extras = non_scoring_ball.get_extras()
    assert non_scoring_extras["wides"] == 0
    assert non_scoring_extras["noballs"] == 0
    assert non_scoring_extras["byes"] == 0
    assert non_scoring_extras["legbyes"] == 0
    assert non_scoring_extras["penalty"] == 0
    wicket_extras = wicket_ball.get_extras()
    assert wicket_extras["wides"] == 0
    assert wicket_extras["noballs"] == 0
    assert wicket_extras["byes"] == 0
    assert wicket_extras["legbyes"] == 0
    assert wicket_extras["penalty"] == 0
    wide_extras = wide_ball.get_extras()
    assert wide_extras["wides"] == 1
    assert wide_extras["noballs"] == 0
    assert wide_extras["byes"] == 0
    assert wide_extras["legbyes"] == 0
    assert wide_extras["penalty"] == 0
    no_ball_extras = no_ball.get_extras()
    assert no_ball_extras["wides"] == 0
    assert no_ball_extras["noballs"] == 1
    assert no_ball_extras["byes"] == 0
    assert no_ball_extras["legbyes"] == 0
    assert no_ball_extras["penalty"] == 0
    leg_bye_extras = leg_bye.get_extras()
    assert leg_bye_extras["wides"] == 0
    assert leg_bye_extras["noballs"] == 0
    assert leg_bye_extras["byes"] == 0
    assert leg_bye_extras["legbyes"] == 1
    assert leg_bye_extras["penalty"] == 0
    bye_extras = bye.get_extras()
    assert bye_extras["wides"] == 0
    assert bye_extras["noballs"] == 0
    assert bye_extras["byes"] == 2
    assert bye_extras["legbyes"] == 0
    assert bye_extras["penalty"] == 0
    penalty_extras = penalty.get_extras()
    assert penalty_extras["wides"] == 0
    assert penalty_extras["noballs"] == 0
    assert penalty_extras["byes"] == 0
    assert penalty_extras["legbyes"] == 0
    assert penalty_extras["penalty"] == 5


def test_get_wickets(
    non_scoring_ball, wicket_ball, wide_ball, no_ball, leg_bye, bye, penalty
):
    non_scoring_wickets = non_scoring_ball.get_wickets()
    assert non_scoring_wickets["player_out_1"] == ""
    assert non_scoring_wickets["kind_1"] == ""
    assert non_scoring_wickets["player_out_2"] == ""
    assert non_scoring_wickets["kind_2"] == ""
    wicket_wickets = wicket_ball.get_wickets()
    assert wicket_wickets["player_out_1"] == "N Hussain"
    assert wicket_wickets["kind_1"] == "caught"
    assert wicket_wickets["player_out_2"] == ""
    assert wicket_wickets["kind_2"] == ""
    wide_wickets = wide_ball.get_wickets()
    assert wide_wickets["player_out_1"] == ""
    assert wide_wickets["kind_1"] == ""
    assert wide_wickets["player_out_2"] == ""
    assert wide_wickets["kind_2"] == ""
    no_ball_wickets = no_ball.get_wickets()
    assert no_ball_wickets["player_out_1"] == ""
    assert no_ball_wickets["kind_1"] == ""
    assert no_ball_wickets["player_out_2"] == ""
    assert no_ball_wickets["kind_2"] == ""
    leg_bye_wickets = leg_bye.get_wickets()
    assert leg_bye_wickets["player_out_1"] == ""
    assert leg_bye_wickets["kind_1"] == ""
    assert leg_bye_wickets["player_out_2"] == ""
    assert leg_bye_wickets["kind_2"] == ""
    bye_wickets = bye.get_wickets()
    assert bye_wickets["player_out_1"] == ""
    assert bye_wickets["kind_1"] == ""
    assert bye_wickets["player_out_2"] == ""
    assert bye_wickets["kind_2"] == ""
    penalty_wickets = penalty.get_wickets()
    assert penalty_wickets["player_out_1"] == ""
    assert penalty_wickets["kind_1"] == ""
    assert penalty_wickets["player_out_2"] == ""
    assert penalty_wickets["kind_2"] == ""


def test_get_ball_data(
    non_scoring_ball, wicket_ball, wide_ball, no_ball, leg_bye, bye, penalty
):
    non_scoring_output = non_scoring_ball.get_ball_data()
    expected_non_scoring_output = pl.DataFrame(
        {
            "batter": ["JL Langer"],
            "non_striker": ["ML Hayden"],
            "runs": [0],
            "batter_runs": [0],
            "extras": [0],
            "bowler": ["AR Caddick"],
            "wides": [0],
            "noballs": [0],
            "byes": [0],
            "legbyes": [0],
            "penalty": [0],
            "player_out_1": [""],
            "kind_1": [""],
            "player_out_2": [""],
            "kind_2": [""],
        }
    )
    assert_frame_equal(non_scoring_output, expected_non_scoring_output)
    wicket_output = wicket_ball.get_ball_data()
    expected_wicket_output = pl.DataFrame(
        {
            "batter": ["N Hussain"],
            "non_striker": ["JP Crawley"],
            "runs": [0],
            "batter_runs": [0],
            "extras": [0],
            "bowler": ["SCG MacGill"],
            "wides": [0],
            "noballs": [0],
            "byes": [0],
            "legbyes": [0],
            "penalty": [0],
            "player_out_1": ["N Hussain"],
            "kind_1": ["caught"],
            "player_out_2": [""],
            "kind_2": [""],
        }
    )
    assert_frame_equal(wicket_output, expected_wicket_output)
    wide_output = wide_ball.get_ball_data()
    expected_wide_output = pl.DataFrame(
        {
            "batter": ["MP Vaughan"],
            "non_striker": ["RWT Key"],
            "runs": [1],
            "batter_runs": [0],
            "extras": [1],
            "bowler": ["B Lee"],
            "wides": [1],
            "noballs": [0],
            "byes": [0],
            "legbyes": [0],
            "penalty": [0],
            "player_out_1": [""],
            "kind_1": [""],
            "player_out_2": [""],
            "kind_2": [""],
        }
    )
    assert_frame_equal(wide_output, expected_wide_output)
    no_ball_output = no_ball.get_ball_data()
    expected_no_ball_output = pl.DataFrame(
        {
            "batter": ["ML Hayden"],
            "non_striker": ["JL Langer"],
            "runs": [1],
            "batter_runs": [0],
            "extras": [1],
            "bowler": ["C White"],
            "wides": [0],
            "noballs": [1],
            "byes": [0],
            "legbyes": [0],
            "penalty": [0],
            "player_out_1": [""],
            "kind_1": [""],
            "player_out_2": [""],
            "kind_2": [""],
        }
    )
    assert_frame_equal(no_ball_output, expected_no_ball_output)
    leg_ball_output = leg_bye.get_ball_data()
    expected_leg_ball_output = pl.DataFrame(
        {
            "batter": ["ML Hayden"],
            "non_striker": ["JL Langer"],
            "runs": [1],
            "batter_runs": [0],
            "extras": [1],
            "bowler": ["AR Caddick"],
            "wides": [0],
            "noballs": [0],
            "byes": [0],
            "legbyes": [1],
            "penalty": [0],
            "player_out_1": [""],
            "kind_1": [""],
            "player_out_2": [""],
            "kind_2": [""],
        }
    )
    assert_frame_equal(leg_ball_output, expected_leg_ball_output)
    bye_output = bye.get_ball_data()
    expected_bye_output = pl.DataFrame(
        {
            "batter": ["ME Trescothick"],
            "non_striker": ["MA Butcher"],
            "runs": [2],
            "batter_runs": [0],
            "extras": [2],
            "bowler": ["SCG MacGill"],
            "wides": [0],
            "noballs": [0],
            "byes": [2],
            "legbyes": [0],
            "penalty": [0],
            "player_out_1": [""],
            "kind_1": [""],
            "player_out_2": [""],
            "kind_2": [""],
        }
    )
    assert_frame_equal(bye_output, expected_bye_output)
    penalty_output = penalty.get_ball_data()
    expected_penalty_output = pl.DataFrame(
        {
            "batter": ["Habibul Bashar"],
            "non_striker": ["Hannan Sarkar"],
            "runs": [5],
            "batter_runs": [0],
            "extras": [5],
            "bowler": ["MJ Hoggard"],
            "wides": [0],
            "noballs": [0],
            "byes": [0],
            "legbyes": [0],
            "penalty": [5],
            "player_out_1": [""],
            "kind_1": [""],
            "player_out_2": [""],
            "kind_2": [""],
        }
    )
    assert_frame_equal(penalty_output, expected_penalty_output)
