import json
from pathlib import Path

import pytest

from cricket.innings_processing import Innings

TEST_DATA_DIR = Path(__file__).parent.joinpath("test_innings_processing")


@pytest.fixture
def test_innings():
    with open(TEST_DATA_DIR.joinpath("innings_input.json")) as f:
        return Innings(json.load(f), 1)


@pytest.fixture
def test_innings_with_powerplay():
    with open(
        TEST_DATA_DIR.joinpath("innings_with_powerplay_input.json")
    ) as f:
        return Innings(json.load(f), 1)


@pytest.fixture
def test_innings_with_target():
    with open(TEST_DATA_DIR.joinpath("innings_with_target_input.json")) as f:
        return Innings(json.load(f), 1)


def test_powerplay_check(
    test_innings, test_innings_with_powerplay, test_innings_with_target
):
    ball = {"delivery": 3.2}
    test_innings.power_play_check(ball)
    assert not ball["powerplay"]
    ball = {"delivery": 2.3}
    test_innings_with_powerplay.power_play_check(ball)
    assert ball["powerplay"]
    ball = {"delivery": 10.2}
    test_innings_with_powerplay.power_play_check(ball)
    assert not ball["powerplay"]
    ball = {"delivery": 17.3}
    test_innings_with_target.power_play_check(ball)
    assert not ball["powerplay"]


def test_target_check(
    test_innings, test_innings_with_powerplay, test_innings_with_target
):
    ball = {"delivery": 3.2}
    test_innings.target_check(ball)
    assert ball["target_runs"] == 0
    assert ball["target_overs"] == 0.0
    ball = {"delivery": 2.3}
    test_innings_with_powerplay.target_check(ball)
    assert ball["target_runs"] == 0
    assert ball["target_overs"] == 0.0
    ball = {"delivery": 10.2}
    test_innings_with_target.target_check(ball)
    assert ball["target_runs"] == 186
    assert ball["target_overs"] == 20


def test_parse_innings_data(
    test_innings, test_innings_with_powerplay, test_innings_with_target
):
    with open(TEST_DATA_DIR.joinpath("innings_output.json")) as f:
        assert test_innings.parse_innings_data() == json.load(f)
    with open(
        TEST_DATA_DIR.joinpath("innings_with_powerplay_output.json")
    ) as f:
        assert test_innings_with_powerplay.parse_innings_data() == json.load(f)
    with open(TEST_DATA_DIR.joinpath("innings_with_target_output.json")) as f:
        assert test_innings_with_target.parse_innings_data() == json.load(f)
