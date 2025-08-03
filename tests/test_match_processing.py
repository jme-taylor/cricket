import datetime
import json
from pathlib import Path

import pytest

from cricket.extraction.match_processing import Match

TEST_DATA_DIR = Path(__file__).parent.joinpath("test_match_processing")


@pytest.fixture
def test_test_match():
    return Match(TEST_DATA_DIR.joinpath("test_match_input.json"), match_id=1)


@pytest.fixture
def test_odi_match():
    return Match(TEST_DATA_DIR.joinpath("odi_match_input.json"), match_id=2)


@pytest.fixture
def test_t20_match():
    return Match(TEST_DATA_DIR.joinpath("t20_match_input.json"), match_id=3)


@pytest.fixture
def test_hundred_match():
    return Match(TEST_DATA_DIR.joinpath("hundred_match_input.json"), match_id=4)


@pytest.fixture
def test_forfeitted_match():
    return Match(TEST_DATA_DIR.joinpath("innings_forfeit_match_input.json"), match_id=5)


def test_lookup_player(test_test_match):
    assert test_test_match.lookup_player("AC Gilchrist") == "2b6e6dec"
    assert test_test_match.lookup_player("David Beckham") is None


def test_get_match_metadata(
    test_test_match,
    test_odi_match,
    test_t20_match,
    test_hundred_match,
    test_forfeitted_match,
):
    assert test_test_match.get_match_metadata() == {
        "match_id": 1,
        "match_type": "Test",
        "city": "Melbourne",
        "venue": "Melbourne Cricket Ground",
        "balls_per_over": 6,
        "gender": "male",
        "start_date": datetime.datetime(2002, 12, 26),
        "end_date": datetime.datetime(2002, 12, 30),
        "team_1": "Australia",
        "team_2": "England",
        "toss_winner": "Australia",
        "toss_decision": "bat",
    }
    assert test_odi_match.get_match_metadata() == {
        "match_id": 2,
        "match_type": "ODI",
        "city": "Darwin",
        "venue": "Marrara Cricket Ground, Darwin",
        "balls_per_over": 6,
        "gender": "male",
        "start_date": datetime.datetime(2003, 8, 6),
        "end_date": datetime.datetime(2003, 8, 6),
        "team_1": "Australia",
        "team_2": "Bangladesh",
        "toss_winner": "Australia",
        "toss_decision": "bat",
    }
    assert test_t20_match.get_match_metadata() == {
        "match_id": 3,
        "match_type": "T20",
        "city": "Windhoek",
        "venue": "United Cricket Club Ground, Windhoek",
        "balls_per_over": 6,
        "gender": "female",
        "start_date": datetime.datetime(2023, 4, 27),
        "end_date": datetime.datetime(2023, 4, 27),
        "team_1": "Hong Kong",
        "team_2": "Uganda",
        "toss_winner": "Hong Kong",
        "toss_decision": "bat",
    }
    assert test_hundred_match.get_match_metadata() == {
        "match_id": 4,
        "match_type": "T20",
        "city": "Cardiff",
        "venue": "Sophia Gardens, Cardiff",
        "balls_per_over": 5,
        "gender": "male",
        "start_date": datetime.datetime(2021, 8, 18),
        "end_date": datetime.datetime(2021, 8, 18),
        "team_1": "London Spirit",
        "team_2": "Welsh Fire",
        "toss_winner": "Welsh Fire",
        "toss_decision": "field",
    }
    assert test_forfeitted_match.get_match_metadata() == {
        "match_id": 5,
        "match_type": "MDM",
        "city": "Southampton",
        "venue": "The Rose Bowl, Southampton",
        "balls_per_over": 6,
        "gender": "male",
        "start_date": datetime.datetime(2021, 5, 19),
        "end_date": datetime.datetime(2021, 5, 22),
        "team_1": "Hampshire",
        "team_2": "Leicestershire",
        "toss_winner": "Leicestershire",
        "toss_decision": "field",
    }


def test_parse_match_data(
    test_test_match,
    test_odi_match,
    test_t20_match,
    test_hundred_match,
    test_forfeitted_match,
):
    with open(TEST_DATA_DIR.joinpath("test_match_output.json")) as f:
        assert test_test_match.parse_match_data() == json.load(f)
    with open(TEST_DATA_DIR.joinpath("odi_match_output.json")) as f:
        assert test_odi_match.parse_match_data() == json.load(f)
    with open(TEST_DATA_DIR.joinpath("t20_match_output.json")) as f:
        assert test_t20_match.parse_match_data() == json.load(f)
    with open(TEST_DATA_DIR.joinpath("hundred_match_output.json")) as f:
        assert test_hundred_match.parse_match_data() == json.load(f)
    with open(TEST_DATA_DIR.joinpath("innings_forfeit_match_output.json")) as f:
        assert test_forfeitted_match.parse_match_data() == json.load(f)
