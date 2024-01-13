import json
from pathlib import Path

import pytest

from cricket.match_processing import Match

TEST_DATA_DIR = Path(__file__).parent.joinpath("test_data")


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
    return Match(
        TEST_DATA_DIR.joinpath("hundred_match_input.json"), match_id=4
    )


@pytest.fixture
def test_forfeitted_match():
    return Match(
        TEST_DATA_DIR.joinpath("innings_forfeit_match_input.json"), match_id=5
    )


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
        "dates": [
            "2002-12-26",
            "2002-12-27",
            "2002-12-28",
            "2002-12-29",
            "2002-12-30",
        ],
        "teams": ["Australia", "England"],
        "toss": {"decision": "bat", "winner": "Australia"},
    }
    assert test_odi_match.get_match_metadata() == {
        "match_id": 2,
        "match_type": "ODI",
        "city": "Darwin",
        "venue": "Marrara Cricket Ground, Darwin",
        "balls_per_over": 6,
        "gender": "male",
        "dates": ["2003-08-06"],
        "teams": ["Australia", "Bangladesh"],
        "toss": {"decision": "bat", "winner": "Australia"},
    }
    assert test_t20_match.get_match_metadata() == {
        "match_id": 3,
        "match_type": "T20",
        "city": "Windhoek",
        "venue": "United Cricket Club Ground, Windhoek",
        "balls_per_over": 6,
        "gender": "female",
        "dates": ["2023-04-27"],
        "teams": ["Hong Kong", "Uganda"],
        "toss": {"decision": "bat", "winner": "Hong Kong"},
    }
    assert test_hundred_match.get_match_metadata() == {
        "match_id": 4,
        "match_type": "T20",
        "city": "Cardiff",
        "venue": "Sophia Gardens, Cardiff",
        "balls_per_over": 5,
        "gender": "male",
        "dates": ["2021-08-18"],
        "teams": ["London Spirit", "Welsh Fire"],
        "toss": {"decision": "field", "winner": "Welsh Fire"},
    }
    assert test_forfeitted_match.get_match_metadata() == {
        "match_id": 5,
        "match_type": "MDM",
        "city": "Southampton",
        "venue": "The Rose Bowl, Southampton",
        "balls_per_over": 6,
        "gender": "male",
        "dates": ["2021-05-19", "2021-05-20", "2021-05-21", "2021-05-22"],
        "teams": ["Hampshire", "Leicestershire"],
        "toss": {"decision": "field", "winner": "Leicestershire"},
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
    with open(
        TEST_DATA_DIR.joinpath("innings_forfeit_match_output.json")
    ) as f:
        assert test_forfeitted_match.parse_match_data() == json.load(f)
